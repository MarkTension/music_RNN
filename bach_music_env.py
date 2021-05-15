import numpy as np
import os

import tensorflow as tf
import scipy
"""
- The MIDI pitches liein the range from MIDI number 36 to 81
- We quantize time into sixteenthnote,
- For notes with a longer duration, we use “hold” symbols to encode their length
    - hold uses a tuple to encode a hold: [(C4, 0), (C4, 1), (C4, 1), (C4, 1)]
- RESTS are encode with a rest symbol
"""

from training import Teacher_model
from midi import GenerateMidiFile
from data_processing import make_dataset

# get all csv's into dataframes for preprocessing
filepath = "./data/jsb_chorales/"
run_name = "newPoo"
checkpoint_root = './training_checkpoints'
checkpoint_dir = f'{checkpoint_root}/{run_name}'
num_epochs = 5

do_training = True


def train_model(model, dataset, validation_dataset):

  # Directory where the checkpoints will be saved
  if not os.path.exists(f'{checkpoint_dir}'):
    os.makedirs(f'{checkpoint_dir}')

  # Name of the checkpoint files
  checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{num_epochs}")

  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=False)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs") # , profile_batch=2

  print('training')
  history = model.fit(dataset,
                      epochs=num_epochs,
                      callbacks=[checkpoint_callback, tensorboard_callback],
                      validation_data=validation_dataset)

  return model

def functione(input_array):
  return tf.random.categorical(input_array, num_samples=1)


def get_new_time(input):
  # todo: just do 1 to 16 instead of 1 to 4
  first = input[0]
  # if all equal
  if (all(x==first for x in input) and first == 3): # and input[0] != 3
    return 0#input[-1]+1
  elif (all(x == first for x in input)):
    return input[-1]+1
  else:
    return input[-1]

def test_model(model, dataset, greedy = False):

  outall = np.zeros(shape=(1, 4))
  input_example_batch = dataset.take(1)
  live_batch = input_example_batch.as_numpy_iterator().next()[0]
  final_batch = input_example_batch.as_numpy_iterator().next()[0]

  live_batch = tf.expand_dims(live_batch[0], axis=0)
  final_batch = tf.expand_dims(final_batch[0], axis=0)
  states = None
  for i in range(400):
    # sample new note
    logits = tf.expand_dims(model(live_batch)[0], axis=0)
    if (greedy):
      # samples = tf.expand_dims(tf.math.argmax(logits[0], axis=1), axis=1)
      top_k = tf.math.top_k(logits[0], k=5)
      # indices = tf.random.categorical(top_k[0], num_samples=1)
      # zeros = tf.zeros(shape=[4], dtype=tf.int64)
      samples = tf.random.categorical(top_k[0], num_samples=1)

    else:
      samples = tf.random.categorical(logits[0], num_samples=1)
    samples = tf.transpose(samples)

    # get time step number
    timestep = get_new_time(live_batch[:,-4:,-1][0])
    timestep = tf.expand_dims(tf.expand_dims(tf.constant(int(timestep), dtype=tf.int64), axis=0), axis=0)
    samples = tf.concat([samples, timestep], axis=1)
    samples = samples.numpy()
    live_batch = np.concatenate([live_batch, np.expand_dims(samples, axis=0)], axis=1)
    final_batch = np.concatenate([final_batch, np.expand_dims(samples, axis=0)], axis=1)

    # remove oldest item
    live_batch = np.delete(live_batch, 0, axis=1)

  final_batch = np.delete(final_batch, list(range(64)), axis=1)
  return final_batch

training_data, validation_dataset = make_dataset(filepath)

model = Teacher_model(
      # Be sure the vocabulary size matches the `StringLookup` layers.
      vocab_size=13, #81-35 + 1,
      embedding_dim=16,
      rnn_units=200)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])

model(training_data.as_numpy_iterator().next()[0])

if (do_training):
  model = train_model(model, training_data, validation_dataset)
  model.save(f"training_checkpoints/{run_name}")
else:
  model = tf.keras.models.load_model(f"training_checkpoints/{run_name}")

# generate notes
choices = test_model(model, training_data, greedy=True)

# set back to original tone height
choices = np.where(choices==0,-36,choices)
choices = choices + 36

GenerateMidiFile(choices[0], run_name, False)

# trash

# if (df.min().min() == 0):
  #   zero_count+=1
  #   print(f'zero at {chorale}')
  # else:
  #   minval = min(df.min().min(), minval)
