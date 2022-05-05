
import time
import tensorflow as tf
from model import MusicRNN
from midi_utils import get_midi_from_numbers
import os
import numpy as np
import pandas as pd
from datetime import datetime
import keras 

def train(training_data,  config):

  # for some reason CPU is faster to train
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

  model = MusicRNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        rnn_units=config.rnn_units)

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps=5000,
    decay_rate=0.9)

  optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

  # models[i].compile(optimizer='adam', loss=loss)
  model.compile(optimizer=optimizer, loss=loss)

  # Directory where the checkpoints will be saved
  checkpoint_dir = f'./results/{config.run_name}/training_checkpoints/'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt") # _{epoch}
  checkpoint = tf.train.Checkpoint(model=model) # optimizer='RMSprop',

  mean = tf.metrics.Mean()

  for input_example_batch, target_example_batch in training_data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

  # model.load_weights(checkpoint_dir + "checkpoint_29.index")
  # training loop
  for epoch in range(config.epochs):
      start = time.time()

      mean.reset_states()
      for (batch_n, (inp, target)) in enumerate(training_data):
        
        logs = model.train_step([inp, target[:,:,0]])
        mean.update_state(logs['loss'])

        if batch_n % 20 == 0:
          template = f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
          print(template)

      # saving (checkpoint) the model every 5 epochs
      if (epoch + 1) % 5 == 0:
          # model.save_weights(checkpoint_prefix.format(epoch=epoch))
          checkpoint.save(checkpoint_prefix)


      print(f'Epoch {epoch+1} Loss: {mean.result().numpy():.4f}')
      print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
      print("_"*80)

  checkpoint.save(f"{checkpoint_prefix}")

  return model


def sample(model, config, training_data):

  timestamp = datetime.now().strftime("%m_%d_%H_%M")


  for temperature in [1, 2, 4, 8]:
    one_step_model = OneStep(model, temperature=temperature)
    state = None
    result = []

    # input = tf.constant([[[12, 5]]], dtype=tf.int64)
    dataset =  training_data.take(1)
    input = tf.constant(list(dataset.as_numpy_iterator())[0][0][:1])

    # sample loop
    for _n in range(config.num_notes_sampled):
      pred, state = one_step_model.generate_one_step(
          input, states=state)
      
      timing = tf.constant(np.floor((_n % 16)/4), dtype=tf.int64, shape=[1,1,1])
      prediction = tf.reshape(pred, shape=[1,1,1])
      prediction = tf.concat([prediction, timing], axis=2)
      input = tf.concat([input, prediction], axis=1)
      input = input[:,1:,:]
      result.append(pred)

    # reconstruct og format
    new = [[],[],[],[]]
    for i, el in enumerate(result):
        new[i % 4].extend(el.numpy())

    result = np.array(new)

    # save to json
    df = pd.DataFrame({"note0": result[0], "note1": result[1], "note2": result[2], "note3": result[3]})
    json_out_dir = os.path.join("results", config.run_name, f"json_{timestamp}_epochs_{config.epochs}_temp_{temperature}.json")

    df.to_json(json_out_dir)

    midi = get_midi_from_numbers(result.transpose(), config.data_details['version'])
    midi_out_dir = os.path.join("results", config.run_name, f"midi_{timestamp}_epochs_{config.epochs}_temp_{temperature}.mid")
    # midi.save("test.mid")
    midi.save(midi_out_dir)


def load_model(config, training_data):

  checkpoint_dir = f'./results/{config.run_name}/training_checkpoints/'
  
  # make model
  model = MusicRNN(
      vocab_size=config.vocab_size,
      embedding_dim=config.embedding_dim,
      rnn_units=config.rnn_units)

  model.compile(optimizer='RMSprop', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

  # flush inputs through to make the graph
  for input_example_batch, target_example_batch in training_data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

  # load weights
  # model.load_weights(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  # status.assert_consumed()

  return model

class OneStep(tf.keras.Model):
  def __init__(self, model, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model

  def generate_one_step(self, inputs, states=None):

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=inputs, states=states,
                                          return_state=True, training=False)
    # Only use the last prediction.
    # predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Return the characters and model state.
    return predicted_ids, states

