
import time
import tensorflow as tf
from model import MusicRNN, OneStep
from midi_utils import get_midi_from_numbers
import os
import numpy as np


def train(training_data,  config):

  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

  model = MusicRNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        rnn_units=config.rnn_units)

  # models[i].compile(optimizer='adam', loss=loss)
  model.compile(optimizer='RMSprop', loss=loss)

  # Directory where the checkpoints will be saved
  checkpoint_dir = f'./results/{config.run_name}/training_checkpoints/'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
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

        if batch_n % 50 == 0:
          template = f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
          print(template)

      # saving (checkpoint) the model every 5 epochs
      if (epoch + 1) % 5 == 0:
          model.save_weights(checkpoint_prefix.format(epoch=epoch))

      print(f'Epoch {epoch+1} Loss: {mean.result().numpy():.4f}')
      print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
      print("_"*80)

  return model


def sample(model, config):

  one_step_model = OneStep(model)
  state = None
  result = []

  input = tf.constant([[[12, 5]]], dtype=tf.int64)

  for _n in range(4000):
    pred, state = one_step_model.generate_one_step(
        input, states=state)
    
    timing = tf.constant(np.floor((_n % 16)/4), dtype=tf.int64, shape=[1,1,1])
    prediction = tf.reshape(pred, shape=[1,1,1])
    input = tf.concat([prediction, timing], axis=2)

    result.append(pred)

  # reconstruct og format
  new = [[],[],[],[]]
  for i, el in enumerate(result):
      
      new[i % 4].extend(el.numpy())

  result = np.array(new)
  midi = get_midi_from_numbers(result.transpose(), config.data_details['version'])
  midi_out_dir = os.path.join("results", config.run_name, f"midi_{config.epochs}_epochs222.mid")
  # midi.save("test.mid")
  midi.save(midi_out_dir)     
