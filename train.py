
import time
import tensorflow as tf
from model import MusicRNN
from midi_utils import get_midi_from_numbers
import os
import numpy as np
import pandas as pd

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

        if batch_n % 20 == 0:
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

  for _n in range(1000):
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

  # save to json
  df = pd.DataFrame({"note0": result[0], "note1": result[1], "note2": result[2], "note3": result[3]})
  json_out_dir = os.path.join("results", config.run_name, f"_{config.epochs}_epochs.json")

  df.to_json(json_out_dir)


  midi = get_midi_from_numbers(result.transpose(), config.data_details['version'])
  midi_out_dir = os.path.join("results", config.run_name, f"midi_{config.epochs}_epochs.mid")
  # midi.save("test.mid")
  midi.save(midi_out_dir)     


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
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Return the characters and model state.
    return predicted_ids, states