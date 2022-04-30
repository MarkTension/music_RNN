
import time
# import time
import tensorflow as tf
from model import MusicRNN, OneStep
from midi_utils import get_midi_from_numbers
import os


def train(training_data,  config):

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
  models = []

  for i in range(4):
    models.append(MusicRNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        rnn_units=config.rnn_units))

    # models[i].compile(optimizer='adam', loss=loss)
    models[i].compile(optimizer='RMSprop', loss=loss)

  # Directory where the checkpoints will be saved
  checkpoint_dir = f'./results/{config.run_name}/training_checkpoints/'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
  mean = tf.metrics.Mean()

  # training loop
  for epoch in range(config.epochs):
      start = time.time()

      mean.reset_states()
      for (batch_n, (inp, target)) in enumerate(training_data):
          
          temp_tensor = a = tf.zeros([32,1,4], dtype=tf.int64)
          inp_new = tf.concat([inp, temp_tensor], axis=1)
          for i in range(4):
            one_voice_target = target[:, :, i]
            logs = models[i].train_step([inp, one_voice_target])
            mean.update_state(logs['loss'])

          if batch_n % 50 == 0:
              template = f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
              print(template)

      # saving (checkpoint) the model every 5 epochs
      if (epoch + 1) % 5 == 0:
        for model in models:
          model.save_weights(checkpoint_prefix.format(epoch=epoch))

      print(f'Epoch {epoch+1} Loss: {mean.result().numpy():.4f}')
      print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
      print("_"*80)

  return models


def sample(models, config):

  one_step_models = []
  for model in models:
    one_step_models.append(OneStep(model))
  states = [None, None, None, None]
  next_char = tf.constant([[[4, 1, 1, 1]]], dtype=tf.int64)
  result = [next_char]

  for _n in range(400):
      pred0, states[0] = one_step_models[0].generate_one_step(
          next_char, states=states[0])
      pred1, states[1] = one_step_models[1].generate_one_step(
          next_char, states=states[1])
      pred2, states[2] = one_step_models[2].generate_one_step(
          next_char, states=states[2])
      pred3, states[3] = one_step_models[3].generate_one_step(
          next_char, states=states[3])
      next_char = tf.reshape(
          tf.stack([pred0, pred1, pred2, pred3], axis=0), shape=[1, 1, 4])

      result.append(next_char)

  result = tf.concat(result[1:], axis=1)
  result = result.numpy()[0]
  midi = get_midi_from_numbers(result, config.data_details['version'])

  midi_out_dir = os.path.join("results", config.run_name, f"midi_{config.epochs}_epochs.mid")
  # midi.save("test.mid")
  midi.save(midi_out_dir)     
