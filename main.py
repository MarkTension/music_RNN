import numpy as np
import os
from pip import main
import scipy
from hmmlearn import hmm
import yaml
from dataset import Dataclass
from model import MusicRNN, OneStep
from midi_utils import get_midi_from_numbers

import tensorflow as tf
import numpy as np
import os
import time


"""
- The MIDI pitches liein the range from MIDI number 36 to 81
- We quantize time into sixteenthnote,
- For notes with a longer duration, we use “hold” symbols to encode their length
    - hold uses a tuple to encode a hold: [(C4, 0), (C4, 1), (C4, 1), (C4, 1)]
- RESTS are encode with a rest symbol
"""


# TODO:
# - refactor code. make training etc easier
#    - results directory
#    - data directory
#    - checkpoint loading
# - experiment with more models
#     - add timing 
#     - add empty notes OG HAS NO EMPTY NOTES!!
#     - experiment with bigger models



def train(training_data,  config):

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
  models = []

  for i in range(4):
    models.append(MusicRNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        rnn_units=config.rnn_units))

    models[i].compile(optimizer='adam', loss=loss)

  # Directory where the checkpoints will be saved
  checkpoint_dir = f'./results/training_checkpoints/{config.run_name}'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

  mean = tf.metrics.Mean()

  for epoch in range(config.epochs):
      start = time.time()

      mean.reset_states()
      for (batch_n, (inp, target)) in enumerate(training_data):

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


def main(config):
    
  if (not os.path.exists(f"results/{config.run_name}")):
    os.makedirs(f"results/{config.run_name}")

  print('creating data')
  dataLoader = Dataclass(dotdict(config.data_details))
  training_data = dataLoader.get_training_data()

  # creat hmm model
  # model = hmm.GaussianHMM(n_components=12, covariance_type="full")
  # model.fit(training_data)
  # feature_matrix , state_sequence = model.sample(100)

  # The embedding dimension
  print('starting training')
  models = train(training_data, config)
  print('sampling, and generating midi')
  sample(models, config)

  # https://www.tensorflow.org/text/tutorials/text_generation


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":

    # load config
    with open("config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        print(config)

    config = dotdict(config)

    config['checkpoint_dir'] = os.path.join(
        config.checkpoint_root, config.run_name)

    # run scripts
    main(config)


# # set back to original tone height
# choices = np.where(choices==0,-36,choices)
# choices = choices + 36

# GenerateMidiFile(choices[0], run_name, False)


# # train each model on a different voice.
#   # they will be somewhat dependent on eachother
#   for input_example_batch, target_example_batch in training_data.take(1):
#     batch0 = input_example_batch #[:,:,0]
#     target_example_batch = target_example_batch #[:,:,0]
#     example_batch_predictions = model(batch0)
#     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

#     sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#     sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
