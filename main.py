import os
# from hmmlearn import hmm
import yaml
import numpy as np
import os

from dataset import Dataclass
from train import train, sample, load_model


"""
- The MIDI pitches liein the range from MIDI number 36 to 81
- We quantize time into sixteenthnote,
- For notes with a longer duration, we use “hold” symbols to encode their length
    - hold uses a tuple to encode a hold: [(C4, 0), (C4, 1), (C4, 1), (C4, 1)]
- RESTS are encode with a rest symbol
- check this out: https://www.tensorflow.org/tutorials/audio/music_generation
"""


# TODO:
# - refactor code. make training etc easier
#    - results directory
#    - data directory
#    - checkpoint loading
# - experiment with more models
#     - experiment with bigger models


def main(config):

  print('creating data')
  dataLoader = Dataclass(dotdict(config.data_details))
  training_data = dataLoader.get_training_data()

  # creat hmm model
  # model = hmm.GaussianHMM(n_components=12, covariance_type="full")
  # model.fit(training_data)
  # feature_matrix , state_sequence = model.sample(100)

  model = None
  # The embedding dimension
  if (not config.sampling_mode):
    print('starting training')
    model = train(training_data, config)
  if model == None:
    model = load_model(config, training_data)
  print('sampling, and generating midi')
  sample(model, config, training_data)

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

  # keep the old format to save for future ref
  config_dot = dotdict(config)

  # make dirs
  if (not os.path.exists(f"results/{config_dot.run_name}")):
    os.makedirs(f"results/{config_dot.run_name}")
    os.makedirs(f"results/{config_dot.run_name}/training_checkpoints")

  # save for future ref
  with open(f"results/{config_dot.run_name}/config.yaml", 'w') as file:
    documents = yaml.dump(config, file)

  # run scripts
  main(config_dot)