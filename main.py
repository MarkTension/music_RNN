import os
import yaml
import os

from sample import sample
from dataset import Dataclass
from train import train, load_model


"""
- check this out: https://www.tensorflow.org/tutorials/audio/music_generation
"""


# TODO:
# - refactor code. make training etc easier
#    - checkpoint loading
# - experiment with more models
#     - using last output of rnn only
#     - using pitch normalization instead of embedding
#     - check if timing data actually works
#     - 



def main(config):

  print('creating data')
  dataLoader = Dataclass(dotdict(config.data_details))
  training_data, valid_data = dataLoader.get_training_data()

  model = None
  # The embedding dimension
  if (not config.sampling_mode):
    print('starting training')
    model = train(training_data, valid_data, config)
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