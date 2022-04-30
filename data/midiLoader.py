import pandas as pd
import numpy as np
from os import listdir
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class Midi_loader:
  """
  Data loader generates the stream of midi input
  logic:
  - if stream is empty, load new csv
  - 


  """
  def __init__(self):
    self.stream_pointer = 0
    self.window_size = 20
    self.train_dir = "./midiData/jsb_chorales/train"
    self.files = listdir(self.train_dir)
    self.numFiles = self.files.__len__()



  def stream_midi(self):

    yield self.chorale[self.stream_pointer : self.stream_pointer + self.window_size]

    self.stream_pointer += 1


  def sample_random_csv(self):

    # sample random number
    sample = np.random.choice(self.files)

    chorale_sample = np.genfromtxt(
      f"{self.train_dir}/{sample}", delimiter=',', )

    # remove NaN's
    a = np.where(np.isfinite(chorale_sample), chorale_sample, 0)
    # add padding
    a = np.pad(a, ((self.window_size, self.window_size), (0, 0)))

    self.chorale = a