from dataclasses import dataclass
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import os
from midi_utils import get_midi_from_numbers, tranpose_midi_notes

import tensorflow as tf

class Dataclass:
  def __init__(self, data_details):
    self.data_details = data_details
    print('ínitialized data class')

  def get_training_data(self):
    midi_file = self._load_or_create_data()
    return midi_file

  def _load_or_create_data(self):
    data_out = os.path.join(self.data_details.out, self.data_details.version)

    if os.path.exists(data_out):
      # load data
      return self._create_dataset() #NotImplementedError("cannot load data yet")
    else:
      print("making new data")
      # create path 
      os.makedirs(data_out)
      # create data
      return self._create_dataset()


  def _create_dataset(self):

    # load data
    chorales_dfs = self._load_data(self.data_details.raw)
    training_data = []
    for i, df in enumerate(chorales_dfs):
      # normalize data
      df = self._normalize_data(df)
      # set tempo
      df = self._set_tempo_data(df)
      # make midi from data
      # midi_file = get_midi_from_numbers(df.to_numpy(), self.data_details.version)
      # # save midi data
      # midi_file_name = f"{self.data_details.out}/{self.data_details.version}/{i}.mid"
      
      # # for a preview:
      # if (i == 0):
      #   midi_file.save(midi_file_name)
      #   self._convert_to_json(df, self.data_details.version)

      # # transpose notes
      # midi_file = tranpose_midi_notes(midi_file_name)
      # save the data

      training_data.append(df)

    training_data = pd.concat(training_data)

    training_data.drop(['time'], axis=1, inplace=True)

    training_data = training_data.to_numpy()

    tf_dataset = self._create_tf_data(training_data)

    return tf_dataset

  def _load_data(self, datapath):
      files = os.listdir(datapath)
      dfs = []
      for chorale in files:
        dfs.append(pd.read_csv(f'{datapath}/{chorale}'))
      return dfs

  def _normalize_data(self, df):
    # squeeze to octave
    df = df % 12
    return df

  def _set_tempo_data(self, df):
    
    # add time column
    df['time'] = (df.index) % 16
    df['time'] = df['time'].map(lambda x: map_to_fourths(x))
    
    # add ending
    df2 = pd.DataFrame(np.full(fill_value=12,shape=(4, df.shape[1])))
    df2.columns = df.columns
    df2['time'] = 5
    combined = pd.concat([df2, df])
    
    return combined


  def _convert_to_json(self, df, version):
  
    example_df = df.drop(['time'], axis=1) + 36
    example_df.to_json(f"example_json_version_{version}.json")


  def _create_tf_data(self, training_data):

    # tf solution
    training_data_tf = tf.data.Dataset.from_tensor_slices(training_data)
    # example
    for ids in training_data_tf.take(10):
        print(ids)
    
    
    print(f"examples per epoch is {len(training_data)}")

    sequences = training_data_tf.batch(self.data_details.seq_length+1, drop_remainder=True)

    def split_input_target(sequence):
      input_text = sequence[:-1]
      target_text = sequence[1:]
      return input_text, target_text

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 32
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    return dataset

def map_to_fourths(n):
    if (n < 4):
      return 0
    if (n < 8):
      return 1
    if (n < 12):
      return 2
    else:
      return 3

  