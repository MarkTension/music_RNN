from dataclasses import dataclass
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import os
from midi_utils import get_midi_from_numbers, tranpose_midi_notes
import json
from mido import MidiFile

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
      return tf.data.experimental.load(data_out)
    else:
      print("making new data")

      # create midi dir
      path = os.path.join(self.data_details.out, self.data_details.version)
      os.makedirs(path)
      os.makedirs(os.path.join(path, 'transposed'))
      # create data
      training_data = self._create_dataset()

      # save dataset
      tf.data.experimental.save(training_data, data_out)

      return training_data


  def _create_dataset(self):

    # load data
    chorales_dfs = self._load_data(self.data_details.raw)
    training_data = []

    # load transpositions
    transpositions_path = os.path.join(self.data_details.out, "midi_analysis.npy")
    if (os.path.exists(transpositions_path)):
      transpositions = np.load(transpositions_path, allow_pickle=True)
    else:
      transpositions = np.array([None]*len(chorales_dfs))

    for i, df in enumerate(chorales_dfs):
      # normalize data
      df, empty_notes_df = self._normalize_data(df)
      # make midi from data
      midi_file = get_midi_from_numbers(df.to_numpy(), self.data_details.version)
      # save midi data to get read by music21 fors transposition
      midi_file_name = f"{self.data_details.out}/{self.data_details.version}/{i}.mid"
      # transpose midi notes
      df, transpositions[i] = tranpose_midi_notes(midi_file_name, midi_file, df, transpositions[i])
      # normalize to one octave again after transpose
      # refill empty notes after transpose
      df[empty_notes_df == np.nan] = self.data_details.token_empty_note
      # set tempo
      df = self._set_tempo_data(df)
      # make sequential
      df = self._make_sequential(df)

      training_data.append(df)

    # save for future reference
    if (not os.path.exists(transpositions_path)):
      np.save(transpositions_path, transpositions)

    training_data = pd.concat(training_data)
    training_data = training_data.to_numpy()
    tf_dataset = self._create_tf_data(training_data)

    return tf_dataset

  # def _load_data(self, datapath):
  #     files = os.listdir(datapath)
  #     dfs = []
  #     for chorale in files:
  #       dfs.append(pd.read_csv(f'{datapath}/{chorale}'))
  #     return dfs

  def _load_data(self, datapath):

    # Opening JSON file
    with open(datapath) as json_file:
        data = json.load(json_file)

    dfs = []

    for chorale in data['train']:

      voice0 = []
      voice1 = []
      voice2 = []
      voice3 = []

      for row in chorale:
        if (len(row) == 4):
          voice0.append(row[0])
          voice1.append(row[1])
          voice2.append(row[2])
          voice3.append(row[3])

      df = pd.DataFrame({"note0": voice0, "note1": voice1, "note2": voice2, "note3": voice3})
      dfs.append(df)

    return dfs

  def _normalize_data(self, df):
    """
    here we normalize every note to one octave. 
    Also locations of empty notes are returned. 
    These are temporarily filled with root notes for transpose program. Later they're assigned the proper note
    """
    # squeeze to octave. Ignore negatives, these are rests
    # df[df >= 0 ] = df % 12

    # save locations of empty notes
    empty_notes_df = df[df == -1]
    
    # temporarilty set empty notes to the most common note.
    mode = df['note0'].mode()[0]
    df = df.replace(-1, mode)

    return df, empty_notes_df

  def _make_sequential(self, df):
    
    notes = []
    timings = []

    for _, row in df.iterrows():

      timings.extend([row.time, 0, 0, 0]) # row.time is just 1s
      notes.append(row.note0)
      notes.append(row.note2)
      notes.append(row.note1)
      notes.append(row.note3)

    df = pd.DataFrame({"notes": notes, "timings": timings})
    return df

  def _set_tempo_data(self, df):
    
    # add time column
    df['time'] = (df.index) % 16
    df['time'] = 1 #df['time'].map(lambda x: map_to_fourths(x)) + 1
    
    # add start and end
    # df_reset = pd.DataFrame(np.full(fill_value=self.data_details.token_reset_song,shape=(1, df.shape[1])))
    # df_reset.columns = df.columns
    # df_reset['time'] = 2
    # df = pd.concat([df_reset, df], axis=0)
    
    return df


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
      target_text = sequence[-1:]
      return input_text, target_text

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .batch(BATCH_SIZE, drop_remainder=True)
        .shuffle(BUFFER_SIZE)
        .cache()
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

  