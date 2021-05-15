import tensorflow as tf
from preprocessing import load_and_normalize
import pandas as pd
import numpy as np
import itertools
from music21 import *
import music21
import glob

def make_shuffled_dataset(data):
  """
    converts 4 voices to 2
    makes all combinations and appends to dataset
  """
  indexlist = list(itertools.combinations([0, 1, 2, 3], 2))
  reversed_list = [list(reversed(element)) for element in indexlist]
  indexlist = indexlist + reversed_list

  all_data = []
  for idcs in indexlist:
    all_data.append(data[:,idcs])
  data = np.concatenate(all_data, axis=0)

  return data

def make_tf_dataset(data):

  window_size = 64 # window size of 2 means 1
  batch_size = 24

  def choose_indices(x):
    # random indices
    indices = tf.random.shuffle(tf.range(3))[:2]
    indices = tf.concat([indices, [4]], 0)
    return tf.gather(x, indices)

  dataset = tf.data.Dataset.from_tensor_slices(data)#.batch(16, drop_remainder=True)
  # dataset = tf.cast(dataset, dtype=tf.int64)
  # dataset = dataset.map(choose_indices)
  dataset = dataset.window(size=window_size, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size, drop_remainder=True))
  dataset = dataset.map(lambda window: (window[:-1], window[-1:, :4]))
  # dataset = dataset.map(split_input_target)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=True)

  return dataset


def convert_midis():
  # major conversions
  majors = dict(
    [("A-", 4), ("A", 3), ("B-", 2),("A+", 2), ("B", 1), ("C", 0), ("D-", -1),("C+", -1), ("D", -2), ("E-", -3),("D+", -3), ("E", -4), ("F", -5),
     ("G-", 6), ("G", 5)])
  minors = dict(
    [("A-", 1), ("A", 0), ("B-", -1), ("B", -2), ("C", -3), ("D-", -4), ("D", -5), ("E-", 6), ("E", 5), ("F", 4),
     ("G-", 3), ("G", 2)])

  # os.chdir("./")
  for file in glob.glob("data/midi/*.mid"):
    score = music21.converter.parse(file)
    key = score.analyze('key')
    #    print key.tonic.name, key.mode
    if key.mode == "major":
      halfSteps = majors[key.tonic.name]

    elif key.mode == "minor":
      halfSteps = minors[key.tonic.name]
    else:
      raise ValueError("music21 error")

    newscore = score.transpose(halfSteps)
    key = newscore.analyze('key')
    newFileName = "C_" + file
    newscore.write('midi', newFileName)


def make_dataset(filepath):

  # load files as midi
  # convert_midis()

  # preprocess files
  # EDA
  print("loading train")
  # train_data = load_and_normalize(filepath + "train")
  # print("loading valid")
  # valid = load_and_normalize(filepath + "valid")



  # train_data = pd.concat(train_data)
  # val_data = pd.concat(valid)

  # train_data.to_csv("./data/jsb_chorales/train.csv")
  # val_data.to_csv("./data/jsb_chorales/val.csv")

  train_data = pd.read_csv('data/jsb_chorales/train.csv', index_col=False)
  val_data = pd.read_csv('data/jsb_chorales/val.csv', index_col=False)

  train_data = train_data.drop('Unnamed: 0', axis=1)
  val_data = val_data.drop('Unnamed: 0', axis=1)
  # train_data = make_shuffled_dataset(train_data)
  # val_data = make_shuffled_dataset(val_data)

  dataset = make_tf_dataset(train_data)
  validation_dataset = make_tf_dataset(val_data)


  return dataset, validation_dataset