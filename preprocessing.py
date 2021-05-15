import pandas as pd
import os
import numpy as np

from midi import GenerateMidiFile


def map_to_fourths(n):
  if (n < 4):
    return 0
  if (n < 8):
    return 1
  if (n < 12):
    return 2
  else:
    return 3


def squeeze_to_octave(x):

  for i in range(len(x)):
    if (x[i] == 0):
      x[i] = 12
    else:
      x[i] = x[i] % 12

  return x


def load_and_normalize(filepath):
  import matplotlib
  matplotlib.use("TkAgg")
  from matplotlib import pyplot as plt

  files = os.listdir(filepath)
  dfs = []

  lowest = [100, 100, 100, 100]
  highest = [0, 0, 0, 0]

  for i, chorale in enumerate(files):
    df = pd.read_csv(f'{filepath}/{chorale}')

    # GenerateMidiFile(df.to_numpy(), name=f"{i}", preprocessing=True)

    # df.note0 -= 12

    highest[0] = max(highest[0], df.note0.max())
    highest[1] = max(highest[1], df.note1.max())
    highest[2] = max(highest[2], df.note2.max())
    highest[3] = max(highest[3], df.note3.max())

    df.apply(lambda x: squeeze_to_octave(x))

    # lowest note is 36. Let that be 1. 0 will then remain
    # df = df - 35
    # df = df.replace(-35, 0)
    df['time'] = (df.index) % 4
    df['time'] = df['time'].map(lambda x: map_to_fourths(x))
    df2 = pd.DataFrame(np.full(fill_value=0,shape=(1, df.shape[1])))
    df2.columns = df.columns
    df2['time'] = 5
    combined = pd.concat([df2, df])


    dfs.append(combined)

    # dfs[1].note1.plot.hist()

  return dfs