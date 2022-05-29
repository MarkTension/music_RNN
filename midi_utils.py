from mido import Message, MidiFile, MidiTrack
from music21 import *
import music21
import os
import numpy as np
import pandas as pd

def get_midi_from_numbers(source, version, velocities=None):
  """
  It is beneficial to train the system with all files in the same key so that they create harmonies rather than discordance
  take dataframe (source), and returns midi file. called for creating training data, and after inference.
  
  music21 method adapted from http://nickkellyresearch.com/python-script-transpose-midi-files-c-minor/
  """

  timestep = 1 # this is the time delta per note
  mid = MidiFile()
  mid.type = 1
  voices = []

  timedelta = 120

  # normalize velocities
  if velocities is not None:
    velocities = np.minimum(np.floor((velocities / np.max(velocities)) * 128).astype(np.int32),127)


  # for each voice make a midi track
  for voice in range(4):
    track = MidiTrack()
    track.name = f"voice{voice}_{version}"
    track.append(Message('program_change', program=12, time=timestep))
    # append to vioces list
    voices.append(track)
    # append to full midi
    mid.tracks.append(track)

  # go through numbers
  for voice in range(4):
    actionKindPrevious = -1
    notelength=0
    for i, event in enumerate(source[:,voice]):
      if (i!= 0 and event == -1):
        print("-1 event")
      if (event!=actionKindPrevious):
        # if not first note, end previous note
        if (i != 0): 
          voices[voice].append(Message('note_off', note=np.int(actionKindPrevious), velocity=80, time=notelength*timedelta))
        # start new note
        if velocities is None:
          velocity = 80
        else:
          velocity = velocities[voice, i]
        voices[voice].append(Message('note_on', note=np.int(event), velocity=velocity, time=0))
        notelength=0
      # else:
      notelength+=1
      actionKindPrevious = event

      if (i == len(source) - 2 and notelength > 0):
        voices[voice].append(Message('note_off', note=np.int(event), velocity=80, time=(notelength+1)*timedelta)) 
      
      actionKindPrevious = event
      # notelength+=1

  return mid


#converting everything into the key of C major or A minor
majors = dict(
  [("A-", 4), ("A", 3), ("B-", 2),("A+", 2), ("B", 1), ("C", 0), ("D-", -1),("C+", -1), ("D", -2), ("E-", -3),("D+", -3), ("E", -4), ("F", -5),
  ("G-", 6), ("G", 5)])
minors = dict(
  [("A-", 1), ("A", 0), ("B-", -1), ("B", -2), ("C", -3), ("C#", -4), ("D-", -4), ("D", -5), ("E-", 6), ("E", 5), ("F", 4), ("F#", 3),
  ("G-", 3), ("G", 2)])


def tranpose_midi_notes(midi_file_name, midi_file, df, transposition):

  if transposition != None:
    halfStepsTranspose = transposition
  else:
    # save the midi file to load it with music21
    # music21 will be used for keysignature analysis and major/minor
    midi_file.save(midi_file_name)
    score = music21.converter.parse(midi_file_name)
    analysis = score.analyze('key')
    keysignature = analysis.tonic.name
    mode = analysis.mode
    
    # apply transpose based on key
    if mode == "major":
      halfStepsTranspose = majors[keysignature]
    elif mode == "minor":
      halfStepsTranspose = minors[keysignature]
    else:
      raise ValueError("music21 error")

  # after analysis, transpose it in pandas
  df = df + halfStepsTranspose

  return df, halfStepsTranspose