from mido import Message, MidiFile, MidiTrack
from music21 import *
import music21
import os

def get_midi_from_numbers(source, version):
  """
  take dataframe (source), and returns midi file. called for creating training data, and after inference.
  """

  timestep = 4 # this is the time delta per note
  mid = MidiFile()
  mid.type = 1
  voices = []

  # for each voice make a midi track
  for voice in range(4):
    track = MidiTrack()
    track.name = f"voice{voice}_{version}"
    track.append(Message('program_change', program=12, time=timestep))
    # append to vioces list
    voices.append(track)
    # append to full midi
    mid.tracks.append(track)

  actionKindPrevious = [-1,-1,-1,-1]
  notelength = [0,0,0,0]

  for i, event in enumerate(source):
    for voice in range(4):
      # generate individual actions.
      # if new action. Create previous note of length 'notelength'
      new_note = event[voice] != actionKindPrevious[voice] and i > 0
      if (new_note):
        # end previous note
        voices[voice].append(Message('note_off', note=int(event[voice]), velocity=80, time=notelength[voice]*60))
        notelength[voice]=0 # reset note
        # start new note
        voices[voice].append(Message('note_on', note=int(event[voice]), velocity=80, time=0))

      notelength[voice] += timestep
      actionKindPrevious[voice] = event[voice]

  return mid


def tranpose_midi_notes(midi_file_name):

  # major conversions
  majors = dict(
    [("A-", 4), ("A", 3), ("B-", 2),("A+", 2), ("B", 1), ("C", 0), ("D-", -1),("C+", -1), ("D", -2), ("E-", -3),("D+", -3), ("E", -4), ("F", -5),
    ("G-", 6), ("G", 5)])
  minors = dict(
    [("A-", 1), ("A", 0), ("B-", -1), ("B", -2), ("C", -3), ("D-", -4), ("D", -5), ("E-", 6), ("E", 5), ("F", 4),
    ("G-", 3), ("G", 2)])

  score = music21.converter.parse(midi_file_name)
  key = score.analyze('key')
  # print key.tonic.name, key.mode
  if key.mode == "major":
    halfSteps = majors[key.tonic.name]

  elif key.mode == "minor":
    halfSteps = minors[key.tonic.name]
  else:
    raise ValueError("music21 error")

  newscore = score.transpose(halfSteps)
  key = newscore.analyze('key')
  newFileName = "C_" + midi_file_name
  newscore.write('midi', newFileName)