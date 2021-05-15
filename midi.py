from mido import Message, MidiFile, MidiTrack

def GenerateMidiFile(source, name, preprocessing):

  timestep = 3 # there are three ticks per beat in general
  mid = MidiFile()
  mid.type = 1
  voices = []

  # for each voice make a midi track
  for id in range(4):
    track = MidiTrack()
    track.name = f"voice{id}_{name}"
    track.append(Message('program_change', program=12, time=timestep))
    # append to vioces list
    voices.append(track)
    # append to full midi
    mid.tracks.append(track)

  actionKindPrevious = [-1, -1, -1,-1]
  notelength = [0,0,0,0]

  for i, event in enumerate(source):
    for id in range(4):
      # generate individual actions.
      # # if new action. Create previous note of length 'notelength'
      if (event[id] != actionKindPrevious[id] and i > 0):
        voices[id].append(Message('note_on', note=int(event[id]), velocity=80, time=0))
        voices[id].append(Message('note_off', note=int(event[id]), velocity=80, time=notelength[id]*60))
        notelength[id]=0 # reset note

      notelength[id] += timestep

      actionKindPrevious[id] = event[id]

  if (preprocessing):
    mid.save(f'data/midi/{name}.mid')
  else:
    mid.save(f'MIDIout/{name}_23.mid')
