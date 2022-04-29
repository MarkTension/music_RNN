# making music with RNNs
Playing around with RNN's to generate music in the style of BACH's chorale's.
Bach's chorale dataset is challenging because it's quite small

### the idea
The thing currently implemented is an experiment to see what happens when each of the 4 voices is generated by a different RNN. 
They all get the same 4-voice input though.
It's like they're all improvising together, because the music is not scripted, each model knows the statistics of the song, but not what the other models will do.

run via 
python main.py