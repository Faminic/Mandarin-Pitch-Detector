Three files in here to serve as an example.

example_audio.wav:
This is sample audio you could use for prototyping, the transcription is '锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。'
It is very clean audio, so you may want to also try with other audio.

pitch_methods.py:
This contains some sample algorithms that the other file uses.

example_testing_algorithm.ipynb:
You can make use of this to test your own algorithms.
It imports methods from pitch_methods.py. 
If you develop your own methods, you can just throw the .py file into the same folder as this (or in a subfolder) and import.
If you look at this file you'll see that after using the imported methods to estimate pitch, the next cell sets up a dictionary of what are essentially [time, frequency, weight] tuplets, and then the visualise_results method I imported will iterate over the dictionary to plot over a spectrogram.
