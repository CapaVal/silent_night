# silent_night
Record your night with a smartphone, get your phonic highlights in the morning.

The sound analysis is performed on .wav file. If your smartphone generates ".m4a" files, then a conversion will be performed to generate the ".wav" file before processing. In the latter case, you need ffmpeg.bin in your environnement (see https://ffmpeg.org/).

You may modify some processing parameters depending on your need, especially the sampling period to detect any event (snoring, teeth grinding, neighbours...).

The programm will generate a graph representing the sound amplitude vs time to highlight the most noisy part of the night. An event type detection algorithm would be nice to automatically label each sound presence.

Enjoy.
