"""The simplest thing that works: load a WAV, plot it.

    python wav_plot.py mario_style_full.wav
"""
import sys, wave
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "mario_style_full.wav"

with wave.open(path, "rb") as w:
    rate = w.getframerate()                                  # samples per second
    data = np.frombuffer(w.readframes(w.getnframes()), "<i2")  # 16-bit -> array

seconds = np.arange(len(data)) / rate       # x axis: sample number -> time

plt.figure(figsize=(10, 3))
plt.plot(seconds, data, linewidth=0.5)
plt.xlabel("seconds")
plt.ylabel("amplitude")
plt.title(path)
plt.tight_layout()
plt.show()
