"""Draw the note C4 as a waveform."""
import numpy as np
import matplotlib.pyplot as plt

rate = 44100          # samples per second
freq = 261.63         # C4 (middle C) repeats 261.63 times per second

t = np.arange(0, 0.015, 1/rate)      # 15 milliseconds of time
y = np.sin(2 * np.pi * freq * t)     # the wave

plt.figure(figsize=(10, 3.5))
plt.plot(t * 1000, y)
plt.axhline(0, color="gray", lw=0.5)

period = 1000 / freq                 # one repeat, in milliseconds
plt.annotate("", xy=(period + 0.955, 1), xytext=(0.955, 1),
             arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
plt.text(period/2 + 0.955, 1.12, f"one repeat = {period:.2f} ms",
         color="red", ha="center")

plt.xlabel("milliseconds")
plt.ylabel("amplitude")
plt.title("C4 (middle C)  —  261.63 Hz")
plt.ylim(-1.3, 1.4)
plt.tight_layout()
plt.savefig("c4.png", dpi=110)
