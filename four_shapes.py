"""Every waveform shape the chip can make, all at C4.

Writes two figures:
    c4_two_shapes.png  - a sine vs this chip's pulse, same note
    four_shapes.png    - all four channels
"""
import numpy as np
import matplotlib.pyplot as plt
from nes_apu import (SR, note_hz, pulse_period, tri_period,
                     gen_pulse, gen_triangle, gen_noise)

hz = note_hz("C4")
n  = int(0.015 * SR)          # 15 ms
t  = np.arange(n) / SR * 1000
F  = 3                        # gen_* take per-frame arrays, 735 samples each

pulse25 = gen_pulse(np.full(F, pulse_period(hz), float), np.full(F, 15.0), 1)[:n]
pulse50 = gen_pulse(np.full(F, pulse_period(hz), float), np.full(F, 15.0), 2)[:n]
triangle = gen_triangle(np.full(F, tri_period(hz), float), np.ones(F))[:n]
noise = gen_noise(np.full(F, 6), np.full(F, 15.0))[:n]
sine = np.sin(2 * np.pi * hz * np.arange(n) / SR)

# --- figure 1: same note, different shape ----------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(t, sine)
ax[0].set_title("C4 as a pure tone (a sine wave)")
ax[1].step(t, pulse25, color="tab:orange", where="post")
ax[1].set_title("C4 on this project's NES pulse channel")
for a in ax:
    a.set_ylabel("amplitude")
    a.axhline(0, color="gray", lw=.5)
    for k in range(5):                       # mark where each repeat begins
        a.axvline(k * 1000 / hz, color="red", ls=":", lw=1)
ax[1].set_xlabel("milliseconds")
fig.suptitle("same note, same repeat rate, different shape")
fig.tight_layout()
fig.savefig("c4_two_shapes.png", dpi=110)

# --- figure 2: the whole palette -------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for a, y, title, colour in [
        (ax[0], pulse25,  "PULSE 1 & 2  (the melody)  —  rectangular", "tab:orange"),
        (ax[1], pulse50,  "same pulse, 50% duty  —  still rectangular, wider", "tab:orange"),
        (ax[2], triangle, "TRIANGLE  (the bass)  —  a staircase, NOT rectangular", "tab:green"),
        (ax[3], noise,    "NOISE  (the drums)  —  no repeating shape at all", "tab:purple")]:
    a.step(t, y, color=colour, lw=1.4, where="post")
    a.set_title(title, fontsize=10, loc="left")
    a.set_ylabel("0-15")
ax[3].set_xlabel("milliseconds")
fig.suptitle("every shape this project can make (all at C4)")
fig.tight_layout()
fig.savefig("four_shapes.png", dpi=110)
print("wrote c4_two_shapes.png and four_shapes.png")
