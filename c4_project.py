"""Draw C4 exactly as this project's NES pulse channel produces it."""
import numpy as np
import matplotlib.pyplot as plt
from nes_apu import SR, note_hz, pulse_period, pulse_actual_hz, gen_pulse

want = note_hz("C4")                 # 261.63 Hz, the ideal C4
timer = pulse_period(want)           # the chip's integer timer value
got = pulse_actual_hz(timer)         # what it can actually produce

n = int(0.015 * SR)                  # 15 ms
y = gen_pulse(np.full(3, timer, float), np.full(3, 15.0), 1)[:n]   # duty 1 = 25%
t = np.arange(n) / SR * 1000

plt.figure(figsize=(10, 3.5))
plt.step(t, y, color="tab:orange", lw=1.5, where="post")

period = 1000 / got
plt.annotate("", xy=(period, 16.4), xytext=(0, 16.4),
             arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
plt.text(period / 2, 17, f"one repeat = {period:.2f} ms", color="red", ha="center")

plt.xlabel("milliseconds")
plt.ylabel("amplitude  (0-15)")
plt.title(f"C4 on this project's pulse channel  —  {got:.2f} Hz")
plt.ylim(-1, 19)
plt.tight_layout()
plt.savefig("c4_project.png", dpi=110)
print(f"C4 = {want:.2f} Hz wanted, {got:.2f} Hz produced, one repeat = {period:.2f} ms")
