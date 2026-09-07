"""What the file holds vs. where the speaker cone actually goes."""
import wave
import numpy as np
import matplotlib.pyplot as plt


def load(path):
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return data, w.getframerate()


before, rate = load("mario_style_full.wav")      # the numbers in the file
after, _ = load("through_microspeaker.wav")      # after the cone's physics

a, b = int(1.600 * rate), int(1.610 * rate)      # 10 ms
ms = np.arange(b - a) / rate * 1000

plt.figure(figsize=(10, 3.5))
plt.plot(ms, before[a:b], lw=1.2, label="what's in the file (digital steps)")
plt.plot(ms, after[a:b], lw=2.0, label="where the cone actually goes")
plt.xlabel("milliseconds")
plt.ylabel("amplitude")
plt.title("the speaker never sees the steps")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("speaker_view.png", dpi=110)
print("wrote speaker_view.png")
