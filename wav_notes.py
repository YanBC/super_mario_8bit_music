"""Samples -> pitch -> note names.

    python wav_notes.py solo_pulse1_lead.wav
"""
import sys, wave
import numpy as np

NAMES = "C C# D D# E F F# G G# A A# B".split()


def load(path):
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return data.astype(np.float64) / 32768, w.getframerate()


def pitch(chunk, rate, fmin=50, fmax=2000, thresh=0.5):
    """Fundamental frequency by autocorrelation.

    Slide the chunk over a copy of itself. At a lag of exactly one period the
    wave lines up with itself and the overlap peaks -> that lag is the period.

    Two traps, both handled below:
      * np.correlate overlaps fewer samples at longer lags, which biases the
        result toward high pitches -- divide by the overlap count to undo it.
      * two periods line up just as well as one, so the TALLEST peak is often
        an octave too low -- take the first strong peak instead.
    """
    c = chunk - chunk.mean()                          # centre it first
    corr = np.correlate(c, c, "full")[len(c) - 1:]    # positive lags only
    corr = corr / np.arange(len(c), 0, -1)            # undo the overlap bias
    corr = corr / corr[0]                             # now corr[0] == 1.0
    lo, hi = int(rate / fmax), int(rate / fmin)
    win = corr[lo:hi]
    peaks = np.where((win[1:-1] > win[:-2]) & (win[1:-1] >= win[2:]))[0] + 1
    if not len(peaks):
        return None
    strong = peaks[win[peaks] > thresh * win[peaks].max()]
    return rate / (lo + strong[0])


def note_name(freq):
    """440 Hz = A4 = MIDI 69, and an octave is 12 equal steps."""
    midi = round(69 + 12 * np.log2(freq / 440.0))
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "solo_pulse1_lead.wav"
    x, rate = load(path)

    print(f"{path}\n{'time':>6} {'Hz':>9}   note")
    for start in np.arange(0.2, len(x) / rate, 0.4):     # sample every 400 ms
        chunk = x[int(start * rate): int(start * rate) + 4096]
        if len(chunk) < 2048 or np.abs(chunk).max() < 0.02:
            continue                                      # skip silence
        f = pitch(chunk, rate)
        if f is None:
            continue
        print(f"{start:6.1f} {f:9.2f}   {note_name(f)}")
