"""Visualise WAV files: waveform, spectrogram, and average spectrum.

    python wav_viz.py *.wav              # one PNG per file
    python wav_viz.py mario_style_full.wav --show     # interactive window
    python wav_viz.py a.wav b.wav --zoom 1.6 1.8      # zoom on a 200 ms slice

Only needs numpy + matplotlib; WAVs are read with the stdlib `wave` module.
"""
import argparse, os, wave
import numpy as np, matplotlib
import matplotlib.pyplot as plt

BG, FG, GRID = "#12101a", "#f2efe6", "#3a3550"
COL = ["#ff5f6d", "#ffc23c", "#4fd6c8", "#9b8cff", "#f2efe6"]


def read_wav(path):
    """-> (samples float32 in [-1,1], shape (n,) or (n,ch); sample rate)."""
    with wave.open(path, "rb") as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:                                  # 8-bit WAV is unsigned
        x = (np.frombuffer(raw, "<u1").astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    elif width == 3:                                # 24-bit: pad to 32
        b = np.frombuffer(raw, "u1").reshape(-1, 3)
        x = (np.c_[np.zeros(len(b), "u1"), b].copy().view("<i4").ravel()
             .astype(np.float32) / 2147483648)
    else:
        raise ValueError(f"{path}: unsupported sample width {width*8}-bit")
    return (x.reshape(-1, ch) if ch > 1 else x), sr


def style(ax, title):
    ax.set_title(title, color=FG, fontsize=10, loc="left", pad=6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(alpha=.15, color=GRID)


def figure_for(path, zoom=None, fmax=None, dyn=70):
    x, sr = read_wav(path)
    mono = x.mean(axis=1) if x.ndim > 1 else x       # spectra use a mono mixdown
    t = np.arange(len(mono)) / sr

    fig, ax = plt.subplots(4, 1, figsize=(11, 10))
    fig.suptitle(f"{os.path.basename(path)}   ·   {sr} Hz   ·   "
                 f"{len(mono)/sr:.2f} s   ·   {'mono' if x.ndim == 1 else f'{x.shape[1]} ch'}",
                 color=FG, fontsize=11)

    # 1 · the whole file, one line per channel
    chans = [(mono, COL[4], "mono")] if x.ndim == 1 else \
            [(x[:, i], COL[i % len(COL)], f"ch {i}") for i in range(x.shape[1])]
    for arr, c, lab in chans:
        ax[0].plot(t, arr, lw=.4, color=c, label=lab)
    if x.ndim > 1:
        ax[0].legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    style(ax[0], "1  WAVEFORM  ·  amplitude over the whole file")
    ax[0].set_xlabel("seconds"); ax[0].set_ylabel("amplitude")

    # 2 · zoom, so individual cycles are visible
    a, b = zoom if zoom else (len(mono) / sr * .5, len(mono) / sr * .5 + .01)
    i, j = int(a * sr), min(int(b * sr), len(mono))
    ax[1].step(t[i:j] * 1000, mono[i:j], lw=1.2, color=COL[0], where="post")
    style(ax[1], f"2  ZOOM  ·  {a:.3f}–{b:.3f} s  ·  {(b-a)*1000:.0f} ms, sample-stepped")
    ax[1].set_xlabel("milliseconds"); ax[1].set_ylabel("amplitude")

    # 3 · spectrogram (matplotlib's own FFT, no scipy needed)
    with np.errstate(divide="ignore"):          # digital silence -> log10(0)
        *_, im = ax[2].specgram(mono, NFFT=2048, Fs=sr, noverlap=1536, cmap="magma")
    top = im.get_array().max()                  # 70 dB window, else it saturates
    im.set_clim(top - dyn, top)
    style(ax[2], f"3  SPECTROGRAM  ·  frequency content over time  ·  top {dyn:.0f} dB")
    ax[2].set_xlabel("seconds"); ax[2].set_ylabel("Hz")
    ax[2].set_ylim(0, fmax or sr / 2)

    # 4 · average spectrum of the whole file
    win = np.hanning(len(mono))
    spec = np.abs(np.fft.rfft(mono * win))
    freq = np.fft.rfftfreq(len(mono), 1 / sr)
    db = 20 * np.log10(spec / max(spec.max(), 1e-12) + 1e-12)
    ax[3].semilogx(freq[1:], db[1:], lw=.6, color=COL[2])
    style(ax[3], "4  SPECTRUM  ·  average over the whole file")
    ax[3].set_xlim(20, fmax or sr / 2); ax[3].set_ylim(-100, 5)
    ax[3].set_xlabel("Hz (log)"); ax[3].set_ylabel("dB")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--show", action="store_true", help="open a window instead of writing PNGs")
    p.add_argument("--zoom", nargs=2, type=float, metavar=("START", "END"),
                   help="seconds to zoom on in panel 2 (default: 10 ms mid-file)")
    p.add_argument("--fmax", type=float, help="top of the frequency axis in Hz")
    p.add_argument("--dyn", type=float, default=70,
                   help="spectrogram dynamic range in dB below peak (default 70)")
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()

    if not args.show:
        matplotlib.use("Agg", force=True)
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
        "ytick.color": FG, "axes.edgecolor": GRID, "font.size": 9})

    for f in args.files:
        fig = figure_for(f, args.zoom, args.fmax, args.dyn)
        if not args.show:
            out = os.path.splitext(os.path.basename(f))[0] + "_viz.png"
            fig.savefig(out, dpi=args.dpi, facecolor=BG)
            print("wrote", out)
    if args.show:
        plt.show()
