import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nes_apu import SR, speaker, mix, DUTY, TRI_STEPS, _LFSR

p1, p2, tri, noi, out = np.load("_stages.npy")
BG, FG, GRID = "#12101a", "#f2efe6", "#3a3550"
COL = ["#ff5f6d", "#ffc23c", "#4fd6c8", "#9b8cff", "#f2efe6"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.edgecolor": GRID, "font.size": 9})

def style(ax, title):
    ax.set_title(title, color=FG, fontsize=10, loc="left", pad=6)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.grid(alpha=.15, color=GRID)

# ============ FIGURE 1: the zoom cascade =====================================
fig, ax = plt.subplots(5, 1, figsize=(11, 12))

t = np.arange(len(out)) / SR
ac = out - out.mean()                      # remove DC so the shape is visible
ax[0].plot(t, ac, lw=.4, color=COL[4])
style(ax[0], "1  THE WHOLE TUNE  ·  12.8 s  ·  you see the drum hits as spikes")
ax[0].set_xlabel("seconds")

a, b = int(1.6*SR), int(3.2*SR)          # one bar
ax[1].plot(t[a:b], ac[a:b], lw=.5, color=COL[4])
style(ax[1], "2  ONE BAR  ·  1.6 s  ·  four beats, each note a burst")

a, b = int(1.60*SR), int(1.80*SR)        # one note, channels separated
for arr, c, lab in zip([p1, p2, tri, noi], COL, ["pulse 1 (lead)", "pulse 2 (harmony)", "triangle (bass)", "noise (drums)"]):
    ax[2].plot(t[a:b], arr[a:b], lw=.7, color=c, label=lab)
style(ax[2], "3  ONE NOTE  ·  200 ms  ·  the 4 channels, each an integer 0-15")
ax[2].legend(loc="upper right", ncol=4, fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG)
ax[2].set_ylabel("4-bit level")

a, b = int(1.605*SR), int(1.615*SR)      # a few cycles
ax[3].step(t[a:b]*1000, p1[a:b], lw=1.6, color=COL[0], where="post", label="pulse 1: only 2 values, ever")
ax[3].step(t[a:b]*1000, tri[a:b], lw=1.6, color=COL[2], where="post", label="triangle: 16 discrete steps")
style(ax[3], "4  10 ms  ·  THIS is the whole trick: square edges + a staircase")
ax[3].legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
ax[3].set_xlabel("milliseconds"); ax[3].set_ylabel("4-bit level")

# Feed ONE clean square wave to the speaker model so the rounding is obvious.
cone = speaker(p1 - p1.mean())
seg  = slice(int(1.605*SR), int(1.618*SR))
tt   = t[seg]*1000
sq   = (p1[seg] - p1[seg].mean()); sq /= np.ptp(sq)
cn   = cone[seg]; cn = cn / np.ptp(cn)
ax[4].step(tt, sq, lw=1.4, color=COL[3], where="post",
           label="voltage sent to the coil  (instant vertical edges)")
ax[4].plot(tt, cn, lw=2.4, color=COL[1],
           label="where the cone actually is  (mass can't turn on a dime)")
ax[4].fill_between(tt, sq, cn, color=COL[1], alpha=.10, step="post")
style(ax[4], "5  THE MICRO-SPEAKER  ·  a mass on a spring cannot draw corners")
ax[4].legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
ax[4].set_xlabel("milliseconds"); ax[4].set_ylabel("normalised")

plt.tight_layout(); plt.savefig("01_zoom_cascade.png", dpi=130, facecolor=BG)
print("wrote 01_zoom_cascade.png")

# ============ FIGURE 2: the four voices, up close ============================
fig, ax = plt.subplots(2, 2, figsize=(11, 6))
n = 400; x = np.arange(n)/SR*1000
ph = np.arange(n)/SR*440

for d, lbl, c in [(0.125,"12.5% (nasal)",COL[0]), (0.25,"25% (classic lead)",COL[1]), (0.5,"50% (fat, hollow)",COL[2])]:
    ax[0,0].step(x, np.where(ph%1 < d, 15, 0), lw=1.4, color=c, where="post", label=lbl)
style(ax[0,0], "PULSE  ·  3 usable tone colours, that's it")
ax[0,0].legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG)

ax[0,1].step(x, TRI_STEPS[(np.floor(ph*32).astype(int))%32], lw=1.5, color=COL[2], where="post")
ax[0,1].set_ylim(-1,16)
style(ax[0,1], "TRIANGLE  ·  16 steps up, 16 down. No volume control at all.")

ax[1,0].step(np.arange(600)/SR*1000, _LFSR[:600]*15, lw=1.0, color=COL[3], where="post")
style(ax[1,0], "NOISE  ·  a 15-bit shift register. Pseudo-random, repeats at 32767.")
ax[1,0].set_xlabel("milliseconds")

s = np.linspace(0, 30, 400)
ax[1,1].plot(s, 95.88/(8128/np.maximum(s,1e-9)+100), lw=2, color=COL[1])
ax[1,1].plot(s, s*95.88/(8128/30+100)/30, lw=1, ls="--", color=GRID)
style(ax[1,1], "THE DAC IS NON-LINEAR  ·  loud parts compress (dashed = linear)")
ax[1,1].set_xlabel("pulse1 + pulse2  (0-30)"); ax[1,1].set_ylabel("volts out")

plt.tight_layout(); plt.savefig("02_the_four_voices.png", dpi=130, facecolor=BG)
print("wrote 02_the_four_voices.png")
