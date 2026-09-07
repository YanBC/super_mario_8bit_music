# The 2A03 Signal Path

An answer to the question *"what does '8-bit music' actually mean?"*, built as a
working NES sound-chip emulator rather than an explanation.

The whole chain is here, end to end:

```
software generator  →  waveform  →  non-linear mixer  →  micro-speaker
   (register writes)     (4 voices)    (resistor ladder)    (mass on a spring)
```

Everything below is computed from the real hardware numbers. Nothing is sampled,
and no audio was recorded — the WAV files in this directory are synthesised from
275 note events.

---

## The gist

**"8-bit" names the CPU, not the audio.** The Ricoh 2A03 has no sample memory and
no notion of "play an A." It has a 1,789,773 Hz clock, five fixed channels, and a
handful of registers. A cartridge contains a *program* that rewrites ~11 bytes
sixty times a second; the chip builds the waveform from integers.

Three consequences follow, and they are the sound you remember:

**1. Pitch is an integer clock divider.** For the pulse channels,

```
f = 1,789,773 / (16 × (t + 1))       t = 11-bit timer at $4002/$4003
```

`t` is a whole number, so most notes are simply unreachable — you get the nearest
one. The error grows as you go up:

```
 note   want Hz   timer   $4002/$4003   actually get   error
   C3    130.81     854   (0x56,0x03)      130.83     +0.2 cents
   C5    523.25     213   (0xd5,0x00)      522.71     -1.8 cents
   G5    783.99     142   (0x8e,0x00)      782.24     -3.9 cents
```

**2. Four voices, fixed jobs, 4-bit volume.** Two pulse channels (melody and
harmony, 2-bit duty: 12.5 / 25 / 50 / 75 %), one triangle (bass, a 16-step
staircase divided by 32 so it lands an octave lower — and with *no volume
register at all*, only on or off), one noise channel driven by a 15-bit LFSR
that is every drum in the kit. Each emits an integer 0–15. **Four bits, not
eight.**

**3. The mixer is non-linear.** The channels drive a resistor ladder, not a sum:

```
pulse_out = 95.88 / (8128 / (p1 + p2) + 100)
tnd_out   = 159.79 / (1 / (tri/8227 + noise/12241) + 100)
```

so loud passages self-compress. That is a real part of why the console sounds
glued together — and why the triangle, at full scale, is the *loudest* of the
three (0.246 vs 0.174 noise, 0.149 pulse). NES basslines cut through by design.

**4. The speaker is a low-pass filter made of physics.** The cone is a mass on a
spring: `m·x″ + c·x′ + k·x = B·l·i(t)`. It can't draw the square wave's corners,
and a small one resonates around 650 Hz, so it *physically cannot move far
enough* to make bass. That's why the triangle bassline nearly vanishes on a
phone and comes back on headphones. Modelled here as an RBJ biquad,
f₀ = 650 Hz, Q = 1.1.

### Why any of this was clever

The advance was economic. The same 12.8 seconds costs:

| as | bytes |
|---|---|
| note data | **550** |
| recorded 16-bit mono @ 44.1 kHz | **1,128,960** |

A ratio of about **2,053 : 1**. Recorded audio in 1985 wasn't slightly too
expensive — it was off by three orders of magnitude. So the music *had* to be an
algorithm, and the composer had to write for a synth he could not modify, with
three timbres and no chords.

The favourite consequence: a pulse channel is monophonic, so drivers fake a
chord by flipping the pitch register every frame — C, E, G, C, E, G — fast
enough that the ear fuses it. A workaround became a genre signature.

---

## Audio from zero

*Written for someone who works with images and video but hasn't opened a WAV
before. Everything here is demonstrated by the small scripts in the table at the
end, run against the files in this directory.*

### A WAV is a one-dimensional array

That's the whole format. A header, then a long list of numbers. It maps onto
image work almost exactly:

| photo / video | WAV |
|---|---|
| 1920×1080 grid of pixels | flat list of 564,480 numbers |
| 30 fps | 44,100 "fps" (the sample rate) |
| 8-bit pixel: 0–255 | 16-bit sample: −32,768 … 32,767 |
| R, G, B channels | mono (1) or stereo (2) channels |

Loading one needs no libraries beyond the standard library:

```python
import wave, numpy as np

with wave.open("mario_style_full.wav", "rb") as w:
    rate = w.getframerate()                                    # 44100
    data = np.frombuffer(w.readframes(w.getnframes()), "<i2")  # 564480 numbers
```

The four header fields, all of which appear in `wav_viz.py`:

| call | means | here |
|---|---|---|
| `getnchannels()` | channels (1 = mono, 2 = stereo) | `1` |
| `getsampwidth()` | **bytes** per sample — not bits | `2`, i.e. 16-bit |
| `getframerate()` | samples per second | `44100` |
| `getnframes()` | frame count → duration | `564480` = 12.8 s |

A *frame* is one sample for every channel. In mono they are the same thing; in
stereo, `n` frames means `2n` numbers interleaved L,R,L,R. That is what the
`reshape(-1, ch)` in `wav_viz.read_wav` is for.

### One sample is a single number

Not a tuple, not a struct — a scalar. `data[70560]` is `17098`, `ndim == 0`. On
disk it is two bytes, `ca 42`, little-endian: 202 + 66×256 = 17,098.

So the dimensionality mirrors images one rank down:

| | one element | whole thing |
|---|---|---|
| grayscale photo | 1 number | 2-D `(height, width)` |
| RGB photo | 3 numbers | 3-D `(height, width, 3)` |
| **mono audio** | **1 number** | **1-D `(samples,)`** |
| stereo audio | 2 numbers | 2-D `(samples, 2)` |

A mono sample is the audio equivalent of a grayscale pixel.

### Amplitude: raw integers or normalised floats

The same audio shows up in two unit systems depending on whether you divide:

| | raw (`wav_plot.py`) | ÷ 32768 (`wav_viz.py`) |
|---|---|---|
| lowest sample in the mix | −15,500 | −0.473 |
| highest sample | 27,851 | 0.850 |
| full scale / clipping | ±32,768 | ±1.0 |

`getsampwidth()` sets the *container*; the plotted range is whatever the data
actually uses — bit depth vs histogram, exactly as in a photo. This mix peaks at
85 % of full scale, so it has 15 % headroom before clipping.

Normalising to −1…1 is the convention every audio library follows, for three
reasons. It is bit-depth independent, `1.0` reads directly as "as loud as this
format goes", and integer audio **overflows silently**:

```python
np.array([20000, 25000], dtype=np.int16) * 2   # -> [-25536, -15536], garbage
```

Doubling the volume wraps past 32,767 into negative numbers. Same trap as
brightening a `uint8` photo past 255.

### A note is a wave that repeats

Pitch is nothing but repetition rate. C4 (middle C) repeats 261.63 times per
second, so one repeat takes 1/261.63 = **3.82 ms**. "Hz" means "repeats per
second", and that is the entire definition of a note's name.

| note | repeats per second | one repeat |
|---|---|---|
| C3 | 130.81 Hz | 7.65 ms |
| **C4** | **261.63 Hz** | **3.82 ms** |
| C5 | 523.25 Hz | 1.91 ms |

An octave up doubles the rate and halves the width. See `c4.png`.

### Shape is the instrument; rate is the note

Two independent properties live in the same waveform:

| property | controls | in a plot |
|---|---|---|
| how *fast* it repeats | **which note** | spacing of the repeats |
| the *shape* of one repeat | **what it sounds like** | the squiggle between them |

This is **timbre**, and it is why this project's C4 and a piano's C4 are both
genuinely C4 yet sound nothing alike. `c4_two_shapes.png` puts a sine and this
chip's pulse at C4 on the same axes: the repeat boundaries land in identical
places, and only the shape differs.

`four_shapes.png` shows the chip's entire palette — two rectangular pulse
voices, a 16-step triangle staircase, and structureless noise. **None of them is
a sine, and the hardware cannot make one.** A sine needs smoothly varying
values; this chip only flips switches and counts in whole numbers. Every
waveform it produces is flat segments and vertical jumps. That, rather than
nostalgia, is what "8-bit sound" actually denotes.

At C4 the chip wants 261.63 Hz, its integer timer lands on 261.36 Hz, and the
quarter-Hz error is inaudible — the general case of the table in *The gist*.

### Instruments add; channels are speakers

Channels describe *where a sound goes* — stereo L/R, or six for 5.1 — never
which instrument it is. Instruments are combined by plain addition:

```python
mix = piano + guitar + drums
```

That is what "mixing" means, and it mirrors physics: air pressure at your ear is
already the sum of every source. Two consequences. Summing **clips** (three
tracks peaking at 0.85 sum to 2.55, past the 1.0 ceiling), which is why faders
go down; and summing is **irreversible** — no arithmetic recovers the parts,
like flattening layers in Photoshop.

Hence the `solo_*.wav` files here are separate *files* (stems), not channels.
Summing the three of them correlates 0.938 with the real mix rather than 1.000,
for three reasons worth knowing: each stem was normalised independently, the
second pulse channel has no stem, and — peculiar to this hardware — the mixer is
non-linear, so the channels genuinely do not simply add.

### What the speaker receives

Not numbers, and not those steps:

1. **Numbers → voltage.** The DAC emits a new voltage every 22.7 µs.
2. **The staircase vanishes.** A reconstruction filter removes everything above
   22,050 Hz. By the sampling theorem exactly one smooth curve passes through
   the samples, and that is what comes out. The steps in a plot are an artifact
   of drawing discrete samples, never present in the sound.
3. **Voltage → force**, via current through the voice coil.
4. **Force → motion**, and here physics intrudes: the cone has mass, so force
   sets acceleration, not position. It lags, overshoots, rings.
5. **Motion → pressure waves.**

Step 4 is the `speaker()` model, and `speaker_view.png` plots it: 10 ms of
`mario_style_full.wav` against `through_microspeaker.wav`. Every sharp edge is
gone. **That orange curve is what the air actually gets.**

### Getting notes back out of samples

The reverse trip, in `wav_notes.py`:

1. **Samples → period.** Slide a chunk over a copy of itself; at a lag of one
   period it lines up with itself and the overlap peaks. That is
   **autocorrelation**.
2. **Period → frequency.** `freq = rate / lag`.
3. **Frequency → name.** Music is logarithmic — an octave doubles, and holds 12
   equal steps. Anchor on A4 = 440 Hz = MIDI 69:
   `midi = round(69 + 12*log2(freq/440))`.

Two traps, both of which this repo hit before the code was correct:

- **Overlap bias.** `np.correlate` overlaps fewer samples at longer lags, so raw
  values shrink with lag and the peak drifts high. Divide by the overlap count.
- **Octave errors.** Two periods line up as well as one, so the *tallest* peak
  is often the sub-octave. Take the first strong peak instead. This is the
  classic failure of every pitch detector ever written.

Resolution is limited because lag is a whole number of samples: near 130 Hz the
steps are ~0.4 Hz, near 660 Hz they are ~10 Hz. The detector quantises pitch by
integer division for the same reason the chip does.

Note that this is pitch detection, not transcription. Real note extraction also
needs **onsets**, and collapses on **polyphony** — a three-note chord returns one
confused number. These files are easy mode because every channel is monophonic
by hardware design.

## Files

| file | what it is |
|---|---|
| `nes_apu.py` | The emulator. Real timers, real 15-bit LFSR, real non-linear DAC, speaker model. Renders every WAV and dumps `_stages.npy`. |
| `plot_stages.py` | Reads `_stages.npy`, draws the two figures. |
| `serve.py` | Local web UI. Serves `ui.html` and renders every sound on demand with the emulator above. |
| `ui.html` | The page `serve.py` serves. Draws and plays; contains no synthesis. |
| `signal-path.html` | Standalone interactive version — a JS port of the emulator, no server needed. Published as an artifact. |
| `mario_style_full.wav` | The full mix, 12.8 s. |
| `solo_pulse1_lead.wav` | Melody channel alone. |
| `solo_triangle_bass.wav` | Bass alone — note it never fades, only switches. |
| `solo_noise_drums.wav` | The entire drum kit: one shift register. |
| `through_microspeaker.wav` | The full mix after the cone model. Compare to `mario_style_full.wav`. |
| `01_zoom_cascade.png` | Five-level zoom: whole song → phrase → note → single cycle → cone displacement. |
| `02_the_four_voices.png` | Duty cycles, triangle staircase, LFSR, DAC transfer curve. |
| `_stages.npy` | Intermediate per-channel signals (22 MB, regenerable). |

### Walkthrough scripts

The *Audio from zero* section above is backed by these. They only need `numpy`
and `matplotlib`, and each is short enough to read in one sitting. Roughly in
the order that section introduces them:

| file | what it is |
|---|---|
| `wav_plot.py` | The minimum: open a WAV, plot it. Six lines of actual work. Also usable as `from wav_plot import ...` in a REPL. |
| `c4_plot.py` | One note, C4, as a pure sine, with one repeat marked. The starting picture. |
| `c4_project.py` | The same C4 as *this chip* produces it. Same repeat rate, square shape. |
| `four_shapes.py` | Writes `c4_two_shapes.png` and `four_shapes.png` — the sine/pulse comparison, and all four voices at C4. |
| `speaker_view.py` | Writes `speaker_view.png`: the file's samples against the cone's actual motion. |
| `wav_notes.py` | The reverse trip — autocorrelation pitch detection, printing note names per 400 ms. |
| `wav_viz.py` | Four-panel diagnostic view (waveform, zoom, spectrogram, spectrum) for any WAV, any bit depth. `--zoom`, `--fmax`, `--dyn`, `--show`. |

## Running it

Requires `numpy` and `matplotlib`. Deliberately **not** `scipy` — the biquad is
hand-written so the filter math is visible rather than imported.

```sh
python3 nes_apu.py      # renders the 5 WAVs, _stages.npy, and the register table
python3 plot_stages.py  # renders the 2 PNGs (needs _stages.npy)
```

The walkthrough scripts are independent of the two above — they read the WAVs
that are already here:

```sh
python3 wav_plot.py mario_style_full.wav     # the simplest possible look
python3 four_shapes.py                       # every waveform the chip can make
python3 wav_notes.py solo_triangle_bass.wav  # what notes the bass is playing
python3 wav_viz.py mario_style_full.wav --zoom 1.605 1.618 --fmax 8000
```

For the interactive version, run the local UI:

```sh
python3 serve.py            # opens http://127.0.0.1:8000
python3 serve.py --port 9000 --no-open
```

A live oscilloscope, a clickable register table that auditions the exact timer
value, per-channel solo, a duty-cycle slider, an arpeggio A/B, and a
speaker-resonance slider.

**Why a server rather than a single file.** `signal-path.html` ports the whole
chip to JavaScript and synthesises in the page. That means two emulators to keep
in agreement, and an audio path that depends on WebAudio scheduling behaving —
in practice it sometimes came up silent. `serve.py` renders every sound with
`nes_apu.py` and hands the browser a finished WAV, so there is one emulator and
the page only has to decode and play. The full mix it serves is byte-identical
to `mario_style_full.wav`.

Edit the score in `nes_apu.py`, reload the page, and everything follows — the
register table, the ledger arithmetic and all five renders are computed from it.

`signal-path.html` is still there for the no-server case.

## Implementation notes

- Synthesis is vectorised (`np.repeat` to sample rate, `np.cumsum(hz)/SR` for
  continuous phase) rather than per-clock emulated. Pitch quantisation — the part
  that matters pedagogically — is preserved exactly.
- The LFSR's full 32,767-sample period is computed once and indexed by a phase
  accumulator, turning a ~5.6 M-iteration loop into a lookup.
- The driver runs at 60 Hz (vblank NMI), which is 735 samples per frame at
  44.1 kHz — exactly, no remainder.
- In the browser, audio is rendered into an `AudioBuffer` in plain JS. No
  AudioWorklet (blob-URL/CSP risk in the artifact sandbox), no
  `ScriptProcessorNode` (deprecated).

## On the music

The tune is an **original composition** in the same idiom — I–vi–IV–V, 150 BPM,
8 bars. No Nintendo material is reproduced anywhere in this project. The
mechanism being demonstrated is identical either way; that's rather the point.
