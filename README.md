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

## Files

| file | what it is |
|---|---|
| `nes_apu.py` | The emulator. Real timers, real 15-bit LFSR, real non-linear DAC, speaker model. Renders every WAV and dumps `_stages.npy`. |
| `plot_stages.py` | Reads `_stages.npy`, draws the two figures. |
| `signal-path.html` | Interactive version of all of the above — a JS port of the emulator. Published as an artifact. |
| `mario_style_full.wav` | The full mix, 12.8 s. |
| `solo_pulse1_lead.wav` | Melody channel alone. |
| `solo_triangle_bass.wav` | Bass alone — note it never fades, only switches. |
| `solo_noise_drums.wav` | The entire drum kit: one shift register. |
| `through_microspeaker.wav` | The full mix after the cone model. Compare to `mario_style_full.wav`. |
| `01_zoom_cascade.png` | Five-level zoom: whole song → phrase → note → single cycle → cone displacement. |
| `02_the_four_voices.png` | Duty cycles, triangle staircase, LFSR, DAC transfer curve. |
| `_stages.npy` | Intermediate per-channel signals (22 MB, regenerable). |

## Running it

Requires `numpy` and `matplotlib`. Deliberately **not** `scipy` — the biquad is
hand-written so the filter math is visible rather than imported.

```sh
python3 nes_apu.py      # renders the 5 WAVs, _stages.npy, and the register table
python3 plot_stages.py  # renders the 2 PNGs (needs _stages.npy)
```

Open `signal-path.html` in a browser for the interactive version: a live
oscilloscope, a clickable register table that auditions the exact timer value,
per-channel solo, a duty-cycle slider, an arpeggio A/B, and a speaker-resonance
slider.

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
