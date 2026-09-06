"""
An emulator of the NES sound chip (Ricoh 2A03 APU) -- the thing that made
"8-bit music". Renders an original Mario-style chiptune to a WAV file and
plots every stage of the signal chain.

    music data (bytes)  ->  APU registers  ->  digital waveform
                        ->  DAC voltage    ->  speaker cone motion  ->  air

Nothing here is sampled audio. Every sample is *computed* from a handful
of integer register values, exactly as the 1983 hardware did it.
"""
import wave, struct
import numpy as np

# ----------------------------------------------------------------------------
# HARDWARE CONSTANTS (real numbers from the NTSC NES)
# ----------------------------------------------------------------------------
CPU_HZ   = 1_789_773   # 6502 clock. EVERY pitch is derived by dividing this.
SR       = 44_100      # our output sample rate
FPS      = 60          # the music driver runs once per video frame (vblank NMI)
SPF      = SR // FPS   # 735 samples per frame -- exact, conveniently

# The noise channel's 16 selectable divider values (NTSC).
NOISE_PERIODS = [4, 8, 16, 32, 64, 96, 128, 160,
                 202, 254, 380, 508, 762, 1016, 2034, 4068]

# The triangle channel is a 32-step staircase: down 15->0, then back up 0->15.
TRI_STEPS = np.array(list(range(15, -1, -1)) + list(range(0, 16)), dtype=np.float64)

DUTY = {0: 0.125, 1: 0.25, 2: 0.50, 3: 0.75}   # the only 4 tone colours available


# ----------------------------------------------------------------------------
# STEP 1: notes -> integer register values
# ----------------------------------------------------------------------------
_NAMES = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}

def note_hz(name):
    """'A4' -> 440.0"""
    if name == 'R':
        return 0.0
    letter, octave = name[:-1], int(name[-1])
    midi = (octave + 1) * 12 + _NAMES[letter]
    return 440.0 * 2 ** ((midi - 69) / 12)

def pulse_period(hz):
    """Hz -> the 11-bit timer value the CPU actually writes to $4002/$4003.
       Hardware then plays back  f = CPU / (16 * (t + 1))."""
    if hz <= 0:
        return None
    t = round(CPU_HZ / (16 * hz) - 1)
    return int(np.clip(t, 8, 0x7FF))          # 11 bits: 0..2047

def tri_period(hz):
    """Triangle divides by 32 instead of 16, so it reaches an octave lower."""
    if hz <= 0:
        return None
    t = round(CPU_HZ / (32 * hz) - 1)
    return int(np.clip(t, 2, 0x7FF))

def pulse_actual_hz(t):  return CPU_HZ / (16 * (t + 1))
def tri_actual_hz(t):    return CPU_HZ / (32 * (t + 1))


# ----------------------------------------------------------------------------
# STEP 2: the song, as a music driver would store it
# ----------------------------------------------------------------------------
# Durations in frames (1/60 s). 150 BPM: quarter = 24 frames, eighth = 12.
Q, E, H = 24, 12, 48

def parse(track):
    """'C5/8 E5/8' -> [('C5',12), ('E5',12)]"""
    out = []
    for tok in track.split():
        name, d = tok.split('/')
        out.append((name, {'2': H, '4': Q, '8': E, '16': 6}[d]))
    return out

MELODY = parse("""
 C5/8 E5/8 G5/8 A5/8 G5/8 E5/8 D5/4
 A4/8 C5/8 E5/8 F5/8 E5/8 C5/8 B4/4
 F4/8 A4/8 C5/8 D5/8 C5/8 A4/8 G4/4
 G4/8 B4/8 D5/8 G5/8 F5/8 D5/8 B4/4
 E5/4 G5/4 C6/4 B5/8 A5/8
 A5/4 E5/4 A5/8 G5/8 E5/8 D5/8
 F5/4 A5/4 G5/8 F5/8 E5/8 D5/8
 D5/8 E5/8 F5/8 G5/8 B5/4 G5/4
""")

HARMONY = parse(" ".join([
 "E4/8 G4/8 C5/8 G4/8 E4/8 G4/8 C5/8 G4/8",
 "A3/8 C4/8 E4/8 C4/8 A3/8 C4/8 E4/8 C4/8",
 "F3/8 A3/8 C4/8 A3/8 F3/8 A3/8 C4/8 A3/8",
 "G3/8 B3/8 D4/8 B3/8 G3/8 B3/8 D4/8 B3/8",
] * 2))

BASS = parse(" ".join([
 "C3/8 C3/8 G2/8 C3/8 E3/8 C3/8 G2/8 G2/8",
 "A2/8 A2/8 E2/8 A2/8 C3/8 A2/8 E2/8 E2/8",
 "F2/8 F2/8 C3/8 F2/8 A2/8 F2/8 C3/8 C3/8",
 "G2/8 G2/8 D3/8 G2/8 B2/8 G2/8 D3/8 D3/8",
] * 2))

N_FRAMES = sum(d for _, d in MELODY)


# ----------------------------------------------------------------------------
# STEP 3: run the "driver" -- produce one register snapshot per frame
# ----------------------------------------------------------------------------
def envelope(n, start=15, sustain=8, decay_every=4):
    """Per-frame volume ramp. Real drivers keep tables exactly like this."""
    v = np.maximum(sustain, start - np.arange(n) // decay_every)
    return v.astype(np.float64)

def render_tonal(track, kind, vol_start, vol_sustain):
    """Expand a note list into per-frame arrays of (period, volume)."""
    per  = np.zeros(N_FRAMES)
    vol  = np.zeros(N_FRAMES)
    i = 0
    for name, dur in track:
        hz = note_hz(name)
        t = pulse_period(hz) if kind == 'pulse' else tri_period(hz)
        if t is not None:
            gap = 1 if dur > 6 else 0            # tiny gap so repeats re-articulate
            per[i:i+dur] = t
            vol[i:i+dur-gap] = envelope(dur-gap, vol_start, vol_sustain)
        i += dur
    return per, vol

def render_drums():
    """Noise channel: hats on every eighth, kick on 1 & 3, snare on 2 & 4."""
    per = np.full(N_FRAMES, 2.0)
    vol = np.zeros(N_FRAMES)
    bar = 96
    for b in range(0, N_FRAMES, bar):
        for k in range(8):                                   # hi-hat
            t = b + k*E
            per[t:t+2], vol[t:t+2] = 1, [4, 2]
        for t in (b, b + 2*Q):                               # kick
            per[t:t+4], vol[t:t+4] = 12, [12, 9, 5, 2]
        for t in (b + Q, b + 3*Q):                           # snare
            per[t:t+7], vol[t:t+7] = 6, [14, 12, 9, 7, 5, 3, 1]
    return per, vol


# ----------------------------------------------------------------------------
# STEP 4: synthesis -- turn register values into a digital waveform
# ----------------------------------------------------------------------------
def to_samples(frame_arr):
    return np.repeat(frame_arr, SPF)

def gen_pulse(period_f, vol_f, duty):
    """A pulse channel outputs one of exactly 2 values: 0, or its 4-bit volume."""
    t = to_samples(period_f)
    v = to_samples(vol_f)
    hz = np.where(t > 0, CPU_HZ / (16 * (t + 1)), 0.0)
    phase = np.cumsum(hz) / SR
    return np.where((phase % 1.0) < DUTY[duty], v, 0.0)      # 0..15

def gen_triangle(period_f, on_f):
    """Triangle has no volume register -- it's on or off, always full scale.
       Its 32 steps are why NES bass has that hollow, slightly buzzy tone."""
    t = to_samples(period_f)
    on = to_samples(on_f)
    hz = np.where(t > 0, CPU_HZ / (32 * (t + 1)), 0.0)
    phase = np.cumsum(hz) / SR
    step = (np.floor(phase * 32).astype(np.int64)) % 32
    return TRI_STEPS[step] * (on > 0)                        # 0..15

def lfsr_sequence():
    """The real 15-bit shift register. Feedback = bit0 XOR bit1.
       It repeats after 32767 clocks -- long enough to sound like static."""
    reg = 1
    out = np.empty(32767, dtype=np.float64)
    for i in range(32767):
        out[i] = 1.0 - (reg & 1)          # channel is silenced when bit0 is set
        fb = (reg & 1) ^ ((reg >> 1) & 1)
        reg = (reg >> 1) | (fb << 14)
    return out

_LFSR = lfsr_sequence()

def gen_noise(idx_f, vol_f):
    idx = to_samples(idx_f).astype(int)
    v   = to_samples(vol_f)
    rate = CPU_HZ / np.array(NOISE_PERIODS, dtype=np.float64)[idx]
    pos = (np.cumsum(rate) / SR).astype(np.int64) % 32767
    return _LFSR[pos] * v                                    # 0..15


# ----------------------------------------------------------------------------
# STEP 5: the mixer -- the DAC is deliberately NON-linear
# ----------------------------------------------------------------------------
def mix(p1, p2, tri, noi):
    """Straight from the NESdev hardware docs. The channels do not simply add;
       they load a resistor ladder, so loud parts compress. This is a big part
       of the NES 'sound'."""
    s = p1 + p2
    pulse_out = np.where(s > 0, 95.88 / (8128.0 / np.maximum(s, 1e-9) + 100), 0.0)
    tnd = tri / 8227.0 + noi / 12241.0
    tnd_out = np.where(tnd > 0, 159.79 / (1.0 / np.maximum(tnd, 1e-12) + 100), 0.0)
    return pulse_out + tnd_out                               # ~0.0 .. 1.0 volts


# ----------------------------------------------------------------------------
# STEP 6: the micro-speaker, as physics
# ----------------------------------------------------------------------------
def speaker(v, f0=650.0, q=1.1):
    """A tiny speaker cone is a mass on a spring, pushed by the coil:

            m*x'' + c*x' + k*x  =  B*l*i(t)   ~  voltage

    i.e. a 2nd-order low-pass on *cone displacement*. Discretise it (RBJ
    biquad) and you get the real thing: it cannot follow the instant vertical
    edges of a square wave, and it can barely move at all below f0 -- which is
    exactly why phone and NES speakers have no bass."""
    w = 2 * np.pi * f0 / SR
    a = np.sin(w) / (2 * q)
    c, b0 = np.cos(w), (1 - np.cos(w)) / 2
    b = np.array([b0, 2 * b0, b0]) / (1 + a)
    A = np.array([(-2 * c) / (1 + a), (1 - a) / (1 + a)])
    y = np.zeros_like(v)
    x1 = x2 = y1 = y2 = 0.0
    for i, x in enumerate(v):
        yn = b[0]*x + b[1]*x1 + b[2]*x2 - A[0]*y1 - A[1]*y2
        x2, x1, y2, y1 = x1, x, y1, yn
        y[i] = yn
    return y


def write_wav(path, sig, gain=0.85):
    s = sig - np.mean(sig)
    s = s / (np.max(np.abs(s)) + 1e-12) * gain
    pcm = (s * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"  wrote {path}  ({len(sig)/SR:.1f}s)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Rendering NES APU...")
    mp, mv = render_tonal(MELODY,  'pulse', 15, 9)
    hp, hv = render_tonal(HARMONY, 'pulse', 10, 6)
    bp, bv = render_tonal(BASS,    'tri',    1, 1)
    npd, nv = render_drums()

    p1  = gen_pulse(mp, mv, duty=2)      # 50%  -- fat lead
    p2  = gen_pulse(hp, hv, duty=1)      # 25%  -- thinner harmony
    tri = gen_triangle(bp, bv)
    noi = gen_noise(npd, nv)
    out = mix(p1, p2, tri, noi)

    write_wav("mario_style_full.wav",    out)
    write_wav("solo_pulse1_lead.wav",    mix(p1, np.zeros_like(p1), np.zeros_like(p1), np.zeros_like(p1)))
    write_wav("solo_triangle_bass.wav",  mix(np.zeros_like(p1), np.zeros_like(p1), tri, np.zeros_like(p1)))
    write_wav("solo_noise_drums.wav",    mix(np.zeros_like(p1), np.zeros_like(p1), np.zeros_like(p1), noi))
    write_wav("through_microspeaker.wav", speaker(out))

    np.save("_stages.npy", np.vstack([p1, p2, tri, noi, out]))

    # ---- the register table: notes are integers, and they are not exact -----
    print("\n  note   want Hz   $4002/$4003 timer   actually get   error")
    for n in ["C5", "E5", "G5", "A5", "C6", "G2", "C3"]:
        hz = note_hz(n)
        t  = pulse_period(hz); got = pulse_actual_hz(t)
        cents = 1200 * np.log2(got / hz)
        print(f"  {n:>4}  {hz:8.2f}   {t:4d}  = ${t:03X}"
              f"   ({t & 0xFF:#04x},{t >> 8:#04x})   {got:8.2f}   {cents:+5.1f} cents")
