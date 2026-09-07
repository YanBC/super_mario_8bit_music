#!/usr/bin/env python3
"""Local web UI for the 2A03 signal path.

Every sound on the page is rendered by the real emulator in `nes_apu.py` and
sent to the browser as a WAV. The browser only draws and plays; it contains no
synthesis code at all.

That is deliberate. The published version ported the whole chip to JavaScript
and synthesised in the page, which meant two emulators to keep in agreement and
an audio path that depended on WebAudio scheduling behaving. Here there is one
emulator -- the Python you can read -- and the browser does the single most
reliable thing WebAudio offers: decode a finished buffer and play it.

    python3 serve.py            # http://127.0.0.1:8000
    python3 serve.py --port 9000 --no-open

Requires numpy. No other dependencies, no framework.
"""
import argparse
import io
import json
import os
import re
import threading
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

import nes_apu as apu

HERE = os.path.dirname(os.path.abspath(__file__))
SR = apu.SR

# Renders are pure functions of their query string, so memoise them. The full
# song is ~40 ms and the speaker model ~270 ms; caching keeps sliders instant.
_cache = {}
_lock = threading.Lock()


def memo(key, fn):
    with _lock:
        if key in _cache:
            return _cache[key]
    val = fn()
    with _lock:
        _cache[key] = val
    return val


# ---------------------------------------------------------------------------
# renders -- all of these call straight into nes_apu
# ---------------------------------------------------------------------------
def channels():
    """The four raw channel signals, 0..15, exactly as __main__ builds them."""
    def build():
        mp, mv = apu.render_tonal(apu.MELODY,  'pulse', 15, 9)
        hp, hv = apu.render_tonal(apu.HARMONY, 'pulse', 10, 6)
        bp, bv = apu.render_tonal(apu.BASS,    'tri',    1, 1)
        np_, nv = apu.render_drums()
        return dict(
            p1=apu.gen_pulse(mp, mv, duty=2),
            p2=apu.gen_pulse(hp, hv, duty=1),
            tri=apu.gen_triangle(bp, bv),
            noise=apu.gen_noise(np_, nv),
        )
    return memo('channels', build)


def song(voices, spk, f0, q):
    """Mix the requested subset of voices, optionally through the cone."""
    ch = channels()
    z = np.zeros(len(ch['p1']))
    sig = apu.mix(*(ch[k] if k in voices else z for k in ('p1', 'p2', 'tri', 'noise')))
    if spk:
        sig = apu.speaker(sig, f0=f0, q=q)
    return sig


def full_mix_peak():
    """Peak of the complete mix, measured the way wav_bytes measures it (after
    mean removal), so a solo levelled against it lands where it really sits."""
    def build():
        s = song(('p1', 'p2', 'tri', 'noise'), False, 650.0, 1.1)
        return float(np.max(np.abs(s - np.mean(s))))
    return memo('peak', build)


def one_note(hz, duty, seconds, arp):
    """A single audition tone -- or an arpeggio, flipping pitch every frame.

    This is the driver's chord trick: a pulse channel is monophonic, so it
    rewrites $4002/$4003 sixty times a second and lets the ear fuse the result.
    """
    frames = max(1, int(round(seconds * apu.FPS)))
    if arp:
        ratios = [1.0, 2 ** (4 / 12), 2 ** (7 / 12)]      # root, major 3rd, 5th
        per = np.array([apu.pulse_period(hz * ratios[i % 3]) for i in range(frames)],
                       dtype=np.float64)
    else:
        per = np.full(frames, float(apu.pulse_period(hz)))
    vol = apu.envelope(frames, start=15, sustain=9)
    vol[-1] = 0                                            # release, avoids a click
    p1 = apu.gen_pulse(per, vol, duty=duty)
    z = np.zeros(len(p1))
    return apu.mix(p1, z, z, z)


def wav_bytes(sig, peak=None, gain=0.85):
    """Float signal -> 16-bit mono WAV in memory.

    Identical maths to nes_apu.write_wav, except that `peak` lets a solo be
    normalised against the full mix instead of against itself -- which is what
    you want when A/B-ing channels, since the triangle really is the loudest.
    """
    s = sig - np.mean(sig)
    denom = peak if peak else np.max(np.abs(s))
    s = s / (denom + 1e-12) * gain
    pcm = (np.clip(s, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# JSON payloads for the oscilloscopes -- also computed by the emulator, so the
# pictures cannot disagree with the sound
# ---------------------------------------------------------------------------
def meta():
    rows = []
    for n in ["G2", "C3", "E3", "A3", "C4", "E4", "G4",
              "C5", "E5", "G5", "A5", "C6", "E6"]:
        hz = apu.note_hz(n)
        t = apu.pulse_period(hz)
        got = apu.pulse_actual_hz(t)
        rows.append(dict(note=n, want=round(hz, 2), timer=t,
                         lo=t & 0xFF, hi=t >> 8, got=round(got, 2),
                         cents=round(1200 * np.log2(got / hz), 1)))
    events = len(apu.MELODY) + len(apu.HARMONY) + len(apu.BASS)
    events += (apu.N_FRAMES // 96) * 12               # 8 hats + 2 kicks + 2 snares
    return dict(cpu=apu.CPU_HZ, sr=SR, fps=apu.FPS,
                frames=apu.N_FRAMES, seconds=apu.N_FRAMES / apu.FPS,
                events=events, registers=rows)


def curve_dac():
    """The pulse ladder's transfer curve, from nes_apu.mix itself."""
    n = np.arange(0, 31)
    z = np.zeros(len(n))
    out = apu.mix(n.astype(float), z, z, z)
    tri = apu.mix(z, z, np.full(len(n), 15.0), z)[0]
    noi = apu.mix(z, z, z, np.full(len(n), 15.0))[0]
    return dict(x=n.tolist(), y=[round(v, 5) for v in out],
                linear=[round(v * out[-1] / 30, 5) for v in n],
                full=dict(pulse=round(float(out[15]), 3),   # one channel at 15
                          tri=round(float(tri), 3), noise=round(float(noi), 3)))


def curve_duty(duty):
    """Two cycles of a pulse channel at the requested duty."""
    hz = apu.note_hz('C5')
    per = np.full(4, float(apu.pulse_period(hz)))
    sig = apu.gen_pulse(per, np.full(4, 15.0), duty=duty)
    n = int(round(2 * SR / apu.pulse_actual_hz(apu.pulse_period(hz))))
    return dict(y=[round(float(v), 3) for v in sig[:n]], duty=duty)


def curve_speaker(f0, q):
    """Coil drive vs. cone displacement for a short burst of C5."""
    hz = apu.note_hz('C5')
    per = np.full(8, float(apu.pulse_period(hz)))
    drive = apu.gen_pulse(per, np.full(8, 15.0), duty=2)
    cone = apu.speaker(drive, f0=f0, q=q)
    n = int(round(3 * SR / apu.pulse_actual_hz(apu.pulse_period(hz))))
    s, e = SR // 20, SR // 20 + n                      # skip the filter's warm-up
    d, c = drive[s:e], cone[s:e]
    c = c / (np.max(np.abs(c)) + 1e-12) * np.max(np.abs(d))
    return dict(drive=[round(float(v), 3) for v in d],
                cone=[round(float(v), 3) for v in c])


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
VOICE_KEYS = ('p1', 'p2', 'tri', 'noise')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass                                            # keep the console quiet

    def _send(self, body, ctype, cache=True):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'max-age=31536000' if cache else 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), 'application/json', cache=False)

    def _fail(self, code, msg):
        body = json.dumps(dict(error=msg)).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)

        def num(name, default, lo, hi):
            try:
                return float(np.clip(float(qs.get(name, [default])[0]), lo, hi))
            except (TypeError, ValueError):
                return float(default)

        try:
            if u.path in ('/', '/index.html'):
                with open(os.path.join(HERE, 'ui.html'), 'rb') as f:
                    return self._send(f.read(), 'text/html; charset=utf-8', cache=False)

            if u.path == '/api/meta':
                return self._json(meta())
            if u.path == '/api/curve/dac':
                return self._json(curve_dac())
            if u.path == '/api/curve/duty':
                return self._json(curve_duty(int(num('duty', 2, 0, 3))))
            if u.path == '/api/curve/speaker':
                return self._json(curve_speaker(num('f0', 650, 120, 1600),
                                                num('q', 1.1, 0.3, 4.0)))

            if u.path == '/api/song':
                want = qs.get('voices', ['p1,p2,tri,noise'])[0]
                voices = tuple(v for v in VOICE_KEYS if v in want.split(','))
                if not voices:
                    return self._fail(400, 'no voices selected')
                spk = qs.get('speaker', ['0'])[0] == '1'
                f0, q = num('f0', 650, 120, 1600), num('q', 1.1, 0.3, 4.0)
                level = qs.get('level', ['self'])[0]
                key = f'song:{",".join(voices)}:{spk}:{f0}:{q}:{level}'
                peak = full_mix_peak() if level == 'mix' else None
                body = memo(key, lambda: wav_bytes(song(voices, spk, f0, q), peak))
                return self._send(body, 'audio/wav')

            if u.path == '/api/note':
                hz = num('hz', 523.25, 20, 12000)
                duty = int(num('duty', 2, 0, 3))
                secs = num('secs', 0.5, 0.05, 4.0)
                arp = qs.get('arp', ['0'])[0] == '1'
                key = f'note:{hz}:{duty}:{secs}:{arp}'
                body = memo(key, lambda: wav_bytes(one_note(hz, duty, secs, arp)))
                return self._send(body, 'audio/wav')

            return self._fail(404, 'no such endpoint')
        except FileNotFoundError:
            self._fail(500, 'ui.html is missing -- it must sit next to serve.py')
        except Exception as e:                          # surface it in the UI
            self._fail(500, f'{type(e).__name__}: {e}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--no-open', action='store_true', help="don't open a browser")
    a = p.parse_args()

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f'http://{a.host}:{a.port}'
    print(f'2A03 signal path -> {url}')
    print('rendering with nes_apu.py; ctrl-c to stop')
    if not a.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    main()
