#!/usr/bin/env python3
"""ดนตรีพื้นหลังเบาๆ สังเคราะห์เอง (ไม่มีลิขสิทธิ์ผู้อื่น): pad อุ่นๆ + พลักกีตาร์ arpeggio + เพอร์คัสชันเบา + รีเวิร์บ

  python3 music.py out.wav [seconds] [bpm]
"""
import sys, numpy as np, soundfile as sf

SR = 48000
NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def hz(name, octave):
    return 440.0 * 2 ** ((NOTE[name] + 12 * (octave + 1) - 69) / 12)


# Cmaj7 - Am7 - Fmaj7 - G6 (โทนสว่าง สบาย เหมาะกับสินค้าแฟชั่น)
CHORDS = [[("C", 3), ("E", 3), ("G", 3), ("B", 3)], [("A", 2), ("C", 3), ("E", 3), ("G", 3)],
          [("F", 2), ("A", 2), ("C", 3), ("E", 3)], [("G", 2), ("B", 2), ("D", 3), ("E", 3)]]


def adsr(n, a, r):
    env = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    env[:na] = np.linspace(0, 1, na)
    env[n - nr:] *= np.linspace(1, 0, nr)
    return env


def pad(freqs, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        for det in (-0.12, 0.0, 0.12):           # detune เล็กน้อยให้กว้าง
            ff = f * 2 ** (det / 12)
            out += np.sin(2 * np.pi * ff * t) + 0.25 * np.sin(4 * np.pi * ff * t) + 0.08 * np.sin(6 * np.pi * ff * t)
    return out / (len(freqs) * 3) * adsr(n, 0.6, 0.8)


def pluck(f, dur, rng):
    """Karplus-Strong: เสียงดีดสายนุ่มๆ"""
    n = int(dur * SR)
    p = int(SR / f)
    buf = rng.uniform(-1, 1, p)
    buf = np.convolve(buf, np.ones(4) / 4, mode="same")
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = 0.996 * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out * adsr(n, 0.003, 0.2)


def kick(dur=0.35):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 50 + 70 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def shaker(rng, dur=0.06):
    n = int(dur * SR)
    x = rng.normal(0, 1, n)
    x = x - np.convolve(x, np.ones(8) / 8, mode="same")   # high-pass หยาบๆ
    return x * np.exp(-np.arange(n) / SR * 60)


def reverb(x, rng, secs=1.6, mix=0.28):
    n = int(secs * SR)
    ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR * 3.2)
    ir = np.convolve(ir, np.ones(24) / 24, mode="same")      # มืดลงให้นุ่ม
    wet = np.fft.irfft(np.fft.rfft(x, len(x) + n) * np.fft.rfft(ir, len(x) + n))[: len(x)]
    wet /= np.abs(wet).max() + 1e-9
    return (1 - mix) * x / (np.abs(x).max() + 1e-9) + mix * wet


def lowpass(x, fc):
    """กรองความถี่สูงแบบนุ่ม (roll-off โค้ง) ให้ดนตรีอุ่น ไม่แหลม"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X / np.sqrt(1 + (f / fc) ** 6), len(x))


def highpass(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X / np.sqrt(1 + (fc / np.maximum(f, 1)) ** 4), len(x))


def make(seconds=10.0, bpm=92, seed=7):
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    bar = beat * 4
    n = int(seconds * SR)
    L, R = np.zeros(n + SR * 2), np.zeros(n + SR * 2)
    t0, ci = 0.0, 0
    while t0 < seconds:
        ch = CHORDS[ci % 4]
        p = pad([hz(a, o + 1) for a, o in ch], bar + 0.6)
        i = int(t0 * SR)
        L[i:i + len(p)] += 0.55 * p
        R[i:i + len(p)] += 0.55 * p
        notes = [hz(a, o + 2) for a, o in ch] + [hz(ch[1][0], ch[1][1] + 3)]
        order = [0, 1, 2, 3, 4, 3, 2, 1]
        for k, j in enumerate(order):                       # arpeggio โน้ตละ 1/2 จังหวะ
            ts = t0 + k * beat / 2
            if ts >= seconds:
                break
            pl = pluck(notes[j], 0.9, rng) * (0.42 if k % 2 == 0 else 0.30)
            a = int(ts * SR)
            pan = 0.35 + 0.3 * (k % 2)
            L[a:a + len(pl)] += pl * (1 - pan)
            R[a:a + len(pl)] += pl * pan
        for b in range(4):                                  # กลองเบามาก
            ts = t0 + b * beat
            if ts >= seconds:
                break
            a = int(ts * SR)
            if b in (0, 2):
                k_ = kick() * 0.18
                L[a:a + len(k_)] += k_
                R[a:a + len(k_)] += k_
            for off in (0.0, 0.5):
                s = shaker(rng) * 0.05
                a2 = int((ts + off * beat) * SR)
                L[a2:a2 + len(s)] += s * 0.7
                R[a2:a2 + len(s)] += s
        t0 += bar
        ci += 1
    L, R = reverb(L[:n], rng), reverb(R[:n], np.random.default_rng(seed + 1))
    st = np.stack([highpass(lowpass(L, 6000), 160), highpass(lowpass(R, 6000), 160)], 1)
    fi, fo = int(0.25 * SR), int(0.8 * SR)
    st[:fi] *= np.linspace(0, 1, fi)[:, None]
    st[-fo:] *= (np.linspace(1, 0, fo) ** 1.3)[:, None]
    return (st / np.abs(st).max() * 0.7).astype(np.float32)


if __name__ == "__main__":
    out = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    bpm = float(sys.argv[3]) if len(sys.argv) > 3 else 92
    sf.write(out, make(secs, bpm), SR)
    print("saved", out)
