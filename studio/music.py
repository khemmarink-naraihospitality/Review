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


def make(seconds=10.0, bpm=92, seed=7, fade=0.8):
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
    fi, fo = int(0.25 * SR), int(fade * SR)
    st[:fi] *= np.linspace(0, 1, fi)[:, None]
    st[-fo:] *= (np.linspace(1, 0, fo) ** 1.3)[:, None]
    return (st / np.abs(st).max() * 0.7).astype(np.float32)


if __name__ == "__main__":
    out = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    bpm = float(sys.argv[3]) if len(sys.argv) > 3 else 92
    sf.write(out, make(secs, bpm), SR)
    print("saved", out)


# ---------------------------------------------------------------- แนวสนุก (upbeat pop) + outro นุ่มจนจบ
FUN_CHORDS = [[("C", 4), ("E", 4), ("G", 4)], [("G", 3), ("B", 3), ("D", 4)],
              [("A", 3), ("C", 4), ("E", 4)], [("F", 3), ("A", 3), ("C", 4)]]          # I-V-vi-IV
FUN_ROOTS = [("C", 2), ("G", 1), ("A", 1), ("F", 1)]
MELODY = [(0, 4, 0.5), (0.5, 7, 0.5), (1.0, 9, 0.5), (1.5, 7, 0.5), (2.5, 4, 0.5), (3.0, 2, 0.5), (3.5, 4, 0.5),
          (4.0, 7, 0.75), (5.0, 9, 0.5), (5.5, 12, 0.5), (6.0, 11, 0.5), (6.5, 9, 0.5), (7.0, 7, 1.0)]   # (beat, semitone จาก C5, ยาว)


def tone(f, dur, decay, harm=(1.0, 0.35, 0.12)):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harm))
    return x * np.exp(-t * decay) * adsr(n, 0.004, min(0.05, dur / 3))


def clap(rng):
    n = int(0.18 * SR)
    x = rng.normal(0, 1, n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    x = np.fft.irfft(X * np.exp(-((f - 1500) / 900) ** 2), n)          # band-pass ~1.5kHz
    env = np.exp(-np.arange(n) / SR * 22)
    for d in (0.0, 0.011, 0.022):                                      # เสียงตบมือซ้อน 3 ครั้ง
        i = int(d * SR)
        env[i:i + 200] += np.linspace(0.6, 0, 200)
    return x / (np.abs(x).max() + 1e-9) * env


def hat(rng, dur=0.05):
    n = int(dur * SR)
    x = rng.normal(0, 1, n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    x = np.fft.irfft(X * (f > 6500), n)
    return x / (np.abs(x).max() + 1e-9) * np.exp(-np.arange(n) / SR * 70)


def make_fun(seconds=20.0, bpm=118, outro_at=None, seed=11):
    """outro_at = วินาทีที่เริ่มช่วงจบ: กลองหยุด เหลือคอร์ดนุ่ม + โน้ตดีดช้าๆ แล้วจางหายพอดีวินาทีสุดท้าย"""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    bar = beat * 4
    n = int(seconds * SR)
    outro_at = seconds - 3.0 if outro_at is None else outro_at
    L, R = np.zeros(n + SR * 3), np.zeros(n + SR * 3)

    def put(x, ts, g=1.0, pan=0.5):
        a = int(ts * SR)
        if a >= n:
            return
        L[a:a + len(x)] += x * g * (1 - pan) * 2 * 0.5
        R[a:a + len(x)] += x * g * pan * 2 * 0.5

    nb = int(np.ceil(seconds / bar))
    for b in range(nb):
        t0 = b * bar
        ch, root = FUN_CHORDS[b % 4], FUN_ROOTS[b % 4]
        intro = b == 0
        for k in range(8):                                   # ทีละ 1/8 โน้ต
            ts = t0 + k * beat / 2
            if ts >= outro_at:
                break
            if not intro:                                    # เบสเด้งออกเทฟ
                f = hz(*root) * (2 if k % 2 else 1) * 2
                put(tone(f, beat / 2 * 0.9, 9, (1.0, 0.5, 0.25, 0.1)), ts, 0.55)
            if k % 2 == 1:                                   # คอร์ดตีจังหวะยก
                for (nm, o), pan in zip(ch, (0.3, 0.5, 0.7)):
                    put(tone(hz(nm, o), beat * 0.4, 7), ts, 0.16, pan)
            put(hat(rng), ts, 0.10 if k % 2 == 0 else 0.07, 0.65)
        for k in range(4):
            ts = t0 + k * beat
            if ts >= outro_at or intro:
                continue
            put(kick(0.3), ts, 0.45)
            if k in (1, 3):
                put(clap(rng), ts, 0.22, 0.45)
        if not intro and b % 2 == 1:                         # ทำนองติดหูทุก 2 ห้อง
            for bt, semi, ln in MELODY:
                ts = t0 - bar + bt * beat
                if ts >= outro_at:
                    break
                f = hz("C", 5) * 2 ** (semi / 12)
                put(pluck(f, ln * beat + 0.3, rng), ts, 0.30, 0.55)
    # ค่อยๆ ลดช่วงสนุกลง 1.2 วินาที ก่อนเข้าช่วงจบ (ไม่ตัดกุด)
    r0, r1 = int((outro_at - 1.0) * SR), int((outro_at + 0.2) * SR)
    ramp = np.ones(len(L))
    ramp[r0:r1] = np.cos(np.linspace(0, np.pi / 2, r1 - r0)) ** 2 * 0.85 + 0.15
    ramp[r1:] = 0.15
    L *= ramp
    R *= ramp
    # ---- outro: คอร์ดหลัก Cmaj9 นุ่มๆ + ดีด arpeggio ช้าลงเรื่อยๆ
    od = seconds - outro_at + 1.0
    padc = pad([hz("C", 3), hz("G", 3), hz("B", 3), hz("D", 4), hz("E", 4)], od)
    put(padc, outro_at - 0.05, 0.55)
    arp = [hz("C", 5), hz("E", 5), hz("G", 5), hz("B", 5), hz("D", 6)]
    ts, step, j = outro_at, beat / 2, 0
    while ts < seconds - 0.6:
        put(pluck(arp[j % len(arp)], 1.6, rng), ts, 0.22 * (1 - (ts - outro_at) / (seconds - outro_at)) + 0.05, 0.35 + 0.3 * (j % 2))
        ts += step
        step *= 1.12                                          # ช้าลงทีละนิดเหมือนเพลงกำลังจบ
        j += 1
    L, R = reverb(L[:n], rng, 1.8, 0.22), reverb(R[:n], np.random.default_rng(seed + 1), 1.8, 0.22)
    st = np.stack([highpass(lowpass(L, 7000), 70), highpass(lowpass(R, 7000), 70)], 1)
    fi = int(0.15 * SR)
    st[:fi] *= np.linspace(0, 1, fi)[:, None]
    i0 = int(outro_at * SR)
    m = n - i0                                                 # จางแบบโค้ง cosine จนเงียบสนิทที่วินาทีสุดท้าย
    st[i0:] *= (0.55 + 0.45 * np.cos(np.linspace(0, np.pi, m)))[:, None] ** 1.6
    return (st / np.abs(st).max() * 0.7).astype(np.float32)
