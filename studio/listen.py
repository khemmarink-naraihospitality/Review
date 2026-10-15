#!/usr/bin/env python3
"""หู AI: จำลองลำโพงมือถือ แล้วใช้โมเดล AudioSet (CED) บอกว่าได้ยินอะไรทีละช่วงเวลา

  python3 listen.py clip.mp4            # รายงานรายช่วง
ใช้ใน qa.py เพื่อยืนยันว่า "ได้ยินดนตรี" จริง ไม่ใช่แค่มีสัญญาณ
"""
import functools, subprocess, sys, tempfile, os
import numpy as np, soundfile as sf, sherpa_onnx

M = "/opt/vstudio/models/sherpa-onnx-ced-base-audio-tagging-2024-04-19/"
SR = 16000


@functools.lru_cache(1)
def _tagger():
    cfg = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(ced=M + "model.int8.onnx", num_threads=4),
        labels=M + "class_labels_indices.csv", top_k=15)
    return sherpa_onnx.AudioTagging(cfg)


def load(path):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), tmp], check=True)
    x, _ = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    return x


def phone(x):
    """ลำโพงมือถือ: ตัดต่ำกว่า ~350Hz และสูงกว่า ~7kHz"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    hp = 1 / np.sqrt(1 + (350 / np.maximum(f, 1)) ** 8)
    lp = 1 / np.sqrt(1 + (f / 7000) ** 8)
    return np.fft.irfft(X * hp * lp, len(x)).astype(np.float32)


def tags(seg):
    s = _tagger().create_stream()
    s.accept_waveform(SR, seg)
    return {e.name: e.prob for e in _tagger().compute(s)}


MUSIC_KEYS = ("Music", "Musical instrument", "Guitar", "Plucked string instrument", "Piano", "Keyboard (musical)",
              "Synthesizer", "Electronic music", "Ambient music", "Background music", "Pop music", "Drum", "Harp",
              "Acoustic guitar", "Music of Asia", "Lullaby", "Bass guitar", "Percussion", "Mallet percussion")


def report(x, win=1.0, hop=0.5, sim_phone=True):
    y = phone(x) if sim_phone else x
    rows = []
    for t in np.arange(0, len(y) / SR - win + 1e-6, hop):
        tg = tags(y[int(t * SR):int((t + win) * SR)])
        music = max([tg.get(k, 0.0) for k in MUSIC_KEYS] + [0.0])
        rows.append({"t": round(float(t), 2), "music": round(music, 2), "speech": round(tg.get("Speech", 0.0), 2),
                     "top": sorted(tg.items(), key=lambda kv: -kv[1])[:3]})
    return rows


if __name__ == "__main__":
    x = load(sys.argv[1])
    for r in report(x):
        top = ", ".join(f"{n} {p:.2f}" for n, p in r["top"])
        print(f'{r["t"]:4.1f}s  ดนตรี {r["music"]:.2f}  เสียงพูด {r["speech"]:.2f}  | {top}')
