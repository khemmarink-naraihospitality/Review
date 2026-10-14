"""สังเคราะห์เสียงไทย (MMS-TTS ผ่าน sherpa-onnx, ออฟไลน์) พร้อมแคชต่อประโยค"""
import hashlib, os, functools, numpy as np, soundfile as sf, sherpa_onnx
from thai_text import normalize

M = "/opt/vstudio/models/vits-mms-tha/"
CACHE = os.path.join(os.path.dirname(__file__), "cache")
SR = 16000

@functools.lru_cache(1)
def _tts():
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=M + "model.onnx", tokens=M + "tokens.txt",
                                                   noise_scale=0.4, noise_scale_w=0.5, length_scale=1.0),
        num_threads=4))
    return sherpa_onnx.OfflineTts(cfg)

def _trim(x, thr=0.004, keep=0.10):
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x
    return x[max(0, idx[0] - int(keep * SR)): idx[-1] + int(keep * SR)]

def synth(say: str, speed: float = 1.0) -> np.ndarray:
    """say = ข้อความดิบ (จะ normalize ให้) -> float32 mono 16 kHz, ตัดช่วงเงียบหัวท้าย"""
    text = normalize(say)
    key = hashlib.sha1(f"{text}|{speed}".encode()).hexdigest()[:16]
    path = os.path.join(CACHE, key + ".wav")
    if os.path.exists(path):
        return sf.read(path, dtype="float32")[0]
    a = _tts().generate(text, sid=0, speed=speed)
    x = _trim(np.asarray(a.samples, dtype=np.float32))
    sf.write(path, x, SR)
    return x
