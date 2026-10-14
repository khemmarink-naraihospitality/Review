"""ถอดเสียงไทยด้วย Whisper (ออฟไลน์) เพื่อตรวจว่าเสียงพากย์ฟังรู้เรื่องจริง"""
import sherpa_onnx, soundfile as sf, numpy as np, functools
M = "/opt/vstudio/models/sherpa-onnx-whisper-small/small-"

@functools.lru_cache(1)
def _rec():
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=M + "encoder.int8.onnx", decoder=M + "decoder.int8.onnx", tokens=M + "tokens.txt",
        language="th", task="transcribe", num_threads=4)

def transcribe(wav_path: str) -> str:
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    s = _rec().create_stream()
    s.accept_waveform(sr, x)
    _rec().decode_stream(s)
    return s.result.text.strip()

if __name__ == "__main__":
    import sys
    print(transcribe(sys.argv[1]))
