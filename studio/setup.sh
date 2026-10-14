#!/usr/bin/env bash
# ติดตั้งชุดเครื่องมือสร้างวิดีโอรีวิวเสียงไทย (ฟรีทั้งหมด ทำงานออฟไลน์หลังติดตั้ง)
set -euo pipefail
M=/opt/vstudio/models; mkdir -p "$M"
pip install -q sherpa-onnx numpy soundfile pythainlp pyloudnorm pillow
apt-get install -y -q ffmpeg fonts-thai-tlwg >/dev/null
# เสียงไทย: Meta MMS-TTS (VITS) แปลงเป็น ONNX โดย sherpa-onnx
[ -f "$M/vits-mms-tha/model.onnx" ] || curl -sSL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-tha.tar.bz2 | tar -xj -C "$M"
# ตัวตรวจ: Whisper small (ถอดเสียงไทยกลับมาเทียบกับบท)
[ -f "$M/sherpa-onnx-whisper-small/small-encoder.int8.onnx" ] || curl -sSL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2 | tar -xj -C "$M"
echo "พร้อมใช้งาน: python3 build.py specs/typec.json"
