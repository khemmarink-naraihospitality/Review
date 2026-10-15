#!/usr/bin/env bash
# ติดตั้งชุดเครื่องมือสร้างวิดีโอรีวิวเสียงไทย (ฟรีทั้งหมด ทำงานออฟไลน์หลังติดตั้ง)
set -euo pipefail
M=/opt/vstudio/models; mkdir -p "$M"
pip install -q sherpa-onnx numpy soundfile pythainlp pyloudnorm pillow "rembg[cpu]" opencv-contrib-python-headless
apt-get install -y -q ffmpeg fonts-thai-tlwg >/dev/null
# เสียงไทย: Meta MMS-TTS (VITS) แปลงเป็น ONNX โดย sherpa-onnx
[ -f "$M/vits-mms-tha/model.onnx" ] || curl -sSL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-tha.tar.bz2 | tar -xj -C "$M"
# ตัวตรวจ: Whisper small (ถอดเสียงไทยกลับมาเทียบกับบท)
[ -f "$M/sherpa-onnx-whisper-small/small-encoder.int8.onnx" ] || curl -sSL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2 | tar -xj -C "$M"
echo "พร้อมใช้งาน: python3 build.py specs/typec.json"
# สไตล์สตูดิโอ: โมเดลตัดพื้นหลัง + ขยายภาพ x4
mkdir -p "$M/u2net" "$M/sr"
[ -f "$M/u2net/isnet-general-use.onnx" ] || curl -sSL -o "$M/u2net/isnet-general-use.onnx" https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx
[ -f "$M/sr/EDSR_x4.pb" ] || curl -sSL -o "$M/sr/EDSR_x4.pb" https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb
echo "สไตล์สตูดิโอ: python3 prep_assets.py รูป.jpg assets/ชื่อ  แล้ว python3 cinematic.py specs/xxx_cine.json"
# หู AI: AudioSet tagger (CED) ตรวจว่าได้ยินดนตรี/เสียงพูดจริง ผ่านลำโพงมือถือจำลอง
[ -f "$M/sherpa-onnx-ced-base-audio-tagging-2024-04-19/model.int8.onnx" ] || curl -sSL https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-ced-base-audio-tagging-2024-04-19.tar.bz2 | tar -xj -C "$M"
