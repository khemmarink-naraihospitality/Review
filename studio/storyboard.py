#!/usr/bin/env python3
"""สตอรี่บอร์ดจากคลิปที่เรนเดอร์แล้ว: ต่อฉาก = ภาพต้น/กลาง/ท้าย + เวลา + บทพากย์ + มุมกล้อง

  python3 storyboard.py clip.mp4 specs/xxx.json   -> clip.mp4.storyboard.png
"""
import json, subprocess, sys, tempfile, os
from PIL import Image, ImageDraw, ImageFont

F = "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"
CAM = {"hero": "มุมกว้าง หมุนช้า + ซูมเข้า + แสงกวาด", "detail": "ซูมใกล้ (ภาพขยาย x4) เลื่อนช้า",
       "pair": "เลื่อนกล้องผ่านรองเท้าสองข้าง", "single": "รองเท้าข้างเดียว เอียงช้าๆ", "hero2": "หมุนกลับอีกทาง + ซูมเข้า", "end": "ตัดจบ ชื่อแบรนด์ + ซูมช้า"}


def grab(video, t, w):
    p = tempfile.mktemp(suffix=".png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                    "-vf", f"scale={w}:-1", p], check=True)
    im = Image.open(p).convert("RGB"); os.remove(p)
    return im


def main(video, spec_path):
    spec = json.load(open(spec_path))
    tm = json.load(open(video + ".timing.json"))
    b = tm.get("bounds") or [0] + [s["end"] for s in tm["segments"]]
    tw = 300
    f1, f2 = ImageFont.truetype(F, 26), ImageFont.truetype(F, 22)
    rows = []
    for i, (seg, sh) in enumerate(zip(tm["segments"], spec.get("shots", [{}] * len(tm["segments"])))):
        t0, t1 = b[i], b[i + 1]
        ims = [grab(video, t0 + (t1 - t0) * k, tw) for k in (0.12, 0.5, 0.88)]
        th = ims[0].height
        row = Image.new("RGB", (tw * 3 + 560, th + 20), (246, 246, 248))
        for k, im in enumerate(ims):
            row.paste(im, (10 + k * (tw + 5), 10))
        d = ImageDraw.Draw(row)
        x = tw * 3 + 40
        d.text((x, 18), f"ฉาก {i + 1}  {t0:.2f}–{t1:.2f} วิ", font=f1, fill=(20, 24, 40))
        d.text((x, 60), f"พากย์: “{seg['text']}”", font=f2, fill=(60, 64, 80))
        d.text((x, 95), f"กล้อง: {CAM.get(sh.get('kind'), '-')}", font=f2, fill=(60, 64, 80))
        rows.append(row)
    W = rows[0].width
    H = sum(r.height for r in rows) + 70
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    ImageDraw.Draw(sheet).text((14, 18), f"สตอรี่บอร์ด: {os.path.basename(video)}  (ดนตรีพื้นหลัง: {'มี' if spec.get('music') else 'ไม่มี'})",
                               font=ImageFont.truetype(F, 30), fill=(20, 24, 40))
    y = 70
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    out = video + ".storyboard.png"
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
