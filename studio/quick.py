#!/usr/bin/env python3
"""คลิปสินค้า 10 วินาทีแบบเร็ว (คำสั่งเดียว ~1-2 นาที ฟรี ไม่ใช้บริการภายนอก)

  python3 quick.py --image รูป.jpg --name "iPhone 18" --points "กล้องหลัง 3 ตัว|ดีไซน์โทนเข้ม" --tag "11.11 SALE" --out ชื่อไฟล์.mp4

ทำให้อัตโนมัติ: ตัดพื้นหลัง (เก็บชิ้นใหญ่สุด ตัดป้ายโฆษณาทิ้ง) -> พื้นหลังสตูดิโอเข้ม -> สินค้าลอยหมุนช้า
-> ข้อความเด่นเคลื่อนไหวทีละฉาก -> ฉากจบชื่อสินค้า + ป้ายโปร -> เพลงเบาๆ -> ตรวจ QA
ไม่มีเสียงพากย์ (เลี่ยงปัญหาสัญญาอนุญาตเสียงสังเคราะห์สำหรับงานขาย)
"""
import argparse, json, os, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def cutout(src, outdir):
    import cv2
    os.environ["U2NET_HOME"] = "/opt/vstudio/models/u2net"
    from rembg import new_session, remove
    im = Image.open(src).convert("RGB")
    cut = remove(im, session=new_session("isnet-general-use"))
    a = (np.asarray(cut.getchannel("A")) > 40).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(a)
    if n > 1:                                   # เก็บชิ้นใหญ่สุด (สินค้า) ทิ้งตัวหนังสือ/ป้าย
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        keep = cv2.dilate((lab == k).astype(np.uint8), np.ones((9, 9), np.uint8))
        alpha = np.asarray(cut.getchannel("A")) * keep
        cut.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    cut = cut.crop(cut.getchannel("A").getbbox())
    al = np.asarray(cut.getchannel("A")).astype(np.float32)
    if (al[al > 10] < 200).mean() > 0.35:      # ขอบ/ตัวสินค้าโปร่งใสเยอะ = ตัดไม่ดี (มักเป็นสินค้าสีขาวบนพื้นขาว)
        print("คำเตือน: ตัดพื้นหลังได้ไม่สะอาด (สินค้าโปร่งใส) แนะนำใช้ --cutout หรือรูปที่พื้นหลังตัดกับสีสินค้า")
    os.makedirs(outdir, exist_ok=True)
    cut.save(f"{outdir}/cutout.png")
    im.resize((im.width * 4, im.height * 4), Image.LANCZOS).save(f"{outdir}/photo_x4.png")
    return cut.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--points", default="", help="จุดเด่น 1-2 ข้อ คั่นด้วย |")
    ap.add_argument("--tag", default="สั่งซื้อได้เลย")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutout", help="ใช้ภาพตัดพื้นหลังที่มีอยู่แล้ว (PNG โปร่งใส) แทนการตัดอัตโนมัติ")
    ap.add_argument("--light", action="store_true", help="พื้นหลังสว่าง (สินค้าสีเข้ม/ดำ ใช้ค่าเริ่มต้นพื้นเข้มก็ได้)")
    a = ap.parse_args()
    slug = os.path.splitext(os.path.basename(a.out))[0]
    adir = os.path.join(HERE, "assets", slug)
    if a.cutout:
        os.makedirs(adir, exist_ok=True)
        c = Image.open(a.cutout).convert("RGBA")
        c = c.crop(c.getchannel("A").getbbox())
        c.save(f"{adir}/cutout.png")
        im = Image.open(a.image).convert("RGB")
        im.resize((im.width * 4, im.height * 4), Image.LANCZOS).save(f"{adir}/photo_x4.png")
        w, h = c.size
    else:
        w, h = cutout(a.image, adir)
    s = 0.62 * 720 / h if h / w > 0.6 else 0.55 * 1280 / w     # ให้สินค้าสูง ~62% ของจอ
    pts = [p.strip() for p in a.points.split("|") if p.strip()][:2]
    fg = [20, 33, 61] if a.light else [255, 255, 255]
    bg = [[205, 209, 217], [249, 250, 252]] if a.light else [[8, 10, 18], [46, 52, 78]]
    word = lambda t: {"word": t, "word_color": fg, "word_y": 0.12, "word_size": 110}
    shots = [dict(kind="float", scale=[s * 0.88, s * 0.96], angle=[-6, 0], cy=0.58, **word(a.name))]
    for i, p in enumerate(pts):
        shots.append(dict(kind="float", scale=[s * 0.9, s * 0.97], angle=[8 if i == 0 else -4, 2], cy=0.58, **word(p)))
    shots.append(dict(kind="end", scale=[s * 0.72, s * 0.78], angle=[0, -3], cx=0.28, cy=0.5, text=a.name, text_color=fg, tag=a.tag))
    secs = {2: [3.0, 5.3], 3: [2.4, 2.4, 3.8], 4: [2.0, 2.0, 1.9, 2.3]}[len(shots)]
    spec = {"size": [1280, 720], "duration": 10.0, "assets": os.path.relpath(adir, os.path.join(HERE, "specs")),
            "subtitles": False, "music": True, "music_style": "soft", "music_lufs": -18.0, "music_fade": 1.4, "voice": False,
            "bg_colors": bg, "brand_font": "montserrat", "brand_size": 60, "marquee": [], "fade_out": 0.5,
            "segments": [{"text": "x", "silent": v} for v in secs], "shots": shots,
            "out": os.path.relpath(os.path.abspath(a.out), os.path.join(HERE, "specs"))}
    sp = os.path.join(HERE, "specs", slug + ".json")
    json.dump(spec, open(sp, "w"), ensure_ascii=False, indent=2)
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "cinematic.py"), sp]))


if __name__ == "__main__":
    main()
