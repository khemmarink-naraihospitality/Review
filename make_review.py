#!/usr/bin/env python3
"""สร้างคลิปรีวิวสินค้า 10 วินาที (1080x1920, 30fps)

ตัวอย่าง:
  python3 make_review.py --name "หูฟังบลูทูธ X1" --points "แบตอึด 30 ชม.|เสียงใส เบสแน่น|น้ำหนักเบา" \
      --price "฿590" --rating 4.5 --image photo.jpg --out x1.mp4
"""
import argparse, math, os, shutil, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DUR = 1080, 1920, 30, 10
FONT_B = "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"
FONT_R = "/usr/share/fonts/opentype/tlwg/Loma.otf"
BG1, BG2, ACCENT, WHITE = (24, 28, 60), (96, 40, 120), (255, 205, 60), (255, 255, 255)


def font(path, size):
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def gradient():
    g = Image.new("RGB", (1, H))
    for y in range(H):
        k = y / H
        g.putpixel((0, y), tuple(int(BG1[i] * (1 - k) + BG2[i] * k) for i in range(3)))
    return g.resize((W, H))


def product_img(path, name):
    size = 640
    if path and os.path.exists(path):
        im = Image.open(path).convert("RGB")
        s = min(im.size)
        l, t = (im.width - s) // 2, (im.height - s) // 2
        im = im.crop((l, t, l + s, t + s)).resize((size, size))
    else:  # placeholder เมื่อไม่มีรูป
        im = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((size // 2, size // 2), "ใส่รูปสินค้า", font=font(FONT_B, 64), fill=(150, 150, 150), anchor="mm")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size, size), 60, fill=255)
    out = Image.new("RGBA", (size, size))
    out.paste(im, mask=mask)
    return out


def centered(d, y, text, f, fill, alpha=1.0, shadow=True):
    if alpha <= 0:
        return
    col = tuple(int(fill[i] * alpha + 40 * (1 - alpha)) for i in range(3))
    if shadow:
        d.text((W // 2 + 3, y + 3), text, font=f, fill=(0, 0, 0), anchor="mm")
    d.text((W // 2, y), text, font=f, fill=col, anchor="mm")


def stars(d, cx, y, rating, size=70):
    for i in range(5):
        x = cx + (i - 2) * (size + 12)
        fill = ACCENT if rating >= i + 1 else ((255, 230, 150) if rating > i else (90, 90, 130))
        pts = []
        for k in range(10):
            r = size / 2 if k % 2 == 0 else size / 4.2
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((x + r * math.cos(a), y + r * math.sin(a)))
        d.polygon(pts, fill=fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--points", required=True, help="จุดเด่น 3 ข้อ คั่นด้วย |")
    ap.add_argument("--price", default="")
    ap.add_argument("--rating", type=float, default=5.0)
    ap.add_argument("--image", default="")
    ap.add_argument("--cta", default="กดสั่งซื้อที่ตะกร้าเลย!")
    ap.add_argument("--audio", default="", help="ไฟล์เพลง/เสียงพากย์ (ไม่บังคับ)")
    ap.add_argument("--out", default="review.mp4")
    a = ap.parse_args()
    points = [p.strip() for p in a.points.split("|") if p.strip()][:3]

    bg, prod = gradient(), product_img(a.image, a.name)
    f_title0, f_pt, f_price, f_cta, f_tag = font(FONT_B, 80), font(FONT_B, 58), font(FONT_B, 130), font(FONT_B, 64), font(FONT_B, 48)
    size = 80
    while size > 40 and font(FONT_B, size).getlength(a.name) > W - 120:
        size -= 2
    f_title = font(FONT_B, size)
    tmp = tempfile.mkdtemp()
    for n in range(FPS * DUR):
        t = n / FPS
        img = bg.copy().convert("RGBA")
        d = ImageDraw.Draw(img)
        # ฉาก 0-2s: สินค้าเด้งเข้า + ชื่อ
        k = ease(t / 0.8)
        sc = 0.6 + 0.4 * k + (0.03 * math.sin(t * 2.5) if t > 1 else 0)
        s = int(640 * sc)
        p = prod.resize((s, s))
        py = 350
        img.alpha_composite(p, ((W - s) // 2, py + (640 - s) // 2))
        d = ImageDraw.Draw(img)
        centered(d, 130, "รีวิวด่วน 10 วิ", f_tag, ACCENT, ease(t / 0.5))
        centered(d, 230, a.name, f_title, WHITE, ease((t - 0.3) / 0.5))
        # ฉาก 2-8s: จุดเด่นทยอยขึ้น
        for i, pt in enumerate(points):
            t0 = 2.2 + i * 1.8
            k = ease((t - t0) / 0.5)
            if k > 0:
                y = 1230 + i * 130
                x_off = int((1 - k) * 300)
                d.rounded_rectangle((90 + x_off, y - 50, W - 90 + x_off, y + 50), 40, fill=(255, 255, 255, int(40 * k)) if False else (60, 60, 110))
                d.line([(140 + x_off, y), (162 + x_off, y + 24), (200 + x_off, y - 26)], fill=ACCENT, width=11, joint="curve")
                d.text((235 + x_off, y), pt, font=f_pt, fill=WHITE, anchor="lm")
        # ดาวคะแนน
        if a.rating > 0 and t >= 1.5:
            stars(d, W // 2, 1060, a.rating * ease((t - 1.5) / 1.0))
        # ฉาก 8-10s: ราคา + CTA
        if t >= 7.8:
            k = ease((t - 7.8) / 0.5)
            d.rectangle((0, 1000, W, H), fill=(24, 28, 60))
            if a.price:
                centered(d, 1330, a.price, f_price, ACCENT, k)
            pulse = 1 + 0.04 * math.sin(t * 8)
            bw, bh = int(820 * pulse), int(150 * pulse)
            d.rounded_rectangle(((W - bw) // 2, 1600 - bh // 2, (W + bw) // 2, 1600 + bh // 2), 75, fill=ACCENT)
            d.text((W // 2, 1600), a.cta, font=f_cta, fill=(30, 30, 30), anchor="mm")
        # แถบเวลา
        d.rectangle((0, H - 14, int(W * t / DUR), H), fill=ACCENT)
        img.convert("RGB").save(f"{tmp}/f{n:04d}.png")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png"]
    if a.audio:
        cmd += ["-i", a.audio, "-shortest", "-af", f"afade=t=out:st={DUR-1}:d=1", "-c:a", "aac"]
    cmd += ["-t", str(DUR), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp)
    print("saved", a.out)


if __name__ == "__main__":
    main()
