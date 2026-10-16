"""ธีมฉากธรรมชาติยามเย็น (twilight) + กราฟิกสาธิตการใช้งาน สำหรับ cinematic.py

- ฉาก: ท้องฟ้าไล่สีม่วง-ชมพู, ภูเขาไกล, ผิวน้ำสะท้อน, หญ้าลาเวนเดอร์ไหวด้านหน้า, แสงโบเก้ลอย
- ช็อต: connect (มือถือ → ตัวแปลง → หูฟัง), uses (MUSIC / CALLS / GAMES)
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

GLYPH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK = (52, 44, 96)          # ม่วงเข้ม (ไอคอน/หญ้า)
LAV = (168, 140, 214)       # ลาเวนเดอร์


def ease(p):
    p = max(0.0, min(1.0, p))
    return p * p * (3 - 2 * p)


def lerp_rows(stops, n):
    ys = np.array([s[0] for s in stops]) * n
    cols = np.array([s[1] for s in stops], np.float32)
    y = np.arange(n)
    return np.stack([np.interp(y, ys, cols[:, c]) for c in range(3)], 1)


def build(W, H, seed=3):
    """เตรียมภาพนิ่งของฉาก + ข้อมูลหญ้า/โบเก้ (ทำครั้งเดียว)"""
    rng = np.random.default_rng(seed)
    hy = int(H * 0.64)
    sky = lerp_rows([(0, (46, 48, 98)), (0.45, (128, 112, 186)), (0.85, (226, 176, 196)), (1.0, (246, 206, 196))], hy)
    water = lerp_rows([(0, (150, 128, 182)), (0.35, (104, 92, 152)), (1.0, (40, 42, 86))], H - hy)
    img = np.zeros((H, W, 3), np.float32)
    img[:hy] = sky[:, None, :]
    img[hy:] = water[:, None, :]
    xs = np.arange(W)
    ridge = hy - (14 + 10 * np.sin(xs / 140.0) + 7 * np.sin(xs / 53.0 + 1.3) + 5 * np.sin(xs / 23.0 + 0.4))
    for x in range(W):                                   # ภูเขาไกล (โทนจาง)
        img[int(ridge[x]):hy, x] = img[int(ridge[x]):hy, x] * 0.55 + np.array([112, 98, 150]) * 0.45
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sun = np.exp(-(((xx - W * 0.52) / (W * 0.30)) ** 2 + ((yy - hy) / (H * 0.16)) ** 2))
    img += sun[..., None] * np.array([46, 22, 6], np.float32)          # แสงอุ่นยามเย็นตรงขอบฟ้า
    img = np.asarray(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)
    blades = []
    for side in (0, 1):                                  # หญ้าลาเวนเดอร์มุมซ้าย/ขวาล่าง
        for _ in range(46):
            bx = rng.uniform(-40, W * 0.30) if side == 0 else rng.uniform(W * 0.70, W + 40)
            blades.append((bx, rng.uniform(150, 330), rng.uniform(-0.35, 0.35), rng.uniform(0, 6.28), rng.uniform(1.5, 3.5),
                           rng.random() < 0.55))
    bokeh = [(rng.uniform(0, W), rng.uniform(0, hy), rng.uniform(6, 22), rng.uniform(6, 16), rng.uniform(0, 6.28)) for _ in range(26)]
    return {"bg": np.clip(img, 0, 255), "hy": hy, "blades": blades, "bokeh": bokeh, "W": W, "H": H}


def scene(SC, W, H, t):
    """ฉากพื้นหลังของเวลา t (RGBA)"""
    im = Image.fromarray(SC["bg"].astype(np.uint8)).convert("RGBA")
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for x, y, r, sp, ph in SC["bokeh"]:                  # โบเก้ลอยขึ้นช้าๆ
        yy = (y - sp * t) % SC["hy"]
        a = int(40 + 30 * math.sin(t * 1.3 + ph))
        gd.ellipse((x - r, yy - r, x + r, yy + r), fill=(255, 236, 240, a))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(4)))
    return im


def grass(SC, W, H, t):
    """ชั้นหญ้าด้านหน้า (วาดทับสินค้าได้เล็กน้อยที่มุม ให้มีมิติ)"""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for bx, h, lean, ph, wd, flower in SC["blades"]:
        sway = lean + 0.06 * math.sin(t * 1.5 + ph)
        pts = []
        for k in range(7):
            q = k / 6
            pts.append((bx + math.sin(sway) * h * q * q * 1.2, H + 8 - h * q))
        d.line(pts, fill=INK + (235,), width=int(wd), joint="curve")
        if flower:
            tx, ty = pts[-1]
            for j in range(6):
                d.ellipse((tx - 4 - j * math.sin(sway) * 3, ty + j * 9 - 4, tx + 4 - j * math.sin(sway) * 3, ty + j * 9 + 6), fill=LAV + (240,))
    return lay.filter(ImageFilter.GaussianBlur(1.6))       # ระยะชัดตื้น: หญ้าด้านหน้านุ่มลง


def reflect(layer, x, y, gap, strength=0.42):
    """เงาสะท้อนบนผิวน้ำ: กลับหัว จางลงด้านล่าง เบลอนิด"""
    r = layer.transpose(Image.FLIP_TOP_BOTTOM).filter(ImageFilter.GaussianBlur(2.5))
    a = np.asarray(r.getchannel("A")).astype(np.float32)
    fade = np.linspace(strength, 0, r.height)[:, None] ** 1.3
    r.putalpha(Image.fromarray((a * fade).astype(np.uint8)))
    return r, (x, y + layer.height + gap)


# ---------------------------------------------------------------- ไอคอน (วาดเอง ไม่ใช้รูปคนอื่น)
def icon_phone(d, cx, cy, s, col):
    w, h = 0.55 * s, s
    d.rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), int(0.12 * s), outline=col, width=max(3, int(s * 0.05)))
    d.rounded_rectangle((cx - w * 0.16, cy + h / 2 - s * 0.03, cx + w * 0.16, cy + h / 2 + s * 0.035), 4, fill=col)   # ช่อง USB-C


def icon_headphones(d, cx, cy, s, col):
    lw = max(4, int(s * 0.07))
    d.arc((cx - s * 0.42, cy - s * 0.45, cx + s * 0.42, cy + s * 0.38), 180, 360, fill=col, width=lw)
    for sx in (-1, 1):
        x = cx + sx * s * 0.40
        d.rounded_rectangle((x - s * 0.11, cy - s * 0.02, x + s * 0.11, cy + s * 0.34), int(s * 0.07), fill=col)


def icon_gamepad(d, cx, cy, s, col, bg):
    d.rounded_rectangle((cx - s * 0.48, cy - s * 0.24, cx + s * 0.48, cy + s * 0.26), int(s * 0.22), fill=col)
    a, t = s * 0.13, s * 0.045
    x0 = cx - s * 0.24
    d.rectangle((x0 - a, cy - t, x0 + a, cy + t), fill=bg)
    d.rectangle((x0 - t, cy - a, x0 + t, cy + a), fill=bg)
    for dx, dy in ((0.20, -0.05), (0.30, 0.06)):
        x, y = cx + dx * s, cy + dy * s
        d.ellipse((x - s * 0.05, y - s * 0.05, x + s * 0.05, y + s * 0.05), fill=bg)


def label(d, x, y, text, font, a):
    """ป้ายตัวหนังสือบนแถบเข้มนุ่ม อ่านง่ายแม้มีหญ้าด้านหลัง"""
    w = font.getlength(text)
    d.rounded_rectangle((x - w / 2 - 16, y - 21, x + w / 2 + 16, y + 21), 21, fill=(30, 26, 60, int(150 * a)))
    d.text((x, y), text, font=font, fill=(255, 255, 255, int(245 * a)), anchor="mm")


def glyph(d, cx, cy, s, ch, col):
    f = ImageFont.truetype(GLYPH, int(s))
    d.text((cx, cy), ch, font=f, fill=col, anchor="mm")


def connect(SC, W, H, t, tl, p, cut, fonts):
    """มือถือ ⟶ ตัวแปลง ⟶ หูฟัง: ขึ้นทีละชิ้น แล้วจุดเสียงวิ่งตามเส้น"""
    im = scene(SC, W, H, t)
    im.alpha_composite(grass(SC, W, H, t))          # หญ้าอยู่หลังกราฟิก ไม่บังตัวหนังสือ
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    y = H * 0.44
    xs = [W * 0.17, W * 0.5, W * 0.83]
    k = [ease((tl - 0.1 - 0.25 * i) / 0.4) for i in range(3)]
    # เส้นเชื่อม + จุดวิ่ง
    lk = ease((tl - 0.6) / 0.6)
    for a, b in ((xs[0] + 80, xs[1] - 190), (xs[1] + 190, xs[2] - 90)):
        e = a + (b - a) * lk
        for x in np.arange(a, e, 18):
            d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 255, 255, 170))
        if lk >= 1:
            pos = a + (b - a) * ((tl * 0.9) % 1.0)
            d.ellipse((pos - 8, y - 8, pos + 8, y + 8), fill=(255, 226, 236, 255))
    # มือถือ / หูฟัง (การ์ดวงกลมใสๆ รองหลัง)
    for i, fn in ((0, icon_phone), (2, icon_headphones)):
        if k[i] <= 0:
            continue
        s = 150 * (0.8 + 0.2 * k[i])
        r = 112
        d.ellipse((xs[i] - r, y - r, xs[i] + r, y + r), fill=(255, 255, 255, int(46 * k[i])), outline=(255, 255, 255, int(120 * k[i])), width=2)
        fn(d, xs[i], y, s, (255, 255, 255, int(255 * k[i])))
    if k[2] > 0 and lk >= 1:                              # โน้ตดนตรีลอยออกจากหูฟัง
        for j in range(3):
            q = ((tl * 0.6 + j / 3) % 1.0)
            glyph(d, xs[2] + 70 + 40 * q, y - 70 - 90 * q, 34, "♫", (255, 255, 255, int(220 * (1 - q))))
    im.alpha_composite(lay)
    if k[1] > 0:                                          # ตัวแปลงตรงกลาง (ภาพจริง)
        w = int(360 * (0.85 + 0.15 * k[1]))
        c = cut.resize((w, int(cut.height * w / cut.width)), Image.LANCZOS)
        c.putalpha(c.getchannel("A").point(lambda v: int(v * k[1])))
        x0, y0 = int(xs[1] - c.width / 2), int(y - c.height / 2 + 10 * math.sin(t * 1.4))
        im.alpha_composite(c, (x0, y0))
    for i, label in enumerate(("PHONE", "TYPE-C TO 3.5MM", "HEADPHONES")):
        if k[i] > 0:
            ld = ImageDraw.Draw(im, "RGBA")
            globals()["label"](ld, xs[i], y + 150, label, fonts["label"], k[i])
    return im


def uses(SC, W, H, t, tl, p, fonts):
    """3 การใช้งาน: เด้งขึ้นทีละวง"""
    im = scene(SC, W, H, t)
    im.alpha_composite(grass(SC, W, H, t))
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    y = H * 0.42
    items = [("MUSIC", "music"), ("CALLS", "call"), ("GAMES", "game")]
    for i, (label, kind) in enumerate(items):
        k = ease((tl - 0.08 - 0.22 * i) / 0.35)
        if k <= 0:
            continue
        x = W * (0.25 + 0.25 * i)
        r = 104 * (0.7 + 0.3 * k) + 4 * math.sin(t * 2 + i)
        yy = y + 30 * (1 - k)
        d.ellipse((x - r, yy - r, x + r, yy + r), fill=(255, 255, 255, int(235 * k)))
        col = INK + (int(255 * k),)
        if kind == "music":
            glyph(d, x, yy + 4, r * 1.0, "♫", col)
        elif kind == "call":
            glyph(d, x, yy + 2, r * 0.95, "☎", col)
        else:
            icon_gamepad(d, x, yy + 4, r * 1.05, col, (255, 255, 255, int(235 * k)))
        globals()["label"](d, x, yy + r + 42, label, fonts["label"], k)
    im.alpha_composite(lay)
    return im
