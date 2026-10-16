"""ตัดอะแดปเตอร์ Type-C ออกจากรูป (ตัวขาวบนพื้นขาว: โมเดลตัดพื้นหลังและ GrabCut แยกไม่ได้)
วิธี: วาดเส้นขอบจากพิกัดที่วัดบนรูป ที่ความละเอียด x4 (ขอบเนียน) + ภาพขยาย x4 ด้วย EDSR"""
import os, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
SRC, OUT, K = "specs/typec_adapter.jpg", "assets/typec", 4
os.makedirs(OUT, exist_ok=True)
im = Image.open(SRC).convert("RGB")
W0, H0 = im.size
S = lambda pts: [(x * K, y * K) for x, y in pts]
mask = Image.new("L", (W0 * K, H0 * K), 0)
d = ImageDraw.Draw(mask)


def spline(pts, n=24):
    """Catmull-Rom: เส้นโค้งเรียบผ่านทุกจุด (แทนเส้นหักมุม)"""
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = (np.array(P[k], float) for k in (i - 1, i, i + 1, i + 2))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(np.array(pts[-1], float))
    return [tuple(p) for p in out]


def ellipse_poly(cx, cy, a, b, rot_deg, n=120):
    r = np.radians(rot_deg)
    t = np.linspace(0, 2 * np.pi, n)
    x, y = a * np.cos(t), b * np.sin(t)
    return [(cx + x[i] * np.cos(r) - y[i] * np.sin(r), cy + x[i] * np.sin(r) + y[i] * np.cos(r)) for i in range(n)]


d.polygon(S([(100, 413), (350, 299), (372, 348), (116, 497)]), fill=255)                       # กระบอกแจ็ค
d.polygon(S(ellipse_poly(100, 455, 20, 45, -13)), fill=255)                                    # หน้าแจ็ค (วงรีเอียง)
d.polygon(S([(348, 299), (398, 297), (432, 296), (441, 303), (426, 318), (372, 348)]), fill=255)  # ปลายต่อสายฝั่งแจ็ค
cable = spline([(434, 301), (470, 286), (510, 279), (560, 281), (607, 294), (643, 318), (663, 348), (669, 383),
                (662, 423), (642, 460), (619, 490), (604, 508)])
pts = np.array(cable)
seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
dist = np.concatenate([[0], np.cumsum(seg)])
for u in np.arange(0, dist[-1], 0.25):                                # ปั๊มวงกลมถี่ๆ ตามเส้น: ขอบเรียบ ไม่หยัก
    x, y = np.interp(u, dist, pts[:, 0]), np.interp(u, dist, pts[:, 1])
    d.ellipse(((x - 8) * K, (y - 8) * K, (x + 8) * K, (y + 8) * K), fill=255)
d.polygon(S([(349, 557), (528, 456), (548, 462), (575, 488), (604, 512), (610, 530), (607, 561), (432, 651)]), fill=255)  # ตัว USB-C
d.polygon(S([(340, 566), (349, 557), (358, 561), (354, 580), (346, 574)]), fill=255)           # ขอบหน้าตัว USB-C
d.polygon(S([(349, 557), (402, 590), (432, 630), (352, 578)]), fill=255)                        # หน้าตัวเครื่องระหว่างหัวโลหะ
d.polygon(S([(276, 610), (282, 602), (346, 573), (352, 578), (430, 628), (428, 660), (352, 692), (340, 694), (290, 660), (276, 630)]), fill=255)  # หัวโลหะ
if os.environ.get("OVERLAY"):
    ov = im.resize((W0 * 2, H0 * 2)).convert("RGBA")
    edge = mask.resize((W0 * 2, H0 * 2)).filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 40 else 0)
    red = Image.new("RGBA", ov.size, (255, 0, 0, 255)); ov.paste(red, (0, 0), edge)
    ov.crop(tuple(int(v) for v in os.environ.get('OVBOX', '120,500,1400,1420').split(','))).convert("RGB").save(os.environ["OVERLAY"])
mask = mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(2.5))   # เว้นขอบนิด + ขอบนุ่ม
if os.path.exists(f"{OUT}/photo_x4.png"):
    big = Image.open(f"{OUT}/photo_x4.png").convert("RGB")
else:
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("/opt/vstudio/models/sr/EDSR_x4.pb"); sr.setModel("edsr", 4)
    big = Image.fromarray(cv2.cvtColor(sr.upsample(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB))
    big.save(f"{OUT}/photo_x4.png")
cut4 = big.convert("RGBA")
cut4.putalpha(mask)
bb = cut4.getchannel("A").getbbox()
cut4 = cut4.crop(bb)
cut4.save(f"{OUT}/cutout_x4.png")
cut4.resize((cut4.width // K, cut4.height // K), Image.LANCZOS).save(f"{OUT}/cutout.png")
print("bbox x4", bb)
