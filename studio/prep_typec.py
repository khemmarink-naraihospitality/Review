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
d.polygon(S([(100, 413), (355, 292), (414, 294), (417, 349), (375, 373), (116, 497)]), fill=255)
d.ellipse(S([(80, 412), (124, 498)]), fill=255)                      # หน้าแจ็คกลม
cable = [(405, 311), (440, 290), (500, 279), (560, 281), (607, 294), (643, 318), (663, 348), (669, 383), (662, 423), (642, 460), (619, 490), (600, 508)]
d.line(S(cable), fill=255, width=18 * K, joint="curve")
for p in (cable[0], cable[-1]):
    d.ellipse((p[0] * K - 9 * K, p[1] * K - 9 * K, p[0] * K + 9 * K, p[1] * K + 9 * K), fill=255)
d.polygon(S([(349, 557), (533, 453), (561, 456), (607, 519), (607, 561), (432, 651)]), fill=255)
d.polygon(S([(279, 608), (295, 595), (349, 574), (427, 640), (420, 655), (360, 689), (345, 685), (283, 630)]), fill=255)   # หัว USB-C
if os.environ.get("OVERLAY"):
    ov = im.resize((W0 * 2, H0 * 2)).convert("RGBA")
    edge = mask.resize((W0 * 2, H0 * 2)).filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 40 else 0)
    red = Image.new("RGBA", ov.size, (255, 0, 0, 255)); ov.paste(red, (0, 0), edge)
    ov.crop((120, 500, 1400, 1420)).convert("RGB").save(os.environ["OVERLAY"])
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
