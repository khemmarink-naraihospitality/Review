#!/usr/bin/env python3
"""คลิปรีวิวสไตล์สตูดิโอ (แนวนอน 1280x720): ฉากหมุน/ซูมเข้าแถบแดง/ไล่ดูลายเชือก/ตัดจบชื่อแบรนด์
ใช้รูปสินค้ารูปเดียว (ตัดพื้นหลัง + ขยาย x4 ด้วย prep_assets.py) ไม่ได้สร้างฟุตเทจ 3D จริง

  python3 prep_assets.py specs/sneaker.jpg assets/sneaker
  python3 cinematic.py specs/sneaker_cine.json
"""
import json, math, multiprocessing as mp, os, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build as B
from build import build_audio, wrap_th, LayoutError

W, H, FPS, DUR = 1280, 720, 30, 10.0
FONT_B = B.FONT_B
ST = {}


def font(sz):
    return ImageFont.truetype(FONT_B, sz, layout_engine=ImageFont.Layout.RAQM)


def ease(p):
    p = max(0.0, min(1.0, p))
    return p * p * (3 - 2 * p)


def studio_bg():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W * 0.5) / (W * 0.62)) ** 2 + ((y - H * 0.46) / (H * 0.75)) ** 2)
    k = np.clip(1 - r, 0, 1) ** 0.8
    base = np.array([205, 209, 217], np.float32) + (np.array([249, 250, 252], np.float32) - np.array([205, 209, 217], np.float32)) * k[..., None]
    return base


def crop_cut(cut):
    bb = cut.getchannel("A").getbbox()
    return cut.crop(bb)


def split_left(cut):
    """แยกรองเท้าข้างซ้ายจากคู่: หาคอลัมน์ว่างที่สุดช่วงกลางภาพ"""
    a = np.asarray(cut.getchannel("A")).astype(np.float32).sum(0)
    w = len(a)
    mid = int(w * 0.3) + int(np.argmin(a[int(w * 0.3):int(w * 0.7)]))
    left = cut.crop((0, 0, mid, cut.height))
    return left.crop(left.getchannel("A").getbbox())


def shoes_layer(sc, angle, single=False):
    im = ST["single"] if single else ST["cut"]
    w, h = im.size
    im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS).rotate(angle, resample=Image.BICUBIC, expand=True)
    return im


def compose_studio(sc, angle, cx, cy, sweep, single=False):
    """พื้นหลังสตูดิโอ + เงา + รองเท้า + แสงกวาด -> RGB uint8"""
    bg = Image.fromarray(ST["bg"].astype(np.uint8)).convert("RGBA")
    sh = shoes_layer(sc, angle, single)
    x, y = int(cx - sh.width / 2), int(cy - sh.height / 2)
    a = sh.getchannel("A").filter(ImageFilter.GaussianBlur(22))
    shadow = Image.new("RGBA", sh.size, (30, 34, 48, 0))
    shadow.putalpha(a.point(lambda v: int(v * 0.42)))
    bg.alpha_composite(shadow, (x + 26, y + 38)) if (x + 26 < W and y + 38 < H) else None
    bg.alpha_composite(sh, (x, y)) if (x < W and y < H and x + sh.width > 0 and y + sh.height > 0) else None
    return finish(np.asarray(bg.convert("RGB")).astype(np.float32), sweep)


def finish(arr, sweep):
    """วิกเน็ต + แสงกวาดเฉียง (ให้ความรู้สึกไฟสตูดิโอ)"""
    arr = arr * ST["vig"]
    if sweep is not None:
        arr = arr + ST["band"](sweep)
    return np.clip(arr, 0, 255).astype(np.uint8)


def detail_crop(box_c, win_w, rot180=False):
    """ตัดมุมกล้องจากภาพ x4 (พิกัดอิงรูปต้นฉบับ 1000px)"""
    big = ST["big"]
    cx, cy = box_c
    ww = win_w
    wh = ww * H / W
    x0, y0, x1, y1 = (cx - ww / 2) * 4, (cy - wh / 2) * 4, (cx + ww / 2) * 4, (cy + wh / 2) * 4
    im = big.crop((int(x0), int(y0), int(x1), int(y1))).resize((W, H), Image.LANCZOS)
    return im.rotate(180) if rot180 else im


def shot_frame(shot, p, t):
    kind = shot["kind"]
    sweep = (p * 1.4 - 0.2) if shot.get("sweep", True) else None
    if kind == "hero":
        return compose_studio(0.58 + 0.06 * ease(p), -34 + 12 * ease(p), W * 0.5, H * 0.50, sweep)
    if kind == "pair":     # เลื่อนกล้องจากรองเท้าข้างหนึ่งไปอีกข้าง
        return compose_studio(0.86 + 0.05 * ease(p), -90, W * (0.47 + 0.06 * ease(p)), H * 0.5, sweep)
    if kind == "single":   # รองเท้าข้างเดียว ความละเอียดจริงของรูป (คมกว่าภาพขยาย)
        return compose_studio(0.70 + 0.06 * ease(p), 58 - 8 * ease(p), W * (0.52 - 0.04 * ease(p)), H * 0.5, sweep, single=True)
    if kind == "detail":
        c0, c1 = shot["from"], shot["to"]
        c = (c0[0] + (c1[0] - c0[0]) * ease(p), c0[1] + (c1[1] - c0[1]) * ease(p))
        win = shot["win"][0] + (shot["win"][1] - shot["win"][0]) * ease(p)
        im = detail_crop(c, win, shot.get("rot180", False)).filter(ImageFilter.UnsharpMask(radius=2, percent=55, threshold=2))
        return finish(np.asarray(im).astype(np.float32), sweep)
    if kind == "end":
        k = ease(p * 2.5)
        arr = compose_studio(0.47 + 0.05 * ease(p), -31 + 3 * ease(p), W * 0.27, H * 0.50, p * 1.2 - 0.1).astype(np.float32)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        f = ST["f_brand"]
        col = tuple(int(c * k + 225 * (1 - k)) for c in (28, 32, 48))
        d.text((W * 0.71, H * 0.50), shot["text"], font=f, fill=col, anchor="mm")
        return np.asarray(img)
    raise ValueError(kind)


def subtitle(img, text, a):
    f, lines = ST["subs"][text]
    d = ImageDraw.Draw(img, "RGBA")
    lh = f.size * 1.3
    h = lh * len(lines) + 24
    top = H - 70 - h / 2
    d.rounded_rectangle((W / 2 - ST["sub_w"] / 2 - 24, top, W / 2 + ST["sub_w"] / 2 + 24, top + h), 22, fill=(0, 0, 0, int(150 * a)))
    for k, ln in enumerate(lines):
        d.text((W / 2, top + 12 + lh * (k + 0.5)), ln, font=f, fill=(255, 255, 255, int(255 * a)), anchor="mm")


def frame(n):
    t = n / FPS
    bounds, shots, marks, segs = ST["bounds"], ST["shots"], ST["marks"], ST["segs"]
    i = next(k for k in range(len(shots)) if t < bounds[k + 1] or k == len(shots) - 1)

    def render(k):
        p = (t - bounds[k]) / (bounds[k + 1] - bounds[k])
        return shot_frame(shots[k], max(0.0, min(1.0, p)), t)

    cur = render(i)
    FADE = 0.16
    if i + 1 < len(shots) and t > bounds[i + 1] - FADE:
        a = ease((t - (bounds[i + 1] - FADE)) / FADE)
        nxt = render(i + 1)
        cur = (cur.astype(np.float32) * (1 - a) + nxt.astype(np.float32) * a).astype(np.uint8)
    img = Image.fromarray(cur)
    if ST["subtitles"]:
        for k, (t0, t1) in enumerate(marks):
            if t0 - 0.05 <= t <= t1 + 0.25:
                a = ease((t - t0 + 0.05) / 0.2) * (1 - ease((t - t1 - 0.05) / 0.2))
                subtitle(img, segs[k]["text"], max(0.0, a))
                break
    return img


def render_range(args):
    lo, hi, tmp = args
    for n in range(lo, hi):
        frame(n).save(f"{tmp}/f{n:04d}.png", compress_level=1)


def denoise_details(shots):
    """ลบเส้น/บล็อก JPEG เฉพาะบริเวณที่ช็อตซูมใกล้ใช้ (ทำครั้งเดียวก่อนเรนเดอร์)"""
    import cv2
    big = np.asarray(ST["big"]).copy()
    for sh in shots:
        if sh["kind"] != "detail":
            continue
        ww = max(sh["win"]) * 1.15
        wh = ww * H / W
        xs = [sh["from"][0], sh["to"][0]]
        ys = [sh["from"][1], sh["to"][1]]
        x0, x1 = int(max(0, (min(xs) - ww / 2) * 4)), int(min(big.shape[1], (max(xs) + ww / 2) * 4))
        y0, y1 = int(max(0, (min(ys) - wh / 2) * 4)), int(min(big.shape[0], (max(ys) + wh / 2) * 4))
        region = cv2.cvtColor(big[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
        region = cv2.fastNlMeansDenoisingColored(region, None, 6, 6, 7, 21)
        big[y0:y1, x0:x1] = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    ST["big"] = Image.fromarray(big)


def mix_music(voice_m4a, tmp, bpm, music_lufs):
    """ดนตรีพื้นหลังเบาๆ + ducking อัตโนมัติตอนมีเสียงพากย์ แล้วจำกัดพีค"""
    import soundfile as sf, pyloudnorm as pyln, music
    x = music.make(DUR, bpm)
    x = x * (10 ** ((music_lufs - pyln.Meter(music.SR).integrated_loudness(x)) / 20))
    bgm = os.path.join(tmp, "bgm.wav")
    sf.write(bgm, x, music.SR)
    out = os.path.join(tmp, "mix.m4a")
    graph = ("[0:a]asplit=2[v][sc];"
             "[1:a][sc]sidechaincompress=threshold=0.06:ratio=3.5:attack=40:release=500:makeup=1[m];"
             "[v][m]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.85:level=false[o]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", voice_m4a, "-i", bgm, "-filter_complex", graph,
                    "-map", "[o]", "-t", str(DUR), "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", out], check=True)
    return out


def main():
    spec_path = os.path.abspath(sys.argv[1])
    spec = json.load(open(spec_path))
    base = os.path.dirname(spec_path)
    assets = os.path.normpath(os.path.join(base, spec["assets"]))
    out = os.path.normpath(os.path.join(base, spec.get("out", "out.mp4")))
    segs, shots = spec["segments"], spec["shots"]
    assert len(segs) == len(shots), "จำนวน shots ต้องเท่ากับ segments"
    tmp = tempfile.mkdtemp()
    try:
        m4a, marks, speed = build_audio(segs, tmp)
        if spec.get("music", True):
            m4a = mix_music(m4a, tmp, spec.get("music_bpm", 92), spec.get("music_lufs", -21.0))
        # ขอบเขตแต่ละฉาก = กึ่งกลางช่องว่างระหว่างประโยค
        bounds = [0.0] + [(marks[i][1] + marks[i + 1][0]) / 2 for i in range(len(marks) - 1)] + [DUR]
        cut = crop_cut(Image.open(f"{assets}/cutout.png").convert("RGBA"))
        ST.update(bg=studio_bg(), cut=cut, single=split_left(cut),
                  big=Image.open(f"{assets}/photo_x4.png").convert("RGB"), shots=shots, segs=segs, marks=marks,
                  bounds=bounds, subtitles=spec.get("subtitles", True), f_brand=font(spec.get("brand_size", 84)))
        denoise_details(shots)
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.85)) ** 2)
        ST["vig"] = (1 - 0.30 * np.clip(r - 0.35, 0, 1) ** 1.5)[..., None]
        ang = math.radians(28)
        proj = (x * math.cos(ang) + y * math.sin(ang))
        lo_, hi_ = proj.min(), proj.max()

        def band(s):
            pos = lo_ + (hi_ - lo_) * s
            return (np.exp(-((proj - pos) / 120.0) ** 2) * 26.0)[..., None]
        ST["band"] = band
        subs = {}
        ST["sub_w"] = W - 360
        for s in segs:
            subs[s["text"]] = B.fit(s["text"], ST["sub_w"], range(40, 31, -2), 2)
        ST["subs"] = subs
        if ST["f_brand"].getlength(spec.get("brand", "")) > W * 0.62:
            raise LayoutError("ชื่อแบรนด์ยาวเกินพื้นที่")
        for sh in shots:
            if sh["kind"] == "end":
                if ST["f_brand"].getlength(sh["text"]) > W * 0.62:
                    raise LayoutError("ข้อความฉากจบยาวเกินพื้นที่")
        n_total = int(FPS * DUR)
        chunks = [(i, min(n_total, i + 25), tmp) for i in range(0, n_total, 25)]
        with mp.Pool(4) as pool:
            pool.map(render_range, chunks)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png", "-i", m4a,
                        "-t", str(DUR), "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                        "-r", str(FPS), "-movflags", "+faststart", "-c:a", "copy", out], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report = {"out": out, "tts_speed": speed, "size": [W, H], "bounds": [round(b, 2) for b in bounds],
              "segments": [{"text": s["text"], "speed": round(speed * s.get("speed", 1.0), 3), "start": round(m[0], 2), "end": round(m[1], 2)} for s, m in zip(segs, marks)]}
    json.dump(report, open(out + ".timing.json", "w"), ensure_ascii=False, indent=1)
    print("สร้างแล้ว:", out)
    for s, b0, b1 in zip(report["segments"], bounds, bounds[1:]):
        print(f"  ฉาก {b0:5.2f}-{b1:5.2f}  พูด {s['start']:5.2f}-{s['end']:5.2f}  {s['text']}")
    import qa
    sys.exit(0 if qa.run(out, spec_path) else 1)


if __name__ == "__main__":
    main()
