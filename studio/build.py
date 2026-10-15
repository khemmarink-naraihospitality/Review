#!/usr/bin/env python3
"""สร้างคลิปรีวิว 10 วินาที จากไฟล์สเปก JSON  (ภาพซิงก์กับเสียงพากย์ไทยทีละประโยค)

  python3 build.py specs/typec.json            # สร้าง + ตรวจ QA อัตโนมัติ
  python3 build.py specs/typec.json --no-qa

สเปก:
{
  "name": "ชื่อสินค้า", "image": "รูป.jpg", "price": "", "cta_label": "กดสั่งซื้อที่ตะกร้าเลย!",
  "segments": [ {"role":"intro","text":"..."}, {"role":"point","text":"..."}, ..., {"role":"cta","text":"..."} ],
  "out": "out.mp4"
}
text = ทั้งแสดงเป็นซับไตเติลและพากย์ (ตัวเลข/อังกฤษถูกแปลงเป็นคำอ่านไทยอัตโนมัติ; ใส่ "say" เพื่อกำหนดคำอ่านเอง)
"""
import argparse, json, math, multiprocessing as mp, os, shutil, subprocess, sys, tempfile
import numpy as np, soundfile as sf, pyloudnorm as pyln
from PIL import Image, ImageDraw, ImageFont
from pythainlp.tokenize import word_tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tts import synth, SR as TTS_SR

W, H, FPS, DUR = 1080, 1920, 30, 10.0
FONT_B = "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"
BG1, BG2, ACCENT, WHITE, PANEL = (24, 28, 60), (96, 40, 120), (255, 205, 60), (255, 255, 255), (60, 60, 110)
LEAD, GAP, TAIL, TARGET_LUFS = 0.35, 0.30, 0.45, -16.0


class LayoutError(Exception):
    pass


def font(size):
    return ImageFont.truetype(FONT_B, size, layout_engine=ImageFont.Layout.RAQM)


def wrap_th(text, f, maxw):
    lines, cur = [], ""
    for w in word_tokenize(text, engine="newmm", keep_whitespace=True):
        if cur and f.getlength(cur + w) > maxw:
            lines.append(cur.strip())
            cur = w.lstrip()
        else:
            cur += w
    if cur.strip():
        lines.append(cur.strip())
    return lines


def fit(text, maxw, sizes, maxlines):
    """เลือกขนาดใหญ่สุดที่ไม่ล้น; ไม่พอ -> LayoutError (ไม่ตัด ไม่เดา)"""
    for s in sizes:
        f = font(s)
        lines = wrap_th(text, f, maxw)
        if len(lines) <= maxlines and all(f.getlength(l) <= maxw for l in lines):
            return f, lines
    raise LayoutError(f"ข้อความยาวเกินพื้นที่ ({maxlines} บรรทัด): {text!r} — ย่อข้อความ")


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


# ---------------------------------------------------------------- audio
def build_audio(segs, tmp, dur=None, lead=None, tail=None, max_extra=0.25):
    """สังเคราะห์ทีละประโยค -> ไทม์ไลน์ + เวลาเริ่ม/จบของแต่ละประโยค"""
    dur = DUR if dur is None else dur
    lead = LEAD if lead is None else lead
    tail = TAIL if tail is None else tail
    for speed in (1.0, 1.05, 1.10):
        clips = [synth(s.get("say") or s["text"], speed=round(speed * s.get("speed", 1.0), 3)) for s in segs]
        durs = [len(c) / TTS_SR for c in clips]
        total = lead + sum(durs) + GAP * (len(segs) - 1)
        if total <= dur - tail:
            break
    else:
        raise LayoutError(f"บทพากย์ยาว {total:.1f}s เกิน {dur - tail:.1f}s แม้เร่งสูงสุด — ย่อบท")
    spare = (dur - tail) - total
    gap = GAP + min(max_extra, spare / max(1, len(segs) - 1))      # กระจายเวลาที่เหลือ
    t, marks = lead, []
    line = np.zeros(int(dur * TTS_SR), dtype=np.float32)
    for c, d in zip(clips, durs):
        a = int(t * TTS_SR)
        line[a:a + len(c)] += c
        marks.append((t, t + d))
        t += d + gap
    meter = pyln.Meter(TTS_SR)
    loud = meter.integrated_loudness(line)
    line = line * (10 ** ((TARGET_LUFS - loud) / 20))
    peak = np.abs(line).max()
    if peak > 0.84:                                             # ~ -1.5 dBFS
        line = line * (0.84 / peak)
    wav = os.path.join(tmp, "voice.wav")
    sf.write(wav, line, TTS_SR)
    m4a = os.path.join(tmp, "voice.m4a")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ar", "48000", "-ac", "2",
                    "-af", "alimiter=limit=0.85", "-c:a", "aac", "-b:a", "160k", m4a], check=True)
    return m4a, marks, speed


# ---------------------------------------------------------------- visuals
STATE = {}


def make_card(path):
    size = 640
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    im = im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size, size), 56, fill=255)
    out = Image.new("RGBA", (size, size))
    out.paste(im, mask=mask)
    return out


def gradient():
    g = Image.new("RGB", (1, H))
    for y in range(H):
        k = y / H
        g.putpixel((0, y), tuple(int(BG1[i] * (1 - k) + BG2[i] * k) for i in range(3)))
    return g.resize((W, H))


def prepare(spec, segs, marks):
    pts = [i for i, s in enumerate(segs) if s["role"] == "point"]
    ctas = [i for i, s in enumerate(segs) if s["role"] == "cta"]
    try:                                   # ชอบบรรทัดเดียวก่อน; ไม่พอค่อยตัดเป็น 2 บรรทัด
        f_title, title_lines = fit(spec["name"], W - 140, range(80, 55, -2), 1)
    except LayoutError:
        f_title, title_lines = fit(spec["name"], W - 140, range(76, 51, -4), 2)
    bullets = []
    for i in pts:
        f, lines = fit(segs[i]["text"], W - 90 * 2 - 140 - 40, range(58, 43, -2), 1)
        bullets.append((i, f, lines[0]))
    subs = []
    for i, s in enumerate(segs):
        f, lines = fit(s["text"], W - 160, range(54, 41, -2), 2)
        subs.append((f, lines))
    STATE.update(spec=spec, segs=segs, marks=marks, bullets=bullets, cta=ctas[0] if ctas else None,
                 f_title=f_title, title_lines=title_lines, subs=subs, bg=gradient(),
                 card=make_card(spec["image"]), f_tag=font(48), f_price=font(130), f_cta=font(62))
    f_cta = STATE["f_cta"]
    if f_cta.getlength(spec.get("cta_label", "")) > 760:
        raise LayoutError("cta_label ยาวเกินปุ่ม")


def star_free_frame(n):
    S, t = STATE, n / FPS
    spec, segs, marks = S["spec"], S["segs"], S["marks"]
    img = S["bg"].copy().convert("RGBA")
    d = ImageDraw.Draw(img)
    # แท็กและชื่อสินค้า
    a = ease(t / 0.5)
    d.text((W // 2 + 3, 113), "รีวิวด่วน 10 วิ", font=S["f_tag"], fill=(0, 0, 0), anchor="mm")
    d.text((W // 2, 110), "รีวิวด่วน 10 วิ", font=S["f_tag"], fill=tuple(int(c * a + 30 * (1 - a)) for c in ACCENT), anchor="mm")
    lh = S["f_title"].size * 1.25
    y0 = 215 if len(S["title_lines"]) == 1 else 200
    for k, ln in enumerate(S["title_lines"]):
        al = ease((t - 0.25) / 0.5)
        col = tuple(int(c * al + 30 * (1 - al)) for c in WHITE)
        yy = y0 + k * lh
        d.text((W // 2 + 3, yy + 3), ln, font=S["f_title"], fill=(0, 0, 0), anchor="mm")
        d.text((W // 2, yy), ln, font=S["f_title"], fill=col, anchor="mm")
    # การ์ดสินค้า (เด้งเข้า + หายใจเบาๆ)
    k = ease(t / 0.8)
    sc = 0.6 + 0.4 * k + (0.015 * math.sin(t * 2.2) if t > 1 else 0)
    s = int(640 * sc)
    p = S["card"].resize((s, s), Image.LANCZOS)
    card_top = 340 if len(S["title_lines"]) == 1 else 405
    img.alpha_composite(p, ((W - s) // 2, card_top + (640 - s) // 2))
    d = ImageDraw.Draw(img)
    # จุดเด่น: โผล่ตอนเริ่มพูดข้อนั้น, ข้อที่กำลังพูดเรืองขอบ
    for n_b, (i, f, text) in enumerate(S["bullets"]):
        t0 = marks[i][0]
        kk = ease((t - t0 + 0.05) / 0.4)
        if kk <= 0:
            continue
        y = 1160 + n_b * 125
        xo = int((1 - kk) * 320)
        active = marks[i][0] <= t <= marks[i][1] + 0.2
        d.rounded_rectangle((90 + xo, y - 48, W - 90 + xo, y + 48), 40, fill=(80, 74, 150) if active else PANEL,
                            outline=ACCENT if active else None, width=4)
        d.line([(135 + xo, y), (157 + xo, y + 22), (195 + xo, y - 24)], fill=ACCENT, width=10, joint="curve")
        d.text((225 + xo, y), text, font=f, fill=WHITE, anchor="lm")
    # CTA
    ci = S["cta"]
    if ci is not None and t >= marks[ci][0] - 0.05:
        kk = ease((t - marks[ci][0] + 0.05) / 0.4)
        if spec.get("price"):                  # มีราคา: ใช้แผงทึบแทนจุดเด่น
            d.rectangle((0, 1060, W, 1640), fill=(24, 28, 60))
            yy = 1250
            d.text((W // 2 + 3, yy + 3), spec["price"], font=S["f_price"], fill=(0, 0, 0), anchor="mm")
            d.text((W // 2, yy), spec["price"], font=S["f_price"], fill=ACCENT, anchor="mm")
        pulse = 1 + 0.035 * math.sin(t * 8) * kk
        bw, bh = int(820 * pulse * (0.6 + 0.4 * kk)), int(150 * pulse)
        by = 1470 if spec.get("price") else 1570
        d.rounded_rectangle(((W - bw) // 2, by - bh // 2, (W + bw) // 2, by + bh // 2), 75, fill=ACCENT)
        d.text((W // 2, by), spec.get("cta_label", ""), font=S["f_cta"], fill=(30, 30, 30), anchor="mm")
    # ซับไตเติล = ประโยคที่กำลังพูด
    for i, (t0, t1) in enumerate(marks):
        if t0 - 0.05 <= t <= t1 + 0.25:
            f, lines = S["subs"][i]
            lh = f.size * 1.3
            h = lh * len(lines) + 36
            top = 1745 - h / 2
            d.rounded_rectangle((60, top, W - 60, top + h), 28, fill=(0, 0, 0, 170))
            for k2, ln in enumerate(lines):
                d.text((W // 2, top + 18 + lh * (k2 + 0.5)), ln, font=f, fill=WHITE, anchor="mm")
            break
    d.rectangle((0, H - 12, int(W * t / DUR), H), fill=ACCENT)
    return img.convert("RGB")


def render_range(args):
    lo, hi, tmp = args
    for n in range(lo, hi):
        star_free_frame(n).save(f"{tmp}/f{n:04d}.png", compress_level=1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--no-qa", action="store_true")
    a = ap.parse_args()
    spec_path = os.path.abspath(a.spec)
    spec = json.load(open(spec_path))
    base = os.path.dirname(spec_path)
    spec["image"] = os.path.normpath(os.path.join(base, spec["image"]))
    out = os.path.normpath(os.path.join(base, spec.get("out", "out.mp4")))
    segs = spec["segments"]
    tmp = tempfile.mkdtemp()
    try:
        m4a, marks, speed = build_audio(segs, tmp)
        prepare(spec, segs, marks)
        n_total = int(FPS * DUR)
        chunks = [(i, min(n_total, i + 25), tmp) for i in range(0, n_total, 25)]
        with mp.Pool(4) as pool:
            pool.map(render_range, chunks)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png",
                        "-i", m4a, "-t", str(DUR), "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart", "-c:a", "copy", out], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report = {"out": out, "tts_speed": speed,
              "segments": [{"text": s["text"], "start": round(m[0], 2), "end": round(m[1], 2)} for s, m in zip(segs, marks)]}
    json.dump(report, open(out + ".timing.json", "w"), ensure_ascii=False, indent=1)
    print(f"สร้างแล้ว: {out}  (TTS speed x{speed})")
    for s in report["segments"]:
        print(f"  {s['start']:5.2f}-{s['end']:5.2f}  {s['text']}")
    if not a.no_qa:
        import qa
        sys.exit(0 if qa.run(out, spec_path) else 1)


if __name__ == "__main__":
    main()
