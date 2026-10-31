#!/usr/bin/env python3
"""คลิปรีวิวแนวตั้ง 1080x1920 แบบยาว (หลายฉาก หลายประโยค) จากสเปก JSON

แต่ละฉากมีกราฟิกของตัวเอง องค์ประกอบโผล่ตามจังหวะประโยคที่พูด + ซับไตเติล + การ์ดคำเตือนท้ายคลิป
  python3 explainer.py specs/acuvue.json
"""
import json, math, multiprocessing as mp, os, subprocess, sys, tempfile
import numpy as np, soundfile as sf, pyloudnorm as pyln
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tts import synth, SR as TTS_SR
import build as B

W, H, FPS = 1080, 1920, 30
TH = "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"
LAT = os.path.join(HERE, "fonts", "montserrat-latin-800-normal.woff")
GLY = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NAVY, TEAL, SKY, AMBER, WHITE, GREY = (16, 42, 74), (0, 140, 160), (226, 244, 249), (255, 196, 64), (255, 255, 255), (110, 124, 140)
ST = {}


def fth(sz):
    return ImageFont.truetype(TH, sz, layout_engine=ImageFont.Layout.RAQM)


def flat(sz):
    return ImageFont.truetype(LAT, sz)


def ease(p):
    p = max(0.0, min(1.0, p))
    return 1 - (1 - p) ** 3


def fade_col(c, a):
    return tuple(c) + (int(255 * max(0, min(1, a))),)


# ---------------------------------------------------------------- audio
def build_voice(segs, lead, gap_in, gap_scene, tail, voice_fx, tmp):
    clips = [synth(s.get("say") or s["text"], speed=s.get("speed", 1.0)) for s in segs]
    t, marks = lead, []
    for i, (s, c) in enumerate(zip(segs, clips)):
        if i:
            t += gap_scene if s["scene"] != segs[i - 1]["scene"] else gap_in
        marks.append((t, t + len(c) / TTS_SR))
        t += len(c) / TTS_SR
    dur = round(t + tail, 2)
    line = np.zeros(int(dur * TTS_SR) + TTS_SR, np.float32)
    for (a, _), c in zip(marks, clips):
        i = int(a * TTS_SR)
        line[i:i + len(c)] += c
    line *= 10 ** ((-16 - pyln.Meter(TTS_SR).integrated_loudness(line)) / 20)
    wav = os.path.join(tmp, "voice.wav")
    sf.write(wav, line, TTS_SR)
    out = os.path.join(tmp, "voice.m4a")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ar", "48000", "-ac", "2", "-af",
                    (B.VOICE_FX if voice_fx else "") + "alimiter=limit=0.85", "-c:a", "aac", "-b:a", "160k", out], check=True)
    return out, marks, dur


def mix(voice, tmp, dur, lufs, fade):
    import music
    x = music.make(dur, 92, fade=fade)
    x = x * (10 ** ((lufs - pyln.Meter(music.SR).integrated_loudness(x)) / 20))
    bgm = os.path.join(tmp, "bgm.wav")
    sf.write(bgm, x, music.SR)
    raw = os.path.join(tmp, "mix.wav")
    g = (f"[0:a]apad=whole_dur={dur},asplit=2[v][sc];[1:a][sc]sidechaincompress=threshold=0.05:ratio=4:attack=25:release=450:makeup=1[m];"
         "[v][m]amix=inputs=2:normalize=0:duration=longest[o]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", voice, "-i", bgm, "-filter_complex", g, "-map", "[o]",
                    "-t", str(dur), "-ar", "48000", "-ac", "2", raw], check=True)
    y, sr = sf.read(raw, dtype="float32")
    gain = -16.0 - pyln.Meter(sr).integrated_loudness(y)
    out = os.path.join(tmp, "mix.m4a")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af", f"volume={gain:.2f}dB,alimiter=limit=0.84:level=false",
                    "-c:a", "aac", "-b:a", "192k", out], check=True)
    return out


# ---------------------------------------------------------------- drawing helpers
def background(t):
    im = Image.fromarray(ST["bg"]).convert("RGBA")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for x, y, r, sp, ph in ST["blobs"]:                      # วงกลมนุ่มลอยช้าๆ
        yy = y + 40 * math.sin(t * sp + ph)
        d.ellipse((x - r, yy - r, x + r, yy + r), fill=(0, 160, 180, 26))
    im.alpha_composite(lay.filter(ImageFilter.GaussianBlur(40)))
    return im


def pill(d, cx, cy, text, font, fg, bg, a=1.0, pad=(34, 18), anchor_left=False):
    w = font.getlength(text)
    h = font.size
    x0 = cx if anchor_left else cx - w / 2 - pad[0]
    d.rounded_rectangle((x0, cy - h / 2 - pad[1], x0 + w + 2 * pad[0], cy + h / 2 + pad[1]), int(h / 2 + pad[1]), fill=fade_col(bg, a))
    d.text((x0 + pad[0], cy), text, font=font, fill=fade_col(fg, a), anchor="lm")


def check(d, x, y, s, col):
    d.line([(x, y), (x + s * 0.35, y + s * 0.35), (x + s, y - s * 0.45)], fill=col, width=max(4, int(s * 0.16)), joint="curve")


def centered(d, y, text, font, col, a=1.0, dy=0.0):
    d.text((W / 2, y + (1 - a) * dy), text, font=font, fill=fade_col(col, a), anchor="mm")


def reveal(tl, t0, dur=0.45):
    return ease((tl - t0) / dur)


# ---------------------------------------------------------------- scenes  (rv = เวลาเริ่มของแต่ละประโยคในฉาก, เทียบกับต้นฉาก)
def scene_hook(d, im, tl, rv, t):
    a = reveal(tl, 0.0, 0.6)
    cx, cy, s = W / 2, 640, 300 * (0.85 + 0.15 * a)
    blink = 1 - 0.85 * max(0, math.sin((tl % 3.2) / 3.2 * math.pi * 10)) if 1.2 < tl % 3.2 < 1.5 else 1
    eye = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    e = ImageDraw.Draw(eye)
    hh = s * 0.55 * blink
    e.ellipse((cx - s, cy - hh, cx + s, cy + hh), fill=fade_col(WHITE, a), outline=fade_col(NAVY, a), width=10)
    if hh > 30:
        r = min(s * 0.42, hh * 0.95)
        e.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fade_col(TEAL, a))
        e.ellipse((cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45), fill=fade_col(NAVY, a))
        e.ellipse((cx + r * 0.2, cy - r * 0.55, cx + r * 0.45, cy - r * 0.3), fill=fade_col(WHITE, a))
    dy = (tl * 120) % 260                                    # หยดน้ำตาไหล (ตาแห้ง/ล้า)
    e.ellipse((cx + s * 0.55 - 22, cy + hh + 30 + dy - 22, cx + s * 0.55 + 22, cy + hh + 30 + dy + 22), fill=fade_col((120, 200, 230), a * (1 - dy / 260)))
    im.alpha_composite(eye)
    d = ImageDraw.Draw(im)
    b = reveal(tl, 0.35)
    centered(d, 1040, "ตาล้า ตาแห้ง", fth(118), NAVY, b, 60)
    centered(d, 1180, "ตอนบ่าย?", fth(118), TEAL, reveal(tl, 0.6), 60)
    if len(rv) > 1:
        pill(d, W / 2, 1360, "วันนี้มาเล่าตัวที่ใช้อยู่", fth(54), WHITE, NAVY, reveal(tl, rv[1]))


def scene_intro(d, im, tl, rv, t):
    a = reveal(tl, 0.0, 0.6)
    centered(d, 470, "ACUVUE", flat(150), NAVY, a, 50)
    centered(d, 620, "OASYS 1-Day", flat(96), TEAL, reveal(tl, 0.2, 0.6), 50)
    w = 520 * reveal(tl, 0.4, 0.7)
    d.rounded_rectangle((W / 2 - w / 2, 700, W / 2 + w / 2, 712), 6, fill=fade_col(TEAL, a))
    items = [(rv[0] + 0.6, "แบบรายวัน"), (rv[1] if len(rv) > 1 else 99, "ใส่เช้า ถอดทิ้งตอนเย็น"),
             (rv[2] if len(rv) > 2 else 99, "ไม่ต้องล้าง"), ((rv[2] + 1.2) if len(rv) > 2 else 99, "ไม่ต้องแช่น้ำยา")]
    for k, (t0, txt) in enumerate(items):
        q = reveal(tl, t0)
        if q <= 0:
            continue
        y = 880 + k * 150
        x = 170 + (1 - q) * 120
        d.rounded_rectangle((x, y - 56, W - 170 + (1 - q) * 120, y + 56), 56, fill=fade_col(WHITE, q), outline=fade_col(TEAL, q * 0.5), width=3)
        d.ellipse((x + 22, y - 34, x + 90, y + 34), fill=fade_col(TEAL, q))
        check(d, x + 38, y + 2, 34, fade_col(WHITE, q))
        d.text((x + 120, y), txt, font=fth(56), fill=fade_col(NAVY, q), anchor="lm")


def scene_box(d, im, tl, rv, t):
    a = reveal(tl, 0.0, 0.5)
    n = int(round(30 * ease(min(1, tl / 1.2))))
    centered(d, 520, str(max(n, 0)), flat(300), TEAL, a, 40)
    centered(d, 720, "ชิ้น / กล่อง", fth(66), NAVY, reveal(tl, 0.3))
    pill(d, W / 2, 860, "ใส่ 2 ตา  =  15 วัน", fth(58), WHITE, NAVY, reveal(tl, 1.4))
    t1 = rv[1] if len(rv) > 1 else 2.5
    for i in range(30):                                       # แผงเลนส์แยกชิ้น 30 ช่อง
        q = reveal(tl, t1 - 0.6 + i * 0.05, 0.3)
        if q <= 0:
            continue
        r, c = divmod(i, 6)
        x, y = 150 + c * 135, 1010 + r * 95
        d.rounded_rectangle((x, y + (1 - q) * 30, x + 112, y + 78 + (1 - q) * 30), 18, fill=fade_col(WHITE, q), outline=fade_col(TEAL, q * 0.7), width=3)
        d.ellipse((x + 36, y + 19 + (1 - q) * 30, x + 76, y + 59 + (1 - q) * 30), outline=fade_col((120, 200, 230), q), width=4)


def scene_spec(d, im, tl, rv, t):
    centered(d, 430, "สเปก", fth(100), NAVY, reveal(tl, 0.0), 40)
    rows = [(rv[0] + 0.3, "ค่าสายตา", "0.00 – 9.00"), ((rv[1] if len(rv) > 1 else 99), "ความโค้ง (BC)", "8.5 / 9.0"),
            ((rv[1] + 1.8) if len(rv) > 1 else 99, "กรองแสง UV", None)]
    for k, (t0, lab, val) in enumerate(rows):
        q = reveal(tl, t0)
        if q <= 0:
            continue
        y = 600 + k * 170
        d.rounded_rectangle((110, y - 70, W - 110, y + 70), 36, fill=fade_col(WHITE, q), outline=fade_col(TEAL, q * 0.4), width=3)
        d.text((160 + (1 - q) * 60, y), lab, font=fth(56), fill=fade_col(NAVY, q), anchor="lm")
        if val:
            d.text((W - 160, y), val, font=flat(60), fill=fade_col(TEAL, q), anchor="rm")
        else:
            d.ellipse((W - 250, y - 44, W - 162, y + 44), fill=fade_col(TEAL, q))
            check(d, W - 228, y + 2, 46, fade_col(WHITE, q))
    if len(rv) > 2:                                           # คำเตือน: ให้ผู้เชี่ยวชาญวัดค่า
        q = reveal(tl, rv[2])
        if q > 0:
            y = 1190
            d.rounded_rectangle((110, y - 120, W - 110, y + 120), 40, fill=fade_col((255, 243, 214), q), outline=fade_col(AMBER, q), width=5)
            d.text((175, y), "!", font=ImageFont.truetype(GLY, 110), fill=fade_col((214, 140, 0), q), anchor="mm")
            d.text((240, y - 40), "ให้ร้านแว่นหรือหมอตา", font=fth(54), fill=fade_col(NAVY, q), anchor="lm")
            d.text((240, y + 40), "วัดค่าให้ก่อน อย่ากะเอง", font=fth(54), fill=fade_col(NAVY, q), anchor="lm")


def scene_close(d, im, tl, rv, t):
    centered(d, 470, "สั่งได้ที่", fth(64), GREY, reveal(tl, 0.0), 40)
    centered(d, 600, "Contacts Direct", flat(92), NAVY, reveal(tl, 0.2), 40)
    centered(d, 720, "by TA-TO.com", flat(66), TEAL, reveal(tl, 0.4), 40)
    if len(rv) > 1:
        q = reveal(tl, rv[1])
        pill(d, W / 2, 900, "ลิงก์อยู่ในคอมเมนต์", fth(60), WHITE, TEAL, q)
        if q > 0:
            y = 1000 + 14 * math.sin(t * 5)
            d.polygon([(W / 2 - 34, y), (W / 2 + 34, y), (W / 2, y + 40)], fill=fade_col(TEAL, q))
    if len(rv) > 2:
        q = reveal(tl, rv[2])
        if q > 0:
            y = 1230
            d.rounded_rectangle((150, y - 110, W - 150, y + 100), 50, fill=fade_col(WHITE, q), outline=fade_col(NAVY, q * 0.3), width=3)
            d.polygon([(260, y + 96), (330, y + 96), (250, y + 170)], fill=fade_col(WHITE, q))
            d.text((W / 2, y - 38), "คุณใส่ตัวไหนอยู่?", font=fth(60), fill=fade_col(NAVY, q), anchor="mm")
            d.text((W / 2, y + 40), "เล่าให้ฟังหน่อย", font=fth(54), fill=fade_col(TEAL, q), anchor="mm")


SCENES = {"hook": scene_hook, "intro": scene_intro, "box": scene_box, "spec": scene_spec, "close": scene_close}


def end_card(im, tl, spec):
    d = ImageDraw.Draw(im)
    a = reveal(tl, 0.0, 0.6)
    d.rounded_rectangle((90, 560, W - 90, 1240), 50, fill=fade_col(WHITE, a), outline=fade_col(TEAL, a * 0.5), width=4)
    d.text((W / 2, 680), "คำเตือน", font=fth(72), fill=fade_col((200, 60, 40), a), anchor="mm")
    y = 820
    for line in spec["end_warning"]:
        f, lines = B.fit(line, W - 300, range(56, 40, -2), 2)
        for ln in lines:
            d.text((W / 2, y), ln, font=f, fill=fade_col(NAVY, a), anchor="mm")
            y += f.size * 1.4
        y += 20
    d.text((W / 2, y + 40), spec["end_number"], font=flat(58), fill=fade_col(NAVY, a), anchor="mm")


# ---------------------------------------------------------------- frame
def frame(n):
    t = n / FPS
    spec, marks, segs = ST["spec"], ST["marks"], ST["segs"]
    im = background(t)
    sc_bounds = ST["scene_bounds"]
    d = ImageDraw.Draw(im)
    pill(d, W / 2, 150, "รีวิว ACUVUE OASYS 1-Day", fth(46), NAVY, WHITE)
    d.text((W / 2, 232), "เสียงบรรยายสังเคราะห์", font=fth(32), fill=GREY + (255,), anchor="mm")

    def draw_scene(k, alpha):
        name, t0, t1, idx = sc_bounds[k]
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        rv = [marks[i][0] - t0 for i in idx]
        SCENES[name](ld, lay, t - t0, rv, t)
        if alpha < 1:
            lay.putalpha(lay.getchannel("A").point(lambda v: int(v * alpha)))
        im.alpha_composite(lay)

    end_t = ST["end_start"]
    if t >= end_t:
        end_card(im, t - end_t, spec)
    else:
        for k, (name, t0, t1, idx) in enumerate(sc_bounds):
            if t0 - 0.0 <= t < t1:
                FADE = 0.3
                a_in = ease((t - t0) / FADE)
                a_out = 1 - ease((t - (t1 - FADE)) / FADE) if k < len(sc_bounds) - 1 or t1 < end_t + 1 else 1
                draw_scene(k, min(a_in, a_out))
        d = ImageDraw.Draw(im)
        for i, (a, b) in enumerate(marks):                    # ซับไตเติลประโยคที่กำลังพูด
            if a - 0.05 <= t <= b + 0.2:
                f, lines = ST["subs"][i]
                lh = f.size * 1.35
                h = lh * len(lines) + 44
                top = 1640 - h / 2
                d.rounded_rectangle((70, top, W - 70, top + h), 34, fill=(255, 255, 255, 235), outline=TEAL + (90,), width=3)
                for k2, ln in enumerate(lines):
                    d.text((W / 2, top + 22 + lh * (k2 + 0.5)), ln, font=f, fill=NAVY + (255,), anchor="mm")
                break
    fo = spec.get("fade_out", 0.6)
    if t > ST["dur"] - fo:
        q = ease((t - (ST["dur"] - fo)) / fo)
        arr = np.asarray(im.convert("RGB")).astype(np.float32)
        return (arr * (1 - q) + 255 * q).astype(np.uint8).tobytes()
    return np.asarray(im.convert("RGB")).tobytes()


def main():
    spec_path = os.path.abspath(sys.argv[1])
    spec = json.load(open(spec_path))
    base = os.path.dirname(spec_path)
    out = os.path.normpath(os.path.join(base, spec["out"]))
    segs = spec["segments"]
    tmp = tempfile.mkdtemp()
    voice, marks, speech_end = build_voice(segs, spec.get("lead", 0.6), spec.get("gap_in", 0.35), spec.get("gap_scene", 0.75),
                                           0.0, spec.get("voice_fx", True), tmp)
    end_start = speech_end + 0.6
    dur = round(end_start + spec.get("end_secs", 4.5), 2)
    audio = mix(voice, tmp, dur, spec.get("music_lufs", -21.0), spec.get("music_fade", 1.5))
    # ฉาก: เริ่มก่อนประโยคแรกของฉาก 0.3s, จบเมื่อฉากถัดไปเริ่ม
    order = []
    for i, s in enumerate(segs):
        if not order or order[-1][0] != s["scene"]:
            order.append([s["scene"], []])
        order[-1][1].append(i)
    sb = []
    for k, (name, idx) in enumerate(order):
        t0 = 0.0 if k == 0 else marks[idx[0]][0] - 0.35
        t1 = (marks[order[k + 1][1][0]][0] - 0.35) if k + 1 < len(order) else end_start
        sb.append((name, t0, t1, idx))
    rng = np.random.default_rng(5)
    g = np.linspace(0, 1, H)[:, None, None]
    bg = (np.array(SKY) * (1 - g) + np.array([250, 253, 255]) * g)
    bg = np.broadcast_to(bg, (H, W, 3)).astype(np.uint8)
    ST.update(spec=spec, segs=segs, marks=marks, scene_bounds=sb, end_start=end_start, dur=dur, bg=np.ascontiguousarray(bg),
              blobs=[(rng.uniform(0, W), rng.uniform(300, 1500), rng.uniform(120, 260), rng.uniform(0.2, 0.5), rng.uniform(0, 6.28)) for _ in range(7)],
              subs=[B.fit(s["text"], W - 220, range(52, 37, -2), 2) for s in segs])
    n_total = int(round(dur * FPS))
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-i", audio, "-t", str(dur), "-c:v", "libx264", "-preset", "medium",
                            "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "copy", out], stdin=subprocess.PIPE)
    with mp.Pool(4) as pool:
        for buf in pool.imap(frame, range(n_total), chunksize=8):
            enc.stdin.write(buf)
    enc.stdin.close()
    enc.wait()
    report = {"out": out, "size": [W, H], "duration": dur,
              "segments": [{"text": s["text"], "start": round(a, 2), "end": round(b, 2)} for s, (a, b) in zip(segs, marks)]}
    json.dump(report, open(out + ".timing.json", "w"), ensure_ascii=False, indent=1)
    spec["duration"] = dur
    json.dump(spec, open(spec_path, "w"), ensure_ascii=False, indent=2)
    print("สร้างแล้ว:", out, f"{dur:.1f}s")
    for name, t0, t1, idx in sb:
        print(f"  ฉาก {name:6s} {t0:6.2f}-{t1:6.2f}")
    import qa
    sys.exit(0 if qa.run(out, spec_path) else 1)


if __name__ == "__main__":
    main()
