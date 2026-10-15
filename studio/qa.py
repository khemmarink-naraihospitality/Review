#!/usr/bin/env python3
"""ตรวจคลิปก่อนส่ง: สเปกไฟล์, ความดัง, จังหวะเสียง, ถอดเสียงย้อนกลับ, เฟรมไม่ว่าง, contact sheet

  python3 qa.py out.mp4 specs/xxx.json
คืนค่า exit 0 เมื่อผ่านทุกข้อบังคับ (FAIL = ไม่ผ่าน, WARN = ควรฟัง/ดูเอง)
"""
import difflib, json, os, re, subprocess, sys, tempfile
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def probe(path):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return j, v, a


def loudness(path):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    sec = p.stderr.split("Summary:")[-1]
    I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", sec).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", sec).group(1))
    return I, tp


def clean(s):
    return re.sub(r"[^\u0e00-\u0e7f]", "", s)


def run(video, spec_path):
    from thai_text import normalize
    spec = json.load(open(spec_path))
    timing = json.load(open(video + ".timing.json"))
    res = []  # (level, name, detail)

    def add(level, name, detail):
        res.append((level, name, detail))

    j, v, a = probe(video)
    dur = float(j["format"]["duration"])
    want_d = float(spec.get("duration", 10.0))
    add("PASS" if abs(dur - want_d) <= 0.05 else "FAIL", "ความยาว", f"{dur:.2f}s (ต้อง {want_d:.2f}±0.05)")
    want = tuple(spec.get("size", [1080, 1920]))
    ok = (v["width"], v["height"]) == want and v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"
    add("PASS" if ok else "FAIL", "วิดีโอ", f'{v["width"]}x{v["height"]} {v["codec_name"]} {v["pix_fmt"]} {v["r_frame_rate"]}')
    if a is None:
        add("FAIL", "เสียง", "ไม่มีแทร็กเสียง")
    else:
        add("PASS" if a["codec_name"] == "aac" and int(a["sample_rate"]) == 48000 else "FAIL", "เสียง",
            f'{a["codec_name"]} {a["sample_rate"]}Hz ch={a["channels"]}')
        I, tp = loudness(video)
        add("PASS" if -18 <= I <= -14 else "FAIL", "ความดัง", f"{I:.1f} LUFS (เป้า -16, ช่วง -18..-14)")
        add("PASS" if tp <= -1.0 else "FAIL", "ระดับพีค", f"{tp:.1f} dBTP (ต้อง ≤ -1.0)")

        tmp = tempfile.mkdtemp()
        wav = os.path.join(tmp, "a.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
        x, sr = sf.read(wav, dtype="float32")
        if spec.get("music", False):   # มีดนตรีตลอด: ดูเวลาพูดจากไทม์ไลน์ + ต้องเฟดจบ ไม่ตัดกุด
            s0, s1 = timing["segments"][0]["start"], timing["segments"][-1]["end"]
            tail = x[-int(0.15 * sr):]
            tail_db = 20 * np.log10(np.sqrt((tail ** 2).mean()) + 1e-9)
            add("PASS" if tail_db < -40 else "FAIL", "ดนตรีเฟดตอนจบ", f"{tail_db:.1f} dB ใน 0.15s สุดท้าย (ต้อง < -40)")
            gaps = [(a["end"], b["start"]) for a, b in zip(timing["segments"], timing["segments"][1:])]
            mid = np.concatenate([x[int(g0 * sr):int(g1 * sr)] for g0, g1 in gaps if g1 - g0 > 0.1])
            mus_db = 20 * np.log10(np.sqrt((mid ** 2).mean()) + 1e-9)
            add("PASS" if mus_db > -45 else "FAIL", "ได้ยินดนตรีพื้นหลัง", f"{mus_db:.1f} dB ช่วงหยุดพูด (ต้อง > -45)")
            import listen   # หู AI (AudioSet) ฟังผ่านลำโพงมือถือจำลอง
            rows = listen.report(listen.load(video))
            frac = sum(r["music"] >= 0.3 for r in rows) / len(rows)
            add("PASS" if frac >= 0.6 else "FAIL", "หู AI ได้ยินดนตรี (มือถือ)",
                f"{frac:.0%} ของช่วงเวลา, เฉลี่ย {np.mean([r['music'] for r in rows]):.2f} (ต้อง ≥60%)")
        else:
            act = np.where(np.abs(x) > 0.01)[0]
            s0, s1 = act[0] / sr, act[-1] / sr
        add("PASS" if s0 >= 0.15 else "FAIL", "เสียงเริ่ม", f"{s0:.2f}s (ไม่ควรตัดหัว ≥0.15s)")
        add("PASS" if s1 <= dur - 0.3 else "FAIL", "เสียงจบ", f"{s1:.2f}s (ต้องจบก่อน {dur - 0.3:.2f}s ไม่ถูกตัดท้าย)")

        # ถอดเสียงย้อนกลับทีละประโยค (ใช้เป็นตัวชี้วัดความชัด ไม่ใช่การตัดสินสุดท้าย)
        from asr import transcribe
        sims = []
        for seg, s in zip(timing["segments"], spec["segments"]):
            lo, hi = max(0, int((seg["start"] - 0.4) * sr)), min(len(x), int((seg["end"] + 0.4) * sr))
            p = os.path.join(tmp, "s.wav")
            sf.write(p, x[lo:hi], sr)
            hyp = transcribe(p)
            ref = normalize(s.get("say") or s["text"])
            r = difflib.SequenceMatcher(None, clean(ref), clean(hyp)).ratio()
            sims.append(r)
            add("PASS" if r >= 0.6 else "WARN", "ถอดเสียงกลับ", f"{r:.2f} | อ่าน: {ref} | ได้ยินเป็น: {hyp}")
        mean = sum(sims) / len(sims)
        add("PASS" if mean >= 0.6 else ("WARN" if mean >= 0.4 else "FAIL"), "ความชัดเฉลี่ย", f"{mean:.2f} (≥0.6 ดี, <0.4 ไม่ผ่าน)")

    # เฟรมต้องไม่ว่าง/ไม่ค้าง + contact sheet
    tmp2 = tempfile.mkdtemp()
    times = [0.6] + [(m["start"] + m["end"]) / 2 for m in timing["segments"]] + [dur - float(spec.get("fade_out", 0.0)) - 0.25]   # ก่อนช่วงจางจบที่ตั้งใจ
    frames = []
    for i, t in enumerate(times):
        fp = os.path.join(tmp2, f"{i}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", fp], check=True)
        frames.append(np.asarray(Image.open(fp).convert("RGB")).astype(np.float32))
    blank = [i for i, f in enumerate(frames) if f.std() < 12]
    add("PASS" if not blank else "FAIL", "เฟรมไม่ว่าง", f"ตรวจ {len(frames)} เฟรม" + (f" ว่าง: {blank}" if blank else ""))
    diffs = [np.abs(frames[i] - frames[i + 1]).mean() for i in range(len(frames) - 1)]
    add("PASS" if min(diffs) > 0.5 else "WARN", "ภาพเปลี่ยนตามประโยค", f"ผลต่างต่ำสุด {min(diffs):.2f}")
    tw = 270 if want[1] > want[0] else 400
    th = int(tw * want[1] / want[0])
    sheet = Image.new("RGB", (tw * len(frames), th), (0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(Image.fromarray(f.astype(np.uint8)).resize((tw, th)), (tw * i, 0))
    sheet_path = video + ".contact.png"
    sheet.save(sheet_path)

    width = max(len(n) for _, n, _ in res)
    print("\n== ตรวจก่อนส่ง ==")
    for lv, n, d in res:
        print(f"[{lv}] {n:<{width}}  {d}")
    fails = [r for r in res if r[0] == "FAIL"]
    warns = [r for r in res if r[0] == "WARN"]
    print(f"\nสรุป: {'ผ่าน' if not fails else 'ไม่ผ่าน'}  (FAIL {len(fails)}, WARN {len(warns)})  contact sheet: {sheet_path}")
    json.dump([{"level": l, "check": n, "detail": d} for l, n, d in res], open(video + ".qa.json", "w"), ensure_ascii=False, indent=1)
    return not fails


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1], sys.argv[2]) else 1)
