"""แปลงข้อความให้พร้อมอ่านออกเสียง (โมเดล MMS-TTS ไทยอ่านได้เฉพาะอักษรไทย)

ตัวเลข -> คำอ่านไทย, ศัพท์อังกฤษ -> คำทับศัพท์ไทยจากพจนานุกรม
คำที่ไม่รู้จักจะ "ไม่ถูกเดา" แต่ raise UnknownWord เพื่อให้เพิ่มคำอ่านเองก่อน
"""
import re
from pythainlp.util import num_to_thaiword

LEXICON = {
    "type-c": "ไทป์ซี", "typec": "ไทป์ซี", "usb": "ยูเอสบี", "usb-c": "ยูเอสบีซี",
    "bluetooth": "บลูทูธ", "wifi": "ไวไฟ", "wi-fi": "ไวไฟ", "led": "แอลอีดี",
    "hd": "เอชดี", "4k": "โฟร์เค", "gps": "จีพีเอส", "ai": "เอไอ", "pc": "พีซี",
    "iphone": "ไอโฟน", "android": "แอนดรอยด์", "ipad": "ไอแพด", "mm": "มิลลิเมตร",
    "cm": "เซนติเมตร", "kg": "กิโลกรัม", "gb": "กิกะไบต์", "tb": "เทระไบต์",
    "mah": "มิลลิแอมป์ชั่วโมง", "w": "วัตต์", "v": "โวลต์", "hz": "เฮิรตซ์",
    "tommy": "ทอมมี่", "hilfiger": "ฮิลฟิเกอร์",
    "x1": "เอ็กซ์วัน", "pro": "โปร", "max": "แม็กซ์", "mini": "มินิ", "plus": "พลัส",
}
LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz",
    "เอ บี ซี ดี อี เอฟ จี เอช ไอ เจ เค แอล เอ็ม เอ็น โอ พี คิว อาร์ เอส ที ยู วี ดับเบิลยู เอ็กซ์ วาย แซด".split()))


class UnknownWord(ValueError):
    pass


def _num(m):
    s = m.group(0).replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        digits = "".join(num_to_thaiword(int(d)) for d in frac)
        return f" {num_to_thaiword(int(whole))}จุด{digits} "
    return f" {num_to_thaiword(int(s))} "


def normalize(text: str) -> str:
    t = text.strip()
    t = re.sub(r"฿\s*([\d,\.]+)", lambda m: f"{m.group(1)} บาท", t)
    t = re.sub(r"(\d+)\s*%", r"\1 เปอร์เซ็นต์", t)
    # ศัพท์ในพจนานุกรม (ยาวก่อน)
    for k in sorted(LEXICON, key=len, reverse=True):
        t = re.sub(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![A-Za-z0-9])", f" {LEXICON[k]} ", t, flags=re.I)
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", _num, t)
    t = t.replace("/", " ").replace("+", " บวก ").replace("&", " และ ")
    t = re.sub(r"[!?.,;:\"'()\[\]{}\-–—•✓*]", " ", t)
    left = re.findall(r"[A-Za-z]+", t)
    if left:
        raise UnknownWord(f"ไม่มีคำอ่านไทยสำหรับ {left} — เพิ่มใน LEXICON หรือใส่ 'say' เป็นอักษรไทย")
    bad = [c for c in t if not (c.isspace() or "฀" <= c <= "๿")]
    if bad:
        raise UnknownWord(f"มีอักขระที่อ่านไม่ได้: {sorted(set(bad))}")
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    for s in ["อะแดปเตอร์ Type-C เป็นแจ็ค 3.5", "ราคา ฿590 ลด 20%", "แบตอึด 30 ชั่วโมง"]:
        print(s, "->", normalize(s))
