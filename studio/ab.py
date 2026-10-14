"""เทียบความชัดของถ้อยคำทางเลือกด้วยการถอดเสียงย้อนกลับ:  python3 ab.py "ประโยค1" "ประโยค2" ..."""
import sys, re, difflib, numpy as np, soundfile as sf
from tts import synth, SR
from asr import transcribe
from thai_text import normalize
clean = lambda s: re.sub(r"[^฀-๿]", "", s)
for line in sys.argv[1:]:
    ref = normalize(line); x = synth(line)
    sf.write("/tmp/ab.wav", np.concatenate([np.zeros(6400), x, np.zeros(6400)]), SR)
    hyp = transcribe("/tmp/ab.wav")
    print(f"{difflib.SequenceMatcher(None, clean(ref), clean(hyp)).ratio():.2f}  {len(x)/SR:.2f}s  {ref}  ->  {hyp}")
