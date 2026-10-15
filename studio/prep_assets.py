"""เตรียมวัตถุดิบจากรูปสินค้า: ตัดพื้นหลัง + ขยายภาพรายละเอียดด้วย EDSR x4 (ออฟไลน์)"""
import os, sys, numpy as np, cv2
os.environ["U2NET_HOME"] = "/opt/vstudio/models/u2net"
from rembg import new_session, remove
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
im = Image.open(src).convert("RGB")
cut = remove(im, session=new_session("isnet-general-use"), alpha_matting=False)
cut.save(f"{out}/cutout.png")
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("/opt/vstudio/models/sr/EDSR_x4.pb"); sr.setModel("edsr", 4)
big = sr.upsample(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{out}/photo_x4.png", big)
print("cutout", cut.size, "x4", big.shape[1], big.shape[0])
