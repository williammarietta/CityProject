import random, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
incoming = ROOT / "data" / "incoming"
train = ROOT / "data" / "train"
val = ROOT / "data" / "val"
CLASSES = ["mixed", "cardboard", "trash"]
EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}
VAL_RATIO = 0.2
random.seed(42)

def gather(folder: Path):
    return [p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in EXTS]

for base in [train, val]:
    for c in CLASSES: (base/c).mkdir(parents=True, exist_ok=True)

for c in CLASSES:
    src = incoming/c; tdst = train/c; vdst = val/c
    imgs = gather(src)
    if not imgs: print(f"⚠️  No images found in {src}"); continue
    random.shuffle(imgs); n_val = max(1, int(len(imgs)*VAL_RATIO))
    val_imgs = imgs[:n_val]; train_imgs = imgs[n_val:]
    for i,p in enumerate(train_imgs,1):
        (tdst/f"{c}_{i:05d}{p.suffix.lower()}").write_bytes(p.read_bytes()); p.unlink()
    for i,p in enumerate(val_imgs,1):
        (vdst/f"{c}_{i:05d}{p.suffix.lower()}").write_bytes(p.read_bytes()); p.unlink()
    print(f"✅ {c}: moved {len(train_imgs)} → train, {len(val_imgs)} → val")

print("\nDone. Summary:")
for c in CLASSES:
    t = len(list((train/c).glob("*"))); v = len(list((val/c).glob("*")))
    print(f"- {c:9s}  train={t:5d}  val={v:5d}")
