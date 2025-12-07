from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
data_dirs = [ROOT/'data'/'train', ROOT/'data'/'val']
MAX_SIDE = 1280

def process_dir(d: Path):
    count=0
    for p in d.rglob("*"):
        if not p.is_file(): continue
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                w,h = im.size; m=max(w,h)
                if m>MAX_SIDE:
                    scale = MAX_SIDE/m
                    im = im.resize((int(w*scale), int(h*scale)))
                if p.suffix.lower() != ".jpg":
                    newp = p.with_suffix(".jpg"); im.save(newp,"JPEG",quality=90); p.unlink()
                else:
                    im.save(p,"JPEG",quality=90)
                count+=1
        except Exception as e:
            print(f"Skip {p.name}: {e}")
    return count

for base in data_dirs:
    if not base.exists(): print(f"⚠️  Missing {base}"); continue
    for clsdir in base.iterdir():
        if clsdir.is_dir():
            n = process_dir(clsdir)
            print(f"✅ Cleaned {n} images in {clsdir}")

print("Done.")
