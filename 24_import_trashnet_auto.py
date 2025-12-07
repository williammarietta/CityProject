from pathlib import Path
import shutil, hashlib, zipfile, sys
from huggingface_hub import snapshot_download

USER = "billy"  # change if your Windows username is not 'billy'
DOWNLOAD_DIR = Path(fr"C:\Users\{USER}\Downloads\trashnet_clean")
PROJECT_ROOT = Path(__file__).resolve().parent
INCOMING = PROJECT_ROOT / "data" / "incoming"
INCOMING.mkdir(parents=True, exist_ok=True)
for c in ["mixed", "cardboard", "trash"]:
    (INCOMING / c).mkdir(parents=True, exist_ok=True)

SRC_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_MAP = {"cardboard":"cardboard","glass":"trash","trash":"trash","metal":"mixed","paper":"mixed","plastic":"mixed"}
EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}

def md5(path: Path):
    h=hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<16), b""): h.update(chunk)
    return h.hexdigest()

def preload_seen():
    seen=set()
    for c in ["mixed","cardboard","trash"]:
        for p in (INCOMING/c).glob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                try: seen.add(md5(p))
                except: pass
    return seen

def find_parent_with_all(base: Path):
    need=set(SRC_CLASSES)
    for p in base.rglob("*"):
        if not p.is_dir(): continue
        names={d.name.lower() for d in p.iterdir() if d.is_dir()}
        if need.issubset(names): return p
    return None

def main():
    # download (if folder empty)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not any(DOWNLOAD_DIR.iterdir()):
        print("⬇️  Downloading TrashNet…"); snapshot_download("garythung/trashnet", repo_type="dataset", local_dir=str(DOWNLOAD_DIR)); print("✅ Download finished.")
    # extract inner zip if present
    zips=list(DOWNLOAD_DIR.rglob("dataset-resized.zip"))
    if zips:
        z=zips[0]; dest=z.parent/"dataset-resized"
        if not (dest.exists() and any(dest.iterdir())):
            print(f"🗜️  Extracting {z} → {dest}"); dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z,"r") as zf: zf.extractall(dest)
            print("✅ Extraction done.")
    parent=find_parent_with_all(DOWNLOAD_DIR)
    if not parent:
        print("❌ Could not find a folder that directly contains the 6 class folders under", DOWNLOAD_DIR); sys.exit(1)
    print("📁 Importing from:", parent)
    seen=preload_seen()
    total=0
    for cls in SRC_CLASSES:
        src=parent/cls
        if not src.is_dir(): print("⚠️  Missing:", src); continue
        dest_bin=CLASS_MAP[cls]; dest=INCOMING/dest_bin; copied=0
        for p in src.iterdir():
            if not (p.is_file() and p.suffix.lower() in EXTS): continue
            try:
                h=md5(p)
                if h in seen: continue
                base=f"{cls}_{p.stem}"; dst=dest/f"{base}{p.suffix.lower()}"; i=1
                while dst.exists(): dst=dest/f"{base}_{i}{p.suffix.lower()}"; i+=1
                shutil.copy2(p,dst); seen.add(h); copied+=1
            except Exception as e:
                print("Skip", p, e)
        print(f"✅ {cls:9s} → {dest_bin:9s} : {copied:4d} files")
        total+=copied
    print(f"\nDone. Total copied into '{INCOMING}': {total}")

if __name__=="__main__": main()
