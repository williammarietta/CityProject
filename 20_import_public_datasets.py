# 20_import_public_datasets.py
# Usage (Windows PowerShell in VS Code):
#   python 20_import_public_datasets.py "C:\path\trashnet" "C:\path\kaggle\garbage-classification"

from pathlib import Path
import shutil, sys, hashlib

ROOT = Path(__file__).resolve().parent
INCOMING = ROOT / "data" / "incoming"
INCOMING.mkdir(parents=True, exist_ok=True)
for c in ["mixed", "cardboard", "trash"]:
    (INCOMING / c).mkdir(parents=True, exist_ok=True)

SRC_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_MAP = {
    "cardboard": "cardboard",
    "glass": "trash",
    "trash": "trash",
    "metal": "mixed",
    "paper": "mixed",
    "plastic": "mixed",
}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def preload_seen_hashes() -> set[str]:
    seen = set()
    for c in ["mixed", "cardboard", "trash"]:
        for p in (INCOMING / c).glob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                try:
                    seen.add(md5_of(p))
                except Exception:
                    pass
    return seen

def find_class_dir(base: Path, name: str) -> Path | None:
    direct = base / name
    if direct.is_dir():
        return direct
    candidates = [p for p in base.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
    if not candidates:
        return None
    # choose directory with most image files
    def count_imgs(d: Path) -> int:
        return sum(1 for f in d.glob("*") if f.is_file() and f.suffix.lower() in EXTS)
    candidates.sort(key=count_imgs, reverse=True)
    return candidates[0]

def import_one_source(src_root: Path, seen: set[str]) -> int:
    total = 0
    for cls in SRC_CLASSES:
        src = find_class_dir(src_root, cls)
        if src is None:
            print(f"⚠️  Skipping '{cls}': not found under {src_root}")
            continue
        dest_bin = CLASS_MAP[cls]
        dest_dir = INCOMING / dest_bin
        copied = 0
        for p in src.glob("*"):
            if not (p.is_file() and p.suffix.lower() in EXTS):
                continue
            try:
                h = md5_of(p)
                if h in seen:
                    continue
                # create a unique filename
                base = f"{cls}_{p.stem}"
                dst = dest_dir / f"{base}{p.suffix.lower()}"
                i = 1
                while dst.exists():
                    dst = dest_dir / f"{base}_{i}{p.suffix.lower()}"
                    i += 1
                shutil.copy2(p, dst)
                seen.add(h)
                copied += 1
            except Exception as e:
                print(f"Skip {p} ({e})")
        print(f"✅ {cls:9s} → {dest_bin:9s} : {copied:4d} files from {src_root}")
        total += copied
    return total

def main():
    sources = [Path(s) for s in sys.argv[1:]]
    if not sources:
        print("Usage:\n  python 20_import_public_datasets.py \"C:\\path\\trashnet\" \"C:\\path\\kaggle\\garbage-classification\"")
        return
    seen = preload_seen_hashes()
    grand_total = 0
    for src in sources:
        if not src.exists():
            print(f"❌ Not found: {src}")
            continue
        grand_total += import_one_source(src, seen)
    print(f"\nDone. Total copied into 'data/incoming/': {grand_total}")

if __name__ == "__main__":
    main()
