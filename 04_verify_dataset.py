from pathlib import Path

ROOT = Path(__file__).resolve().parent
for split in ["train", "val"]:
    base = ROOT / "data" / split
    print(f"\n[{split.upper()}]")
    if not base.exists():
        print("  (missing)")
        continue
    total = 0
    for cls in ["mixed", "cardboard", "trash"]:
        clsdir = base / cls
        n = len([p for p in clsdir.glob("*") if p.is_file()])
        print(f"  {cls:9s} : {n}")
        total += n
    print(f"  TOTAL     : {total}")
