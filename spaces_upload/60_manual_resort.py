# 60_manual_resort.py — Manually re-file images into cardboard / mixed / trash.
# Windows-friendly. No "Skip" — choosing a bin moves the file and loads the next.
#
# It scans these folders if they exist:
#   data/train/**, data/val/**, data/incoming/**
#
# Where it moves files:
#   If the image is under data/train/..., it moves within data/train/<chosen_bin>/
#   If under data/val/..., it moves within data/val/<chosen_bin>/
#   If under data/incoming/..., it moves within data/incoming/<chosen_bin>/
#
# Undo last is provided in case of misclick.

from pathlib import Path
import shutil, json, os
from typing import List, Tuple, Optional
from PIL import Image
import gradio as gr
import time

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SPLITS = ["train", "val", "incoming"]
CLASSES = ["cardboard", "mixed", "trash"]
STATE_FILE = ROOT / "manual_resort_state.json"

# Which file types to show/handle
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic"}

def list_images() -> List[Path]:
    files = []
    for split in SPLITS:
        base = DATA_DIR / split
        if not base.exists():
            continue
        # grab images anywhere under this split (even if currently in the wrong subfolder)
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    # Sort for stability
    files = sorted(files)
    return files

def find_split_base(p: Path) -> Optional[Tuple[str, Path]]:
    """Return (split_name, split_base_path) for a given file path."""
    for split in SPLITS:
        base = DATA_DIR / split
        try:
            p.relative_to(base)
            return split, base
        except Exception:
            continue
    return None

def unique_dest(dest_dir: Path, name: str) -> Path:
    """Avoid overwriting by adding -1, -2, ... suffix if needed."""
    out = dest_dir / name
    if not out.exists():
        return out
    stem = out.stem
    suf = out.suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}-{i}{suf}"
        if not cand.exists():
            return cand
        i += 1

def load_state(files: List[Path]):
    # Default state
    state = {
        "queue": [str(p) for p in files],
        "index": 0,
        "moved_counts": {c: 0 for c in CLASSES},
        "undo_stack": []  # list of dicts: {"src": "...", "dest": "..."}
    }
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            # Keep only files that still exist
            s["queue"] = [f for f in s.get("queue", []) if Path(f).exists()]
            # If queue is empty, reset
            if not s["queue"]:
                return state
            # If index beyond end, clamp
            s["index"] = int(min(max(0, s.get("index", 0)), len(s["queue"]) - 1))
            # Ensure moved_counts keys exist
            s["moved_counts"] = {c: int(s.get("moved_counts", {}).get(c, 0)) for c in CLASSES}
            # Prune undo_stack entries whose paths no longer exist
            cleaned_undo = []
            for rec in s.get("undo_stack", []):
                if Path(rec.get("dest","")).exists():
                    cleaned_undo.append(rec)
            s["undo_stack"] = cleaned_undo
            return s
        except Exception:
            return state
    return state

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def current_image(state) -> Optional[Path]:
    q = state["queue"]
    idx = state["index"]
    if idx < 0 or idx >= len(q):
        return None
    return Path(q[idx])

def ensure_dest_dirs():
    # Create class folders under each split that exists
    for split in SPLITS:
        base = DATA_DIR / split
        if base.exists():
            for c in CLASSES:
                (base / c).mkdir(parents=True, exist_ok=True)

def preview_image(p: Path):
    """Return path for gr.Image (string) if previewable, else None + a message."""
    # Gradio can display a file path directly. HEIC may not preview; that's okay.
    try:
        # Try opening to catch corrupt files, but still allow display by path
        with Image.open(p) as im:
            im.verify()
        return str(p), ""
    except Exception as e:
        return None, f"(Preview not available: {e}. You can still move the file.)"

def format_status(state) -> str:
    total = len(state["queue"])
    idx = state["index"] + 1 if total > 0 else 0
    moved = sum(state["moved_counts"].values())
    return (
        f"File {idx} of {total} | "
        f"Moved — Cardboard: {state['moved_counts']['cardboard']}, "
        f"Mixed: {state['moved_counts']['mixed']}, "
        f"Trash: {state['moved_counts']['trash']} | "
        f"Undo available: {len(state['undo_stack'])}"
    )

def load_queue():
    ensure_dest_dirs()
    files = list_images()
    st = load_state(files)
    save_state(st)
    p = current_image(st)
    if p is None:
        return None, "No images found under data/train, data/val, or data/incoming.", format_status(st)
    img, note = preview_image(p)
    caption = f"{p}"
    if note:
        caption += f"\n{note}"
    return img, caption, format_status(st)

def move_to(label):
    assert label in CLASSES
    # load state
    st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    p = current_image(st)
    if p is None:
        return None, "All done — no more files.", format_status(st)

    # Find which split this file belongs to
    found = find_split_base(p)
    if not found:
        # If the file is not under data/{train,val,incoming}, default to data/incoming
        split_base = DATA_DIR / "incoming"
    else:
        split_name, split_base = found

    dest_dir = split_base / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    # If the file currently lives in a class folder, that's okay—we will move it if needed
    dest_path = unique_dest(dest_dir, p.name)
    # Make a record for undo
    rec = {"src": str(p), "dest": str(dest_path)}
    # Perform move
    shutil.move(str(p), str(dest_path))

    # Update queue: replace current path with new path (so subsequent moves refer to new location if we undo)
    st["queue"][st["index"]] = str(dest_path)
    # Update counters & undo
    st["moved_counts"][label] = st["moved_counts"].get(label, 0) + 1
    st["undo_stack"].append(rec)

    # Advance to next
    st["index"] += 1
    save_state(st)

    # Show next
    p2 = current_image(st)
    if p2 is None:
        return None, "Finished! No more files to label.", format_status(st)
    img, note = preview_image(p2)
    caption = f"{p2}"
    if note:
        caption += f"\n{note}"
    return img, caption, format_status(st)

def undo_last():
    if not STATE_FILE.exists():
        return None, "No state file.", ""
    st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    if not st.get("undo_stack"):
        # nothing to undo; just refresh current
        p = current_image(st)
        if p is None:
            return None, "Nothing to undo; and no more files.", format_status(st)
        img, note = preview_image(p)
        cap = f"{p}"
        if note:
            cap += f"\n{note}"
        return img, cap, format_status(st)

    rec = st["undo_stack"].pop()
    src = Path(rec["src"])   # where it came from originally
    dest = Path(rec["dest"]) # where we moved it to
    try:
        # move back (if dest still exists)
        if dest.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(src))
    except Exception as e:
        # Keep going even if undo fails
        pass

    # Step back index if possible, and point to restored file
    st["index"] = max(0, st["index"] - 1)
    if st["index"] < len(st["queue"]):
        st["queue"][st["index"]] = str(src)

    save_state(st)

    p = current_image(st)
    if p is None:
        return None, "Undo performed. No more files.", format_status(st)
    img, note = preview_image(p)
    cap = f"{p}"
    if note:
        cap += f"\n{note}"
    return img, cap, format_status(st)

with gr.Blocks(title="Manual Resort — Cardboard / Mixed / Trash") as demo:
    gr.Markdown(
        "# ♻️ Manual Resort\n"
        "One photo at a time. Choose **Cardboard**, **Mixed**, or **Trash** — the file is moved and the next photo loads.\n"
        "**No Skip.** Use **Undo** if you misclick.\n\n"
        "Scans: `data/train`, `data/val`, `data/incoming` (if they exist)."
    )

    img = gr.Image(label="Image", interactive=False, height=480)
    caption = gr.Textbox(label="File path", interactive=False)
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        btn_card = gr.Button("Send to CARDBOARD", variant="primary")
        btn_mix  = gr.Button("Send to MIXED")
        btn_trash= gr.Button("Send to TRASH")
        btn_undo = gr.Button("Undo last")

    btn_card.click(lambda: move_to("cardboard"), outputs=[img, caption, status])
    btn_mix.click(lambda: move_to("mixed"), outputs=[img, caption, status])
    btn_trash.click(lambda: move_to("trash"), outputs=[img, caption, status])
    btn_undo.click(undo_last, outputs=[img, caption, status])

    # Load queue when the app starts
    demo.load(load_queue, outputs=[img, caption, status])

# Local launch; you don't need HTTPS for this admin tool
demo.launch(inbrowser=True, debug=True)
