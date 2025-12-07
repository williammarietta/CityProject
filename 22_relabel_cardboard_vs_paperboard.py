# 22_relabel_cardboard_vs_paperboard.py
# Relabel tool WITHOUT a Skip/Next button.
# After you click one of the three actions, it auto-advances to the next image.
#
# Buttons:
#   - ✅ Keep as Corrugated Cardboard  (stays in cardboard)
#   - 📦 Move to Paperboard (MIXED)    (moves to mixed)
#   - 🗑️ Move to Trash (Greasy/Contam.) (moves to trash)
#
# Run: python 22_relabel_cardboard_vs_paperboard.py

from pathlib import Path
import shutil
import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# We review images currently labeled as "cardboard" in both splits
REVIEW_DIRS = [
    DATA / "train" / "cardboard",
    DATA / "val"   / "cardboard",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def gather_images():
    files = []
    for d in REVIEW_DIRS:
        if d.exists():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    files.append(p)
    # Stable order so you can resume mentally if you stop/restart
    files.sort(key=lambda p: p.name.lower())
    return files

# Global list of remaining images to review
ALL = gather_images()

def load_image(idx: int):
    """Load image at index, return (PIL image or None, status text, normalized idx)."""
    if not ALL:
        return None, "Done — no images left to review.", 0
    # clamp idx to range
    idx = max(0, min(int(idx), len(ALL) - 1))
    path = ALL[idx]
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        # remove unreadable image & advance
        bad = ALL.pop(idx)
        if not ALL:
            return None, f"Removed unreadable: {bad.name}\nDone — no images left.", 0
        # show next at same index (since current was removed)
        path2 = ALL[min(idx, len(ALL)-1)]
        im = Image.open(path2).convert("RGB")
        status = f"Removed unreadable: {bad.name}\n[{idx+1}/{len(ALL)}] {path2}"
        return im, status, min(idx, len(ALL)-1)

    status = f"[{idx+1}/{len(ALL)}] {path}"
    return im, status, idx

def keep_cardboard(idx: int):
    """Keep current image in cardboard and advance to the next."""
    if not ALL:
        return None, "Done — no images left.", 0

    idx = max(0, min(int(idx), len(ALL) - 1))
    stayed = ALL[idx].name

    # Advance to the next image (do NOT remove from list)
    next_idx = idx + 1
    if next_idx >= len(ALL):
        # we were at the last image
        return None, f"Kept as CARDBOARD: {stayed}\nDone — no images left.", idx

    im, _, normalized = load_image(next_idx)
    remaining = len(ALL) - next_idx
    msg = f"Kept as CARDBOARD: {stayed}\nRemaining: {remaining}\n[{normalized+1}/{len(ALL)}]"
    return im, msg, normalized

def move_current(idx: int, dest_class: str):
    """
    Move current image to mixed or trash, remove it from review list,
    then show the next image (same index now points to next).
    """
    if not ALL:
        return None, "Done — no images left.", 0

    idx = max(0, min(int(idx), len(ALL) - 1))
    src = ALL[idx]

    # Determine split from path: .../data/<split>/cardboard/<file>
    split = src.parents[1].name  # "train" or "val"
    dest_dir = DATA / split / dest_class
    dest_dir.mkdir(parents=True, exist_ok=True)

    dst = dest_dir / src.name
    i = 1
    while dst.exists():
        dst = dest_dir / f"{src.stem}_{i}{src.suffix.lower()}"
        i += 1

    shutil.move(str(src), str(dst))
    moved_name = dst.name

    # Remove from list; now ALL[idx] is the next image (if any)
    ALL.pop(idx)

    if not ALL:
        return None, f"Moved → {dest_class.upper()}: {moved_name}\nDone — no images left.", 0

    # Stay at same index to show the next item that slid into this position
    im, _, normalized = load_image(min(idx, len(ALL)-1))
    remaining = len(ALL) - normalized
    msg = f"Moved → {dest_class.upper()}: {moved_name}\nRemaining: {len(ALL)}\n[{normalized+1}/{len(ALL)}]"
    return im, msg, normalized

def move_to_mixed(idx: int):
    return move_current(idx, "mixed")

def move_to_trash(idx: int):
    return move_current(idx, "trash")

with gr.Blocks(title="Relabel: Corrugated vs Paperboard (No Skip)") as demo:
    gr.Markdown(
        "# 🛠️ Relabel: Corrugated Cardboard vs Paperboard (Mixed)\n"
        "For each image currently labeled **cardboard**, pick one:\n\n"
        "• **✅ Keep as Corrugated Cardboard** — leave it in `cardboard/`\n\n"
        "• **📦 Move to Paperboard (MIXED)** — move to `mixed/`\n\n"
        "• **🗑️ Move to Trash (Greasy/Contam.)** — move to `trash/`\n\n"
        "After you click, it will **automatically advance** to the next image."
    )

    idx_state = gr.State(0)
    img = gr.Image(label="Image", height=420)
    info = gr.Textbox(label="Status", lines=3)

    # Initial load
    demo.load(lambda: load_image(0), outputs=[img, info, idx_state])

    btn_keep  = gr.Button("✅ Keep as Corrugated Cardboard")
    btn_mixed = gr.Button("📦 Move to Paperboard (MIXED)")
    btn_trash = gr.Button("🗑️ Move to Trash (Greasy/Contam.)")

    btn_keep.click(keep_cardboard, inputs=[idx_state], outputs=[img, info, idx_state])
    btn_mixed.click(move_to_mixed, inputs=[idx_state], outputs=[img, info, idx_state])
    btn_trash.click(move_to_trash, inputs=[idx_state], outputs=[img, info, idx_state])

demo.launch(share=False, server_name="127.0.0.1", server_port=7870, inbrowser=True, debug=True)
