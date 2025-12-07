# app.py — Hugging Face Spaces version (upload-only camera; Chesapeake rules + logging)

import json, csv
import gradio as gr
from PIL import Image
from pathlib import Path
from datetime import datetime

# ML helpers (assistive)
from inference import load_model, predict_pil

# --------------------
# Load labels & keywords
# --------------------
labels = json.load(open("labels.json", "r", encoding="utf-8"))
keywords = json.load(open("keywords_fallback.json", "r", encoding="utf-8"))

# --------------------
# Load trained model (optional; suggestion only)
# --------------------
CKPT = Path("models/recycler_v1.pt")
model = preprocess = idx_to_class = None
if CKPT.exists():
    try:
        model, preprocess, idx_to_class = load_model(CKPT)
    except Exception:
        model = preprocess = idx_to_class = None  # keep app running even if load fails

# --------------------
# Site options / flags
# --------------------
OPT_GLASS       = "Glass item"
OPT_GREASE      = "Visible grease residues or food on item"
OPT_CORRUGATED  = "Corrugated cardboard (fluted middle, common in shipping boxes)"
SPECIAL_OPTIONS = [OPT_GLASS, OPT_GREASE, OPT_CORRUGATED]

LOW_CONF_THRESHOLD = 0.65  # show hint when model is uncertain

# --------------------
# CSV logging (saved inside the Space)
# --------------------
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.csv"

def log_final(final_label: str, base_pred: str | None, conf: float | None, flags: list[str], corr_choice: str | None):
    write_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp","final_label","base_pred","confidence","flags","corr_choice"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            final_label,
            base_pred or "",
            f"{conf:.4f}" if conf is not None else "",
            "; ".join(flags or []),
            corr_choice or ""
        ])

# --------------------
# Helpers
# --------------------
def is_trash_by_flags(flags): 
    f=set(flags or []); return (OPT_GLASS in f) or (OPT_GREASE in f)

def is_corrugated_flag(flags): 
    return (flags is not None) and (OPT_CORRUGATED in set(flags))

def friendly_message(final_label, base_pred=None, conf=None):
    title = labels[final_label]["title"]
    hint  = labels[final_label]["hint"]
    parts = [f"{title} — {hint}"]
    if base_pred is not None and conf is not None:
        parts.append(f"(Model suggested: {base_pred} @ {conf:.2f})")
        if conf < LOW_CONF_THRESHOLD:
            parts.append("Low confidence — please double-check posted signage.")
    return "\n".join(parts)

def corrugation_prompt_text():
    return (
        "### Quick check: Is it **corrugated**?\n"
        "- **Yes** → Cardboard (shipping/moving box with a **fluted/wavy** inner layer)\n"
        "- **No / Not sure** → Mixed (thin **paperboard** like cereal/tissue boxes)\n"
        "_Photo tip: Include a box **edge/corner** so the flutes are visible._"
    )

# --------------------
# Core classify
# --------------------
def classify_image(img: Image.Image, flags: list[str]):
    """
    Priority:
      1) Glass/Grease flags -> TRASH (final)
      2) Corrugated checkbox -> CARDBOARD (final)
      3) Else: run model; if model says 'cardboard' or 'mixed', ask corrugation
      4) Else: finalize with model (e.g., 'trash')
    Returns: result text, corr_md, corr_radio, corr_btn, pending_pred, pending_conf
    """
    if img is None:
        return ("No image provided.",
                gr.update(visible=False),
                gr.update(visible=False, value=None),
                gr.update(visible=False),
                None, None)

    # 1) Hard TRASH overrides
    if is_trash_by_flags(flags):
        msg = friendly_message("trash", None, None)
        log_final("trash", None, None, flags, corr_choice=None)
        return (msg,
                gr.update(visible=False),
                gr.update(visible=False, value=None),
                gr.update(visible=False),
                None, None)

    # 2) Hard CARDBOARD override
    if is_corrugated_flag(flags):
        msg = friendly_message("cardboard", None, None)
        log_final("cardboard", None, None, flags, corr_choice="checkbox")
        return (msg,
                gr.update(visible=False),
                gr.update(visible=False, value=None),
                gr.update(visible=False),
                None, None)

    # 3) Model suggestion
    base_label, conf = (None, None)
    if model is not None:
        try:
            base_label, conf = predict_pil(img, model, preprocess, idx_to_class)
        except Exception:
            base_label, conf = (None, None)

    # Ask corrugation for cardboard or mixed predictions
    if base_label in ("cardboard","mixed"):
        placeholder = f"Model suggests **{base_label}** @ {conf:.2f}. Please confirm:"
        return (placeholder,
                gr.update(visible=True, value=corrugation_prompt_text()),
                gr.update(visible=True, value=None),
                gr.update(visible=True),
                base_label, conf)

    # 4) Finalize directly (trash or anything else)
    final_label = base_label if base_label in ("trash","mixed","cardboard") else "trash"
    msg = friendly_message(final_label, base_label, conf)
    log_final(final_label, base_label, conf, flags, corr_choice=None)
    return (msg,
            gr.update(visible=False),
            gr.update(visible=False, value=None),
            gr.update(visible=False),
            None, None)

# --------------------
# Corrugation confirmation
# --------------------
def confirm_corrugation(choice: str, flags: list[str], base_pred: str | None, conf: float | None):
    # Re-check rules in case user toggled now
    if is_trash_by_flags(flags):
        msg = friendly_message("trash", base_pred, conf)
        log_final("trash", base_pred, conf, flags, corr_choice="overridden-to-trash")
        return (msg,
                gr.update(visible=False), gr.update(visible=False, value=None), gr.update(visible=False),
                None, None)

    if is_corrugated_flag(flags):
        msg = friendly_message("cardboard", base_pred, conf)
        log_final("cardboard", base_pred, conf, flags, corr_choice="checkbox")
        return (msg,
                gr.update(visible=False), gr.update(visible=False, value=None), gr.update(visible=False),
                None, None)

    if not choice:
        return ("Please choose an option above.",
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                base_pred, conf)

    final = "cardboard" if choice.lower().startswith("yes") else "mixed"
    msg = friendly_message(final, base_pred, conf)
    log_final(final, base_pred, conf, flags, corr_choice=choice)
    return (msg,
            gr.update(visible=False), gr.update(visible=False, value=None), gr.update(visible=False),
            None, None)

# --------------------
# Type-to-text fallback
# --------------------
def text_lookup(q: str):
    if not q:
        return "Type an item (e.g., 'aluminum can', 'pizza box', 'shipping box')."
    ql = q.lower()
    for k, lab in keywords.items():
        if k in ql:
            return f"{labels[lab]['title']}: {labels[lab]['hint']}"
    return "Not found — please check the posted signage."

# --------------------
# UI (Upload-only: opens native camera on phones)
# --------------------
with gr.Blocks(title="CityProject — phone camera (upload-only)") as demo:
    gr.Markdown(
        "# ♻️ CityProject\n"
        "**Tip:** On phones, tap the image area and choose **Take Photo** to open the rear camera.\n"
        "If you see a permission prompt, choose **Allow**."
    )

    pending_pred = gr.State(None)
    pending_conf = gr.State(None)

    with gr.Tab("Take Photo / Upload"):
        img = gr.Image(
            sources=["upload"],     # opens native camera / photo library on phones
            label="Tap to Take Photo or choose from library",
            type="pil",
            height=420
        )

        flags = gr.CheckboxGroup(SPECIAL_OPTIONS, label="Special checks (tap all that apply)")
        out = gr.Textbox(label="Result", lines=5)

        # Corrugation confirmation UI (hidden until needed)
        corr_md = gr.Markdown(visible=False)
        corr_radio = gr.Radio(
            choices=["Yes — corrugated (shipping box)",
                     "No — thin paperboard (cereal/tissue box)",
                     "Not sure"],
            label="Confirm corrugation",
            visible=False
        )
        corr_btn = gr.Button("Confirm", visible=False)

        btn = gr.Button("Classify")
        btn.click(
            classify_image,
            inputs=[img, flags],
            outputs=[out, corr_md, corr_radio, corr_btn, pending_pred, pending_conf]
        )
        corr_btn.click(
            confirm_corrugation,
            inputs=[corr_radio, flags, pending_pred, pending_conf],
            outputs=[out, corr_md, corr_radio, corr_btn, pending_pred, pending_conf]
        )

    with gr.Tab("Type the item"):
        q = gr.Textbox(label="e.g., 'aluminum can', 'pizza box', 'shipping box'")
        out2 = gr.Textbox(label="Result")
        q.submit(text_lookup, q, out2)

# IMPORTANT for Spaces: keep this minimal launch (no share/server args)
demo.launch()
