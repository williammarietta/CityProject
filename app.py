# app.py — Chesapeake Sorting Assistant
# NOTE:
# - Search Assistant UPDATED (new approach):
#     * Single Textbox stays ("Search for an item")
#     * Suggestions are NOT a Dropdown anymore (no second bar).
#     * Suggestions appear as a Radio list ONLY after typing >= 1 character.
#     * Clicking a suggestion fills the textbox + hides the list + shows result.
# - Visual Helper (Beta) unchanged from your version.

import json
import re
from pathlib import Path
import importlib.util

import gradio as gr


# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "reference" / "keywords_fallback.json"


# Canonical category keys -> nice labels
CATS = {
    "hazard": "Hazardous Waste / E-Waste",
    "cardboard": "Corrugated Cardboard (Container 2)",
    "mixed": "Mixed Recyclables (Container 1)",
    "trash": "Trash / Not Accepted",
}


# ---------- Helpers to normalize category labels ----------


def normalize_category(raw_cat: str) -> str:
    """
    Turn whatever top-level key is in keywords_fallback.json
    into one of: hazard, trash, cardboard, mixed, or unknown.
    """
    lc = (raw_cat or "").lower()
    if "hazard" in lc or "hhw" in lc or "e-waste" in lc or "e waste" in lc:
        return "hazard"
    if "cardboard" in lc or "corrugated" in lc:
        return "cardboard"
    if "mixed" in lc or "recyclable" in lc:
        return "mixed"
    if "trash" in lc or "not accepted" in lc:
        return "trash"
    return "unknown"


# ---------- Load data from your canonical JSON ----------


if not DATA_PATH.exists():
    raise SystemExit(f"❌ Missing {DATA_PATH}")

with DATA_PATH.open("r", encoding="utf-8") as f:
    raw = json.load(f)

records = []
for raw_cat, items in raw.items():
    cat_key = normalize_category(raw_cat)
    for item in items:
        text = item or ""
        bulk = "(bulk pickup)" in text.lower()
        clean = re.sub(r"\(bulk pickup\)", "", text, flags=re.I).strip()
        records.append(
            {
                "item": clean,  # display name
                "name": clean.lower(),  # lowercase for searching
                "category": cat_key,  # hazard / trash / cardboard / mixed / unknown
                "category_raw": raw_cat,  # original label
                "bulk": bulk,
            }
        )

ALL_ITEMS = sorted({r["item"] for r in records}, key=str.lower)


# ---------- Text templates (your exact wording) ----------


def classification_message(category_key: str, bulk: bool) -> str:
    """
    Detailed text for each category + bulk pickup note.
    """
    if category_key == "hazard":
        base = (
            "This item is considered either as household hazard waste or electronic waste. "
            "Please drop this item off at the Southeastern Public Service Authority Center on "
            "901 Hollowell Lane, Chesapeake, Virginia."
        )
    elif category_key == "trash":
        base = (
            "This item is considered as trash. Please do not drop off this item at this recycling center."
        )
    elif category_key == "cardboard":
        base = (
            'This item is recyclable! Please place this item into the "Corrugated Cardboard" Dumpster.'
        )
    elif category_key == "mixed":
        base = (
            'This item is recyclable! Please place this item into the "Mixed Recyclables" dumpster.'
        )
    else:
        base = "No guidance available."

    if bulk:
        base += (
            " This item is also considered as a bulk pickup item. Please call 757-382-2489 "
            "to schedule a bulk pickup if needed."
        )

    return base


def best_option_label(category_key: str) -> str:
    """
    Short 'Best disposal option' line for the card.
    """
    if category_key == "hazard":
        return "Household Hazardous Waste / E-Waste (SPSA facility)"
    if category_key == "trash":
        return "Trash — Do NOT drop at this recycling center"
    if category_key == "cardboard":
        return '"Corrugated Cardboard" Dumpster'
    if category_key == "mixed":
        return '"Mixed Recyclables" Dumpster'
    return "No guidance available"


def render_detail_panel(category_key: str, bulk: bool, item_name: str) -> str:
    """
    Detail card that feels like a Waste Wizard material card.
    """
    title = item_name or "Classified item"
    best_line = best_option_label(category_key)
    label = CATS.get(category_key, "Chesapeake material")
    details = classification_message(category_key, bulk)

    return f"""
<div style='border:1px solid #ddd;border-radius:12px;padding:16px;background:#fff;
            box-shadow:0 2px 6px rgba(0,0,0,0.05);
            font-family:Inter,-apple-system,Helvetica,Arial,sans-serif;
            color:#000 !important;line-height:1.45;max-width:640px;'>
  <div style="font-size:18px;font-weight:600;margin-bottom:4px;color:#000 !important;">{title}</div>
  <div style="font-size:13px;color:#555;margin-bottom:6px;">
    City of Chesapeake guidance · {label}
  </div>

  <div style="margin-bottom:8px;">
    <span style="font-size:13px;font-weight:600;color:#111;">Best disposal option:</span>
    <span style="margin-left:4px;font-size:13px;color:#1f2933;">{best_line}</span>
  </div>

  <div style="margin-top:8px;font-size:14px;color:#222;">
    {details}
  </div>

  <div style="margin-top:12px;font-size:11px;color:#777;">
    This information is intended for City of Chesapeake, VA recycling drop-off locations.
    Always follow on-site signage and staff instructions.
  </div>
</div>
"""


# ---------- Matching logic (internal encyclopedia search) ----------


def _score_match(query: str, record_name: str) -> float:
    """
    Higher = better match. Favors exact, then startswith, then contains.
    """
    q = query.lower()
    name = record_name.lower()
    if not q:
        return 0.0
    if q == name:
        return 100.0
    if name.startswith(q):
        return 80.0
    if q in name:
        return 50.0
    return 0.0


def _scored_matches(query: str):
    q = (query or "").strip().lower()
    if not q:
        return []
    scored = []
    for r in records:
        score = _score_match(q, r["name"])
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], x[1]["item"].lower()))
    return [r for _, r in scored]


def best_match(query: str):
    matches = _scored_matches(query)
    return matches[0] if matches else None


# ---------- Search + suggestions (Search Assistant tab) ----------


def search_item(query: str):
    """
    Called when the user presses the Search button.
    They can type partial words; we pick the best match.
    """
    q = (query or "").strip()
    if not q:
        return '<em>Start typing an item name, then select a match from the list.</em>'

    match = best_match(q)
    if not match:
        return (
            f'<em>No result for "{q}". Try another word (e.g., "box", "bottle", "paint").</em>'
        )

    return render_detail_panel(match["category"], match["bulk"], match["item"])


def live_filter_suggestions(query: str):
    """
    NEW (Radio suggestions):
    - Empty textbox -> hide suggestions
    - 1+ char -> show Radio list of matches
    """
    q = (query or "").strip().lower()

    if len(q) == 0:
        return gr.Radio(choices=[], value=None, visible=False)

    starts = [item for item in ALL_ITEMS if item.lower().startswith(q)]
    contains = [item for item in ALL_ITEMS if q in item.lower() and item not in starts]
    choices = (starts + contains)[:75]

    return gr.Radio(choices=choices, value=None, visible=True)


def on_suggestion_pick(choice):
    """
    When user clicks a suggestion:
    - put it into the textbox
    - hide suggestions list
    - show result
    """
    if not choice:
        return (
            gr.Textbox(value=""),
            gr.Radio(visible=False),
            '<em>Start typing an item name, then select a match from the list.</em>',
        )

    return (
        gr.Textbox(value=choice),
        gr.Radio(visible=False),
        search_item(choice),
    )


# ---------- Visual Helper (Beta) + ML fallback ----------

_INFERENCE_MODULE = None  # cached module


def _infer_category_from_label(text: str) -> str:
    t = (text or "").lower()
    if "hazard" in t or "hhw" in t or "e-waste" in t or "e waste" in t:
        return "hazard"
    if "cardboard" in t or "corrugated" in t:
        return "cardboard"
    if "mixed" in t and "recycl" in t:
        return "mixed"
    if "recycle" in t or "recyclable" in t:
        return "mixed"
    if "trash" in t or "landfill" in t or "not accepted" in t:
        return "trash"
    return "unknown"


def _load_inference_module():
    global _INFERENCE_MODULE
    if _INFERENCE_MODULE is not None:
        return _INFERENCE_MODULE

    path = ROOT / "inference.py"
    if not path.exists():
        raise RuntimeError("inference.py not found in project folder.")

    spec = importlib.util.spec_from_file_location("inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load inference.py module spec.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _INFERENCE_MODULE = module
    return module


def _run_ml_classifier(image):
    mod = _load_inference_module()

    fn = None
    for name in ("classify", "predict", "infer"):
        candidate = getattr(mod, name, None)
        if callable(candidate):
            fn = candidate
            break
    if fn is None:
        raise RuntimeError(
            "No usable predict/classify/infer(img) function found in inference.py."
        )

    result = fn(image)

    label = None
    confidence = None

    if isinstance(result, (list, tuple)):
        if len(result) >= 1:
            label = result[0]
        if len(result) >= 2:
            confidence = result[1]
    elif isinstance(result, dict):
        for key in ("label", "category", "pred", "prediction"):
            if key in result:
                label = result[key]
                break
        for key in ("confidence", "score", "prob", "probability"):
            if key in result:
                confidence = result[key]
                break
    else:
        label = result

    label_str = str(label or "")

    cat_key = normalize_category(label_str)
    if cat_key == "unknown":
        cat_key = _infer_category_from_label(label_str)

    raw = label_str.strip().lower()
    if cat_key == "unknown" and raw in CATS:
        cat_key = raw

    return cat_key, confidence, label_str


def _pretty_visual_title(category_key: str, raw_label: str) -> str:
    mapping = {
        "trash": "Trash",
        "mixed": "Mixed Recyclable",
        "cardboard": "Corrugated Cardboard",
    }
    r = (raw_label or "").strip()
    if r.lower() in mapping:
        return mapping[r.lower()]
    if category_key in mapping and (not r):
        return mapping[category_key]
    return r or "Item from photo"


def visual_helper(image, glass, grease, corrugated, plastic12):
    if image is None:
        return "<em>Please upload a photo first.</em>"

    if grease:
        return render_detail_panel(
            "trash", False, "Greasy/food-soiled item (from photo)"
        )
    if glass:
        return render_detail_panel("trash", False, "Glass Item (from photo)")
    if corrugated:
        return render_detail_panel(
            "cardboard", False, "Corrugated Cardboard (from photo)"
        )
    if plastic12:
        return render_detail_panel(
            "mixed", False, "Plastic #1 or 2 (from photo)"
        )

    try:
        cat_key, confidence, raw_label = _run_ml_classifier(image)
    except Exception as e:
        msg = (
            "The Visual Helper's machine-learning model is not fully configured on this device.<br>"
            "Please rely on the Search Assistant tab for the most accurate sorting guidance.<br>"
            f"<span style='font-size:11px;color:#666;'>Error: {str(e)}</span>"
        )
        return (
            "<div style='border:1px solid #ddd;border-radius:12px;padding:16px;background:#fff;"
            "box-shadow:0 2px 6px rgba(0,0,0,0.05);"
            "font-family:Inter,-apple-system,Helvetica,Arial,sans-serif;"
            "color:#222;line-height:1.45;max-width:640px;'>"
            f"{msg}</div>"
        )

    if cat_key == "unknown":
        msg = (
            "The Visual Helper could not confidently classify this item from the photo.<br>"
            "Please try the Search Assistant tab and type the item name directly "
            '(for example, "pizza box", "glass bottle", or "microwave").'
        )
        return (
            "<div style='border:1px solid #ddd;border-radius:12px;padding:16px;background:#fff;"
            "box-shadow:0 2px 6px rgba(0,0,0,0.05);"
            "font-family:Inter,-apple-system,Helvetica,Arial,sans-serif;"
            "color:#222;line-height:1.45;max-width:640px;'>"
            f"{msg}</div>"
        )

    display_name = _pretty_visual_title(cat_key, raw_label)
    return render_detail_panel(cat_key, False, display_name)


# ---------- UI layout ----------

theme = gr.themes.Soft(primary_hue="green")

# Custom gray "pill" label styling (replaces the green Gradio label)
GRAY_LABEL_CSS = r"""
.gray-pill {
  display: inline-block;
  background: #e5e7eb;      /* light gray */
  color: #111827;           /* near-black text */
  padding: 6px 12px;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
  font-weight: 700;
  font-size: 18px;
  margin-bottom: 8px;
}
@media (prefers-color-scheme: dark) {
  .gray-pill {
    background: #374151;    /* dark gray */
    color: #f9fafb;         /* near-white text */
    border-color: #4b5563;
  }
}
"""

with gr.Blocks(
    title="Chesapeake Sorting Assistant",
    theme=theme,
    css=GRAY_LABEL_CSS,
) as demo:
    gr.Markdown("## Chesapeake Sorting Assistant")

    with gr.Tabs():
        # ---- Tab 1: Text Search Assistant ----
        with gr.TabItem("🔎 Search Assistant (Recommended)"):
            gr.Markdown(
                'Type the name of a waste item to search inside a fixed list of Chesapeake materials. '
                'You can type part of a word (like "mic" for microwave), then tap a match from the list.'
            )

            # GRAY pill label (custom)
            gr.HTML("<div class='gray-pill'>Search for an item and click from the list below</div>")

            # SINGLE search box (real label hidden so it won't turn green)
            query_box = gr.Textbox(
                label=None,
                show_label=False,
                placeholder='Start typing… (ex: "mic", "battery", "bottle")'
            )

            # Suggestions list (Radio = no bar, just choices)
            suggestions = gr.Radio(
                choices=[],
                value=None,
                visible=False,
                show_label=False,
                interactive=True,
            )

            result_html = gr.HTML()
            search_btn = gr.Button("Search", variant="primary")

            # Show suggestions only after typing
            query_box.input(
                fn=live_filter_suggestions,
                inputs=query_box,
                outputs=suggestions
            )

            # Picking a suggestion fills box, hides suggestions, shows result
            suggestions.change(
                fn=on_suggestion_pick,
                inputs=suggestions,
                outputs=[query_box, suggestions, result_html]
            )

            # Search button / Enter key
            search_btn.click(
                fn=search_item,
                inputs=query_box,
                outputs=result_html
            )
            query_box.submit(
                fn=search_item,
                inputs=query_box,
                outputs=result_html
            )

        # ---- Tab 2: Visual Helper (Beta) ----
        with gr.TabItem("🖼️ Visual Helper (Beta)"):
            gr.Markdown(
                "To see where your item belongs: Upload a photo and use the checkboxes below if applicable."
            )
            image = gr.Image(label="Upload a photo", type="pil")

            with gr.Row():
                glass_cb = gr.Checkbox(label="Glass Item")
                grease_cb = gr.Checkbox(label="Visible Grease/Food Residue")
                corr_cb = gr.Checkbox(label="Corrugated Cardboard (shipping boxes)")
                plastic_cb = gr.Checkbox(label="Plastic #1 or 2")

            analyze_btn = gr.Button("Analyze Photo", variant="primary")
            ml_result = gr.HTML()

            analyze_btn.click(
                fn=visual_helper,
                inputs=[image, glass_cb, grease_cb, corr_cb, plastic_cb],
                outputs=ml_result,
            )

if __name__ == "__main__":
    demo.launch(share=True)
