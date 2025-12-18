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

# ----------------------------
# Load canonical list + keywords
# ----------------------------

MASTER_LIST_PATH = ROOT / "data" / "reference" / "ultimate_master_list.txt"


def _load_keywords():
    if not DATA_PATH.exists():
        return {}
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _load_master_items():
    items = []
    if not MASTER_LIST_PATH.exists():
        return items

    for line in MASTER_LIST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # master list lines look like: "Aluminum cans — R"
        # we just want the item name part before the dash
        # but tolerate variations
        parts = re.split(r"\s+[-—]\s+", line, maxsplit=1)
        name = parts[0].strip()
        if name:
            items.append(name)
    return items


KEYWORDS = _load_keywords()
ALL_ITEMS = sorted(set(_load_master_items() + list(KEYWORDS.keys())))

# records: internal “encyclopedia” entries derived from master list + keywords
records = []
for item in ALL_ITEMS:
    key = item
    # Default values
    cat = "trash"
    bulk = False

    # If we have structured hints in keywords_fallback.json
    if key in KEYWORDS:
        entry = KEYWORDS[key]
        if isinstance(entry, dict):
            cat = entry.get("cat", cat)
            bulk = bool(entry.get("bulk", False))

    records.append(
        {
            "name": key.lower(),
            "item": key,
            "cat": cat,
            "bulk": bulk,
        }
    )


# ----------------------------
# Guidance text
# ----------------------------

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
            "<br><br><b>Bulk pickup option:</b> Call <b>757-382-2489</b> to schedule a <b>FREE</b> bulk pickup, if needed."
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

  <div style="margin:10px 0;padding:10px;border-radius:10px;background:#f5f7fa;border:1px solid #e5e7eb;">
    <div style="font-size:12px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;color:#111;">
      Best disposal option
    </div>
    <div style="margin-top:2px;font-size:16px;font-weight:800;color:#000;">
      {best_line}
    </div>
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
    r = record_name.lower()

    if q == r:
        return 100.0
    if r.startswith(q):
        return 70.0
    if q in r:
        return 40.0

    # token overlap
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    r_tokens = set(re.findall(r"[a-z0-9]+", r))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & r_tokens) / max(1, len(q_tokens))
    return overlap * 25.0


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


def search_item(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "<em>Please enter an item.</em>"

    match = best_match(q)
    if not match:
        return "<em>No strong match found.</em>"

    cat = match.get("cat", "trash")
    bulk = bool(match.get("bulk", False))
    title = match.get("item") or q
    return render_detail_panel(cat, bulk, title)


# ---------- Visual Helper (Beta) ----------

MODEL_META_PATH = ROOT / "model_meta.json"


def _run_ml_classifier(image):
    """
    Loads & runs your trained model checkpoint if available.
    Returns (category_key, confidence, raw_label).
    """
    if not MODEL_META_PATH.exists():
        raise FileNotFoundError(f"Missing model_meta.json at {MODEL_META_PATH}")

    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    checkpoint_path = ROOT / meta["checkpoint_path"]
    labels_path = ROOT / meta["labels_path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint at {checkpoint_path}. Run train_finetune.py first to create it."
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file at {labels_path}")

    # Lazy import to avoid crashing when torch isn't installed
    spec = importlib.util.spec_from_file_location("inference", str(ROOT / "inference.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.run_inference(image, meta, checkpoint_path, labels_path)


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
            f"Error: {e}"
        )
        return f"<div style='color:#000 !important;'><b>{msg}</b></div>"

    # Confidence hint
    if confidence is not None and confidence < 0.55:
        hint = (
            "<div style='margin-top:8px;font-size:12px;color:#555;'>"
            "Low confidence: please double-check and consider using the Search Assistant.</div>"
        )
    else:
        hint = ""

    panel = render_detail_panel(cat_key, False, f"{raw_label} (from photo)")
    return panel + hint
