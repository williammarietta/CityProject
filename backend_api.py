from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
import os
import json
import re
from html import escape

from PIL import Image

from apppro import (
    search_item,
    visual_helper,
    _scored_matches,
    render_detail_panel,
    ALL_ITEMS,
)

# We’ll import OpenAI safely so the app can still run
# even if the package is not installed yet.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

app = FastAPI()

# Allow your React dev server to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    html: str


class SuggestionsResponse(BaseModel):
    choices: List[str]


class VisualHelperResponse(BaseModel):
    html: str


# --- Joke “Easter eggs” for exact absurd inputs (no API cost) ---
EASTER_EGGS = {
    "love": (
        "Love is 100% reusable—please keep it. 💚",
        "Type a physical item and press Search (e.g., “battery”, “bottle”, “microwave”).",
    ),
    "saturn": (
        "Saturn belongs in space… and is slightly too large for the drop-off bins. 🪐",
        "Type a physical item and press Search (e.g., “battery”, “bottle”, “microwave”).",
    ),
}


def _get_openai_client():
    """
    Returns an OpenAI client if the library and API key are available.
    Otherwise returns None so the app can gracefully fall back.
    """
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _safe_json_loads(text: str) -> Optional[dict]:
    """
    Tries to parse JSON. If the model returns extra text,
    attempts to extract the first {...} block.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _render_fun_card(query: str, joke: str, hint: str) -> str:
    """
    Returns a friendly joke-style HTML card.
    """
    q = escape(query or "")
    j = escape(joke or "")
    h = escape(hint or "")

    return f"""
<div style='border:1px solid #ddd;border-radius:12px;padding:16px;background:#fff;
            box-shadow:0 2px 6px rgba(0,0,0,0.05);
            font-family:Inter,-apple-system,Helvetica,Arial,sans-serif;
            color:#000 !important;line-height:1.45;max-width:640px;'>
  <div style="font-size:18px;font-weight:700;margin-bottom:6px;color:#000 !important;">{q}</div>
  <div style="font-size:14px;color:#111;margin-bottom:10px;">{j}</div>
  <div style="font-size:13px;color:#555;">{h}</div>
</div>
"""


def _llm_non_item_or_classify(query: str) -> Optional[dict]:
    """
    Single OpenAI call that decides:
    - If query is NOT a physical household item -> return {"non_item": True, "joke":..., "hint":...}
    - Else -> return normal classification {"non_item": False, "category":..., "bulk":..., "reason":...}

    Returns None on failure.
    """
    q = (query or "").strip()
    if not q:
        return None

    client = _get_openai_client()
    if client is None:
        return None

    allowed_categories = ["cardboard", "mixed", "trash", "hazard"]

    rules_text = """
You help classify household waste items for Chesapeake, Virginia drop-off recycling centers.

There are ONLY four allowed categories:
- "cardboard": Corrugated shipping/moving boxes ONLY (flattened). Corrugated = has a fluted/wavy inner layer.
- "mixed": plastics #1 or #2 BOTTLES/JUGS, metal cans, mixed paper, and thin paperboard boxes (like cereal/tissue boxes) that are NOT greasy.
- "trash": NOT accepted at these recycling centers, including glass, plastic bags/bagged recyclables, clamshell plastics, greasy/food-soiled paper or cardboard, styrofoam, and general household trash.
- "hazard": household hazardous waste or electronic waste (e-waste) to SPSA: paints, chemicals, gasoline, pesticides, oil, car batteries, TVs, computers, etc.

Some items may also be BULK PICKUP candidates (furniture, large appliances, mattresses, etc.).
Bulk pickup is only true when the item is clearly a large bulky object.

Also: Sometimes users type words that are NOT physical items (e.g., emotions, planets, ideas).
If the query is not a physical household item, mark it as non_item and generate a friendly joke + hint.
"""

    system_prompt = rules_text + """
Return ONLY valid JSON in this exact shape:
{
  "non_item": true/false,
  "category": "cardboard|mixed|trash|hazard",
  "bulk": true/false,
  "reason": "one short sentence",
  "joke": "one friendly sentence (only if non_item is true, else empty string)",
  "hint": "one short helpful sentence with examples like battery/bottle/microwave (only if non_item is true, else empty string)"
}

Rules:
- If non_item is true: set category="trash", bulk=false, and include joke + hint.
- If non_item is false: choose category safely (if unsure, usually "trash"), bulk as appropriate, and set joke/hint to "".
No extra keys. No extra text.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": f'Classify this query: "{q}"'},
            ],
            temperature=0.0,
        )
    except Exception:
        return None

    try:
        content = completion.choices[0].message.content
    except Exception:
        return None

    data = _safe_json_loads(content or "")
    if not isinstance(data, dict):
        return None

    non_item = bool(data.get("non_item", False))
    category = str(data.get("category", "")).strip().lower()
    bulk_flag = bool(data.get("bulk", False))
    reason = str(data.get("reason", "")).strip()
    joke = str(data.get("joke", "")).strip()
    hint = str(data.get("hint", "")).strip()

    # If non-item, we only need joke + hint (category defaults to trash)
    if non_item:
        if not joke:
            joke = "That doesn’t look like a physical item you can drop off. 🙂"
        if not hint:
            hint = "Try a physical item like “battery”, “bottle”, or “microwave”, then press Search."
        return {"non_item": True, "joke": joke, "hint": hint}

    # Normal classification path
    if category not in allowed_categories:
        return None

    return {
        "non_item": False,
        "category": category,
        "bulk": bulk_flag,
        "reason": reason,
    }


def _search_with_fallback(query: str) -> str:
    """
    Main search logic used by the /api/search endpoint.

    1. Try JSON-based search using _scored_matches.
    2. If we get any match at all, reuse search_item(q).
    3. If truly no match, do:
       - Easter egg jokes for a few exact absurd words
       - Otherwise OpenAI decides non-item joke vs real classification
    4. If OpenAI unavailable/fails, return a gentle message.
    """
    q = (query or "").strip()
    if not q:
        return '<em>Start typing an item name, then select a match from the list.</em>'

    # Step 1: Try normal scoring
    matches = _scored_matches(q)
    best = matches[0] if matches else None

    if best is not None:
        return search_item(q)

    # Step 3a: Instant jokes (no API cost)
    egg = EASTER_EGGS.get(q.lower())
    if egg:
        joke, hint = egg
        return _render_fun_card(q, joke, hint)

    # Step 3b: No match -> OpenAI decides joke vs classify
    llm = _llm_non_item_or_classify(q)
    if llm:
        if llm.get("non_item") is True:
            return _render_fun_card(q, llm.get("joke", ""), llm.get("hint", ""))

        cat = llm.get("category")
        if cat:
            bulk_flag = bool(llm.get("bulk", False))
            title = q
            return render_detail_panel(cat, bulk_flag, title)

    # Step 4: OpenAI unavailable or failed
    safe_q = escape(q)
    return (
        f'<em>No result for "{safe_q}". '
        "You can still type any item and press Search. "
        'Try “battery”, “bottle”, or “microwave”.</em>'
    )


def _build_suggestions(query: str) -> List[str]:
    """
    Lightweight suggestion logic for the React search box.

    - Empty query -> no suggestions
    - 1+ character -> list of items that start with the query,
      then items that merely contain the query, up to 75 results.
    """
    q = (query or "").strip().lower()
    if len(q) == 0:
        return []

    starts = [item for item in ALL_ITEMS if item.lower().startswith(q)]
    contains = [item for item in ALL_ITEMS if q in item.lower() and item not in starts]
    choices = (starts + contains)[:75]
    return choices


@app.post("/api/search", response_model=SearchResponse)
async def api_search(body: SearchRequest):
    html = _search_with_fallback(body.query)
    return SearchResponse(html=html)


@app.get("/api/suggestions", response_model=SuggestionsResponse)
async def api_suggestions(q: str = ""):
    choices = _build_suggestions(q)
    return SuggestionsResponse(choices=choices)


@app.post("/api/visual-helper", response_model=VisualHelperResponse)
async def api_visual_helper(
    image: UploadFile = File(None),
    glass: bool = Form(False),
    grease: bool = Form(False),
    corrugated: bool = Form(False),
    plastic12: bool = Form(False),
):
    pil_image = None
    if image is not None:
        data = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pil_image = None

    html = visual_helper(pil_image, glass, grease, corrugated, plastic12)
    return VisualHelperResponse(html=html)
