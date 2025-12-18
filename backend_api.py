from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
import os
import json

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

# Allow your Netlify frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for now; can tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    html: str
    used_llm: bool = False


class SuggestResponse(BaseModel):
    suggestions: List[str]


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


def _llm_fallback_classify(query: str) -> Optional[dict]:
    """
    Ask the ChatGPT model to classify an arbitrary item description
    into one of our Chesapeake categories OR a special "not_a_waste_item" category.

    Returns:
        dict like {"category": "trash", "bulk": False, "reason": "..."}
        or None if anything goes wrong.
    """
    q = (query or "").strip()
    if not q:
        return None

    client = _get_openai_client()
    if client is None:
        # Library not installed or API key missing
        return None

    allowed_categories = ["cardboard", "mixed", "trash", "hazard", "not_a_waste_item"]

    rules_text = """
You are helping classify household waste items for Chesapeake, Virginia drop-off recycling centers.

You must classify the user's input into exactly one category from this list:

- "cardboard": Corrugated shipping/moving boxes ONLY (flattened). Corrugated = has a fluted/wavy inner layer.
- "mixed": Mixed recyclables accepted at Chesapeake drop-off centers. This includes plastics #1 or #2 BOTTLES or JUGS (with necks), metal cans (aluminum, tin, or steel), mixed paper, and clean paperboard boxes (like cereal or tissue boxes) that are not greasy.
- "trash": Items that are NOT accepted in these recycling containers, including glass, plastic bags or bagged recyclables, clamshell plastics, food- or grease-soiled paper/cardboard (like greasy pizza boxes), styrofoam, and general household trash.
- "hazard": Household hazardous waste or electronic waste (e-waste), such as chemicals, paint, gasoline, pesticides, oil, car batteries, TVs, computers, etc. These do NOT go in the recycling containers and must go to proper HHW/e-waste programs.
- "not_a_waste_item": Use this ONLY when the text clearly does NOT describe a physical object someone could place in trash or recycling (for example, emotions, ideas, planets, school grades, or other abstract concepts).

Some items may also be BULK PICKUP candidates (furniture, large appliances, mattresses, etc.) if they are a large household item that would not fit into a normal trash cart.

CRITICAL INSTRUCTIONS:
- ALWAYS output one of these categories: "cardboard", "mixed", "trash", "hazard", or "not_a_waste_item".
- Use "not_a_waste_item" when the input is not a concrete physical item.
- DO NOT invent new categories.
- When you are unsure between "mixed" and "trash", choose "trash" to avoid contaminating recycling.
"""

    system_prompt = rules_text + """
You MUST respond ONLY with a valid JSON object of this form:
{"category": "...", "bulk": true/false, "reason": "short explanation"}

- "category" must be exactly one of: "cardboard", "mixed", "trash", "hazard", or "not_a_waste_item".
- "bulk" is true only if the item is clearly a large bulky object.
- "reason" should be a single short sentence.
Do not include any extra keys or text.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": f'Classify this item: "{q}"'},
            ],
            temperature=0.0,
        )
    except Exception:
        return None

    try:
        content = completion.choices[0].message.content.strip()
        data = json.loads(content)
    except Exception:
        return None

    category = str(data.get("category", "")).strip().lower()
    bulk_flag = bool(data.get("bulk", False))
    reason = str(data.get("reason", "")).strip()

    if category not in allowed_categories:
        return None

    return {"category": category, "bulk": bulk_flag, "reason": reason}


def _render_not_a_waste_item_panel(query: str, reason: str) -> str:
    """
    Render a friendly message when the LLM thinks the text is not
    a real waste item (e.g., an emotion, idea, planet, etc.).
    """
    q = (query or "").strip() or "that"
    extra = (reason or "").strip()

    extra_html = (
        f'<div style="margin-top:8px;font-size:13px;color:#555;">Model note: {extra}</div>'
        if extra
        else ""
    )

    return f"""
<div style='border:1px solid #ddd;border-radius:12px;padding:16px;background:#fff;
            box-shadow:0 2px 6px rgba(0,0,0,0.05);
            font-family:Inter,-apple-system,Helvetica,Arial,sans-serif;
            color:#000 !important;line-height:1.45;max-width:640px;'>
  <div style="font-size:18px;font-weight:600;margin-bottom:4px;color:#000 !important;">
    Not a waste item
  </div>
  <div style="font-size:13px;color:#555;margin-bottom:6px;">
    City of Chesapeake guidance · Search Assistant
  </div>

  <div style="margin:10px 0;padding:10px;border-radius:10px;background:#f5f7fa;border:1px solid #e5e7eb;">
    <div style="font-size:12px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;color:#111;">
      Best next step
    </div>
    <div style="margin-top:2px;font-size:16px;font-weight:800;color:#000;">
      Try a physical item
    </div>
  </div>

  <div style="margin-top:8px;font-size:14px;color:#222;">
    "{q}" isn't something you can toss in a bin. Try entering a specific physical item
    you'd throw away or recycle, like "pizza box", "plastic bottle", or "cardboard box".
  </div>

  {extra_html}

  <div style="margin-top:12px;font-size:11px;color:#777;">
    This information is intended for City of Chesapeake, VA recycling drop-off locations.
    Always follow on-site signage and staff instructions.
  </div>
</div>
""".strip()


def _search_with_fallback(q: str) -> tuple[str, bool]:
    """
    Returns (html, used_llm).
    - First tries exact/strong matching using existing logic.
    - If weak/no match, tries LLM fallback.
    """
    q = (q or "").strip()
    if not q:
        return "<em>Please enter an item.</em>", False

    # Step 1: Let existing logic handle exact matches / strong matches
    html = search_item(q)
    if html and "No strong match found" not in html:
        return html, False

    # Step 2: Check scored matches, and only proceed if weak
    matches = _scored_matches(q)
    if matches:
        top = matches[0]
        # If we have a decent match, prefer the local encyclopedia logic.
        # (Your scoring function already prefers exact/startswith.)
        if top.get("name", "").lower().startswith(q.lower()) or top.get("name", "").lower() == q.lower():
            cat = top.get("cat")
            bulk_flag = bool(top.get("bulk", False))
            item_title = top.get("item") or q
            return render_detail_panel(cat, bulk_flag, item_title), False

    # Step 3: No match at all -> try LLM fallback
    llm_result = _llm_fallback_classify(q)
    if llm_result and llm_result.get("category"):
        cat = llm_result["category"]
        bulk_flag = bool(llm_result.get("bulk", False))
        title = q  # use the user's phrase as the "item name" on the card

        if cat == "not_a_waste_item":
            # Friendly "try again" message instead of defaulting to Trash
            return _render_not_a_waste_item_panel(q, llm_result.get("reason", "")), True

        return render_detail_panel(cat, bulk_flag, title), True

    # Step 4: Absolute fallback
    return render_detail_panel("trash", False, q), False


@app.post("/api/search", response_model=SearchResponse)
def api_search(req: SearchRequest):
    html, used_llm = _search_with_fallback(req.query)
    return SearchResponse(html=html, used_llm=used_llm)


@app.get("/api/suggestions", response_model=SuggestResponse)
def api_suggestions(q: str = ""):
    q = (q or "").strip().lower()
    if not q:
        return SuggestResponse(suggestions=[])

    # show suggestions from ALL_ITEMS (from apppro.py)
    results = []
    for item in ALL_ITEMS:
        if item.lower().startswith(q):
            results.append(item)
        if len(results) >= 12:
            break

    return SuggestResponse(suggestions=results)


@app.post("/api/visual-helper")
async def api_visual_helper(
    file: UploadFile = File(...),
    glass: bool = Form(False),
    grease: bool = Form(False),
    corrugated: bool = Form(False),
    plastic12: bool = Form(False),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    html = visual_helper(image, glass, grease, corrugated, plastic12)
    return {"html": html}
