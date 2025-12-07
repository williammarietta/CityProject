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
    into one of our 4 categories + bulk flag.

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

    # Only allow these exact category strings
    allowed_categories = ["cardboard", "mixed", "trash", "hazard"]

    rules_text = """
You are helping classify household waste items for Chesapeake, Virginia drop-off recycling centers.

There are ONLY four allowed categories:
- "cardboard": Corrugated shipping/moving boxes ONLY (flattened). Corrugated = has a fluted/wavy inner layer.
- "mixed": Mixed recyclables: plastics #1 or #2 BOTTLES/JUGS, metal cans, mixed paper, and thin paperboard boxes (like cereal/tissue boxes) that are NOT greasy.
- "trash": Items that are NOT accepted at these recycling centers, including glass, plastic bags/bagged recyclables, clamshell plastics, greasy/food-soiled paper or cardboard, styrofoam, and general household trash.
- "hazard": Household hazardous waste or electronic waste (e-waste) that must go to a special SPSA facility: paints, chemicals, gasoline, pesticides, oil, car batteries, TVs, computers, etc.

Some items may also be BULK PICKUP candidates (furniture, large appliances, mattresses, etc.). Bulk pickup is only true when the item is clearly a large household item that would not fit into a normal trash cart.

CRITICAL INSTRUCTIONS:
- ALWAYS output one of these categories: "cardboard", "mixed", "trash", or "hazard".
- DO NOT invent new categories.
- If unsure, choose the safest option for the recycling center contamination: usually "trash".
"""

    # Ask for a tiny JSON object we can parse easily.
    system_prompt = rules_text + """
You MUST respond ONLY with a valid JSON object of this form:
{"category": "...", "bulk": true/false, "reason": "short explanation"}

- "category" must be exactly one of: "cardboard", "mixed", "trash", "hazard".
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
        content = completion.choices[0].message.content
    except Exception:
        return None

    if not content:
        return None

    try:
        data = json.loads(content)
    except Exception:
        return None

    category = str(data.get("category", "")).strip().lower()
    bulk_flag = bool(data.get("bulk", False))
    reason = str(data.get("reason", "")).strip()

    if category not in allowed_categories:
        return None

    return {
        "category": category,
        "bulk": bulk_flag,
        "reason": reason,
    }


def _search_with_fallback(query: str) -> str:
    """
    Main search logic used by the /api/search endpoint.

    1. Try the existing JSON-based search using _scored_matches.
    2. If we get any match at all, keep your current JSON-based behavior.
    3. If there is truly no match, call the LLM fallback to classify.
    4. If LLM is unavailable or fails, return a gentle 'no result' message.
    """
    q = (query or "").strip()
    if not q:
        return '<em>Start typing an item name, then select a match from the list.</em>'

    # Step 1: Try normal scoring
    matches = _scored_matches(q)
    best = matches[0] if matches else None

    # Preferred behavior for now:
    # If we found ANY match via your existing logic, just reuse search_item(q).
    # This keeps your current behavior exactly the same for known items.
    if best is not None:
        return search_item(q)

    # Step 3: No match at all -> try LLM fallback
    llm_result = _llm_fallback_classify(q)
    if llm_result and llm_result.get("category"):
        cat = llm_result["category"]
        bulk_flag = bool(llm_result.get("bulk", False))
        title = q  # use the user's phrase as the "item name" on the card
        return render_detail_panel(cat, bulk_flag, title)

    # Step 4: LLM unavailable or failed -> safe fallback
    return (
        f'<em>No result for "{q}". '
        "Try another word (for example, a simpler item name), "
        "or ask a recycling center attendant for help.</em>"
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
