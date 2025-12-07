"""
inference.py — minimal, dependency-free stub so Visual Helper always responds.

- Exposes predict(image), classify(image), infer(image)
- Accepts a PIL.Image.Image or a file path string
- Returns (label, confidence) using a simple color heuristic:
  * brownish → Corrugated Cardboard (Container 2)
  * very bright white/gray → Mixed Recyclables (Container 1) (paper-ish)
  * otherwise → Mixed Recyclables (Container 1) with low confidence

Later, you can replace the body of predict() with your real PyTorch model.
"""

from pathlib import Path
from typing import Tuple, Union

try:
    from PIL import Image
except ImportError as e:
    raise RuntimeError("Pillow is required. Install with: pip install pillow") from e

# Official labels expected by the app
HAZ  = "Hazardous Waste / E-Waste"
CARD = "Corrugated Cardboard (Container 2)"
MIX  = "Mixed Recyclables (Container 1)"
TRASH= "Trash / Not Accepted"

ImageLike = Union[Image.Image, str, Path]

def _as_pil(img: ImageLike) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    p = Path(img)
    if not p.exists():
        raise FileNotFoundError(f"Image path does not exist: {p}")
    return Image.open(p)

def _avg_rgb(im: Image.Image) -> Tuple[float, float, float]:
    im = im.convert("RGB").resize((64, 64))
    data = list(im.getdata())
    n = len(data)
    r = sum(px[0] for px in data) / n
    g = sum(px[1] for px in data) / n
    b = sum(px[2] for px in data) / n
    return r, g, b

def _brightness_variance(im: Image.Image) -> float:
    # quick rough variance of luminance (0..255)
    im = im.convert("L").resize((64, 64))
    vals = list(im.getdata())
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean)**2 for v in vals) / n
    return var

def predict(image: ImageLike):
    """
    Minimal heuristic predictor returning (label, confidence).
    This is just to make the Visual Helper usable immediately.
    Replace this with your real model later.
    """
    try:
        im = _as_pil(image)
    except Exception:
        # If we can't read the image, return a safe fallback
        return (TRASH, 0.01)

    try:
        r, g, b = _avg_rgb(im)
        var_l   = _brightness_variance(im)

        # Heuristic 1: "brownish" — often corrugated boxes
        #   r noticeably higher than g/b; not too bright
        if (r > g + 15) and (r > b + 15) and (r < 200) and (g < 190) and (b < 190):
            return (CARD, 0.55)

        # Heuristic 2: bright, low-variance grayscale — likely plain paper/paperboard
        if (r > 200 and g > 200 and b > 200) and (var_l < 500):
            return (MIX, 0.45)

        # Default: Mixed (weak confidence) — safe guess for Visual Helper
        return (MIX, 0.20)
    except Exception:
        # Any unexpected error: fail soft
        return (TRASH, 0.01)

# Aliases the app will also accept
def classify(image: ImageLike):
    return predict(image)

def infer(image: ImageLike):
    return predict(image)
