# inference.py — REAL Visual Helper inference (MobileNetV3-Large, 3 classes)
#
# What this file does:
# - Loads your fine-tuned Chesapeake model: models/recycler_chesapeake.pt
# - Loads class order from: models/recycler_chesapeake_labels.json
# - Builds the matching MobileNetV3-Large architecture
# - Runs a forward pass on a PIL image
# - Returns (label, confidence)
#
# Visual Helper in app.py/apppro.py will call predict(img) / classify(img) / infer(img).

import json
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn
from torchvision import models, transforms

# ----- Paths -----
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "recycler_chesapeake.pt"
LABELS_PATH = ROOT / "models" / "recycler_chesapeake_labels.json"

# ----- Device -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Cache -----
_MODEL = None
_LABELS = None

# ----- Preprocess -----
# IMPORTANT: match training transforms
# Your training used Resize((224,224)) + ToTensor() (no normalization).
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def _load_labels():
    global _LABELS
    if _LABELS is not None:
        return _LABELS

    if not LABELS_PATH.exists():
        # Fallback to expected order if labels file is missing
        _LABELS = ["cardboard", "mixed", "trash"]
        return _LABELS

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        _LABELS = json.load(f)

    return _LABELS

def _build_model(num_classes: int):
    """
    Build the same architecture that your checkpoint was trained on:
    MobileNetV3-Large with a 3-class head.
    """
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    labels = _load_labels()
    num_classes = len(labels)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint at {MODEL_PATH}. "
            f"Run train_finetune.py first to create recycler_chesapeake.pt."
        )

    model = _build_model(num_classes=num_classes)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    # This checkpoint is saved as state_dict only
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(DEVICE)

    _MODEL = model
    return _MODEL

def predict(img) -> Tuple[str, float]:
    """
    Main inference function.
    Input: PIL image (from Gradio Image type="pil")
    Output: (label, confidence)
            label is one of: "cardboard", "mixed", "trash"
    """
    if img is None:
        return ("trash", 0.0)

    model = _load_model()
    labels = _load_labels()

    # Preprocess
    x = _preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    best_idx = int(probs.argmax())
    best_label = str(labels[best_idx])
    best_conf = float(probs[best_idx])

    return best_label, best_conf

# Provide aliases so your app can find a usable function
def classify(img):
    return predict(img)

def infer(img):
    return predict(img)
