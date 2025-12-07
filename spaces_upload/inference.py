# inference.py — loads a saved model checkpoint and runs a PIL image through it.
# Supports "resnet18" and "mobilenet_v3_large" checkpoints.

from pathlib import Path
from typing import Tuple, Dict
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

def _build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=None)  # head replaced below
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)  # head replaced below
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported arch: {arch}")

def load_model(ckpt_path: Path):
    """
    Loads a checkpoint saved by our training scripts.
    Returns: (model.eval(), preprocess_transform, idx_to_class_dict)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Architecture and classes
    arch = ckpt.get("arch", "resnet18")
    class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)

    # Rebuild model and load weights
    model = _build_model(arch, num_classes)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Preprocess (use values saved in the checkpoint when available)
    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std  = ckpt.get("std",  [0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, preprocess, idx_to_class

@torch.no_grad()
def predict_pil(img: Image.Image, model, preprocess, idx_to_class) -> Tuple[str, float]:
    """
    Runs a single PIL image through the model.
    Returns: (predicted_label, confidence_float_0to1)
    """
    x = preprocess(img.convert("RGB")).unsqueeze(0)  # (1,C,H,W)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    conf, idx = prob.max(dim=0)
    return idx_to_class[int(idx)], float(conf)
