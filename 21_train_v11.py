# 21_train_v11.py — a stronger training recipe (fixed class count)
# - ResNet18 pretrained
# - Warmup: train last linear layer (2 epochs)
# - Fine-tune: unfreeze layer4 (last block) with a smaller LR (8 more epochs)
# - Class-weighted loss (helps if classes are imbalanced)
# - Early stopping (patience=3)

from pathlib import Path
import json, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR   = ROOT / "data" / "val"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

for sub in ["mixed","cardboard","trash"]:
    if not (TRAIN_DIR/sub).exists() or not (VAL_DIR/sub).exists():
        raise SystemExit(f"Missing class folder: {sub}. Run Day 2 split.")

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    transforms.RandomPerspective(0.1),
    transforms.RandomAdjustSharpness(1.5, p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.25,0.25,0.25,0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cpu")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# freeze all
for p in model.parameters():
    p.requires_grad = False

# replace head
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# ---------- FIXED: count files per class safely ----------
def count_files(dirpath: Path) -> int:
    return sum(1 for p in dirpath.glob("*") if p.is_file())

# Use train_ds.classes (e.g., ['cardboard','mixed','trash'] in alpha order)
counts = np.array([count_files(TRAIN_DIR / c) for c in train_ds.classes], dtype=float)

# class weights (inverse frequency; avoid divide-by-zero)
inv = 1.0 / np.clip(counts, 1, None)
class_w = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_w)

def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / max(1,total)

best_acc = 0.0
patience = 3
no_improve = 0

# Phase 1: train only final fc
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
E1 = 2
print(f"Phase 1: training fc for {E1} epochs…")
for epoch in range(1, E1+1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    val_acc = evaluate()
    print(f"  [warmup {epoch}/{E1}] val_acc={val_acc:.3f}")

# Phase 2: unfreeze layer4 (last block) and fine-tune
for p in model.layer4.parameters():
    p.requires_grad = True
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 5e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},
])
E2 = 8
print(f"Phase 2: fine-tuning layer4 for {E2} epochs…")

for epoch in range(1, E2+1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    val_acc = evaluate()
    improved = val_acc > best_acc
    print(f"  [fine {epoch}/{E2}] val_acc={val_acc:.3f}  {'(improved)' if improved else ''}")

    if improved:
        best_acc = val_acc
        ckpt = {
            "arch": "resnet18",
            "state_dict": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "img_size": IMG_SIZE, "mean": MEAN, "std": STD
        }
        out = MODELS_DIR / "recycler_v1_1.pt"
        torch.save(ckpt, out)
        with open(MODELS_DIR / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump({"best_val_acc": best_acc, "classes": train_ds.classes}, f, indent=2)
        print(f"  ✅ Saved improved model → {out}")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("  ⏹ Early stopping (no improvement).")
            break

print(f"Best val_acc={best_acc:.3f}")
