# 31_train_balanced.py — balanced training with MobileNetV3-Large
from pathlib import Path
import json, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
from time import time

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR   = ROOT / "data" / "val"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CLASSES = ["cardboard","mixed","trash"]  # alphabetical is fine; ImageFolder sets order by name
for c in CLASSES:
    if not (TRAIN_DIR/c).exists() or not (VAL_DIR/c).exists():
        raise SystemExit(f"❌ Missing folder for class: {c}. Run your Day-2 split.")

# Image size a bit larger helps texture (corrugation) & bottles/cans edges
IMG_SIZE = 320
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.2)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3,0.3,0.3,0.15),
    transforms.RandomPerspective(0.1),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomAdjustSharpness(1.5, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
])

val_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.2)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tfms)

# ---------- Balanced sampling (so bottles/cans aren't underseen) ----------
targets = np.array(train_ds.targets)
class_counts = np.bincount(targets, minlength=len(train_ds.classes))
class_weights = 1.0 / np.clip(class_counts, 1, None)
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cpu")

# ---------- Model: MobileNetV3-Large ----------
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
in_f = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_f, len(train_ds.classes))
model = model.to(device)

# ---------- Loss: class-weighted CE with label smoothing ----------
# (helps with label noise from public datasets)
counts = np.array([max(1, class_counts[i]) for i in range(len(train_ds.classes))], dtype=float)
inv = 1.0 / counts
ce_weights = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=0.1)

def evaluate() -> float:
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            all_t.extend(y.cpu().tolist())
            all_p.extend(pred.cpu().tolist())
    return accuracy_score(all_t, all_p)

best_acc = 0.0
patience = 4
no_improve = 0

# ---------- Phase 1: train head only ----------
optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=1e-3, weight_decay=1e-4)
E1 = 3
print(f"Phase 1: training classifier head for {E1} epochs")
for ep in range(1, E1+1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    val_acc = evaluate()
    print(f"  [warmup {ep}/{E1}] val_acc={val_acc:.3f}")

# ---------- Phase 2: unfreeze last feature block + head ----------
for p in model.features[-1].parameters():
    p.requires_grad = True
optimizer = torch.optim.AdamW([
    {"params": model.features[-1].parameters(), "lr": 5e-4, "weight_decay": 1e-4},
    {"params": model.classifier[-1].parameters(), "lr": 1e-3, "weight_decay": 1e-4},
])
E2 = 12
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=E2)

print(f"Phase 2: fine-tuning for up to {E2} epochs (early stopping)")
for ep in range(1, E2+1):
    t0 = time()
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    val_acc = evaluate()
    dt = time()-t0
    improved = val_acc > best_acc
    print(f"  [fine {ep}/{E2}] val_acc={val_acc:.3f}  {'(improved)' if improved else ''}  {dt:.1f}s")

    if improved:
        best_acc = val_acc
        ckpt = {
            "arch": "mobilenet_v3_large",
            "state_dict": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "img_size": IMG_SIZE, "mean": MEAN, "std": STD
        }
        out = MODELS_DIR / "recycler_balanced.pt"
        torch.save(ckpt, out)
        with open(MODELS_DIR / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump({"best_val_acc": best_acc, "classes": train_ds.classes}, f, indent=2)
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("  ⏹ Early stopping (no improvement).")
            break

print(f"Best val_acc={best_acc:.3f}")
print("Saved model →", MODELS_DIR / "recycler_balanced.pt")
