# train.py — trains a small image classifier on data/train and data/val
# Classes must be folders: data/train/{mixed,cardboard,trash}, data/val/{...}

from pathlib import Path
import json, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR   = ROOT / "data" / "val"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 1) Safety checks
if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise SystemExit("data/train or data/val not found. Finish Day 2 first.")
for sub in ["mixed","cardboard","trash"]:
    if not (TRAIN_DIR/sub).exists() or not (VAL_DIR/sub).exists():
        raise SystemExit(f"Missing class folder: {sub}. Expected in train/ and val/")

# 2) Transforms (ImageNet normalization because we use a pretrained ResNet18)
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
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

# 3) Model: ResNet18 pretrained, train only the last layer (fast & stable)
device = torch.device("cpu")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / max(1,total)

EPOCHS = 5   # keep small for CPU; you can raise later
best_acc = 0.0
start = time.time()

for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
    train_loss = running / max(1, len(train_ds))
    val_acc = evaluate()
    print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        ckpt = {
            "arch": "resnet18",
            "state_dict": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,   # e.g., {'cardboard':0,'mixed':1,'trash':2} (alphabetical)
            "img_size": IMG_SIZE,
            "mean": MEAN, "std": STD
        }
        torch.save(ckpt, MODELS_DIR / "recycler_v1.pt")
        with open(MODELS_DIR / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump({"best_val_acc": best_acc, "classes": train_ds.classes}, f, indent=2)
        print(f"  ✅ Saved best model (val_acc={best_acc:.3f}) → models/recycler_v1.pt")

elapsed = time.time() - start
print(f"Done in {elapsed/60:.1f} min. Best val_acc={best_acc:.3f}")
