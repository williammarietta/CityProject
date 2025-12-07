# train_finetune.py
# Fine-tune your EXISTING 3-class MobileNet model on Chesapeake 3-bin folders.
# Uses your old checkpoint as the starting point (no restart).
# Saves a new model: models/recycler_chesapeake.pt

from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR   = ROOT / "data" / "val"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ✅ Pick which old checkpoint to start from:
START_CKPT = MODELS_DIR / "recycler_balanced.pt"
# START_CKPT = MODELS_DIR / "recycler_v1.pt"

# ✅ Output checkpoint name:
OUT_CKPT = MODELS_DIR / "recycler_chesapeake.pt"
OUT_LABELS = MODELS_DIR / "recycler_chesapeake_labels.json"

# ✅ Training settings (beginner-friendly defaults)
EPOCHS = 6
BATCH_SIZE = 24
LR = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 0  # Windows-safe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(num_classes=3):
    """
    Your checkpoint matches MobileNetV3-Large style keys ("features.*").
    So we build MobileNetV3-Large and set the final layer to 3 classes.
    """
    model = models.mobilenet_v3_large(weights=None)
    # classifier[3] is the final Linear layer in torchvision MobileNetV3-Large
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Checkpoint format not recognized.")

    # If the architecture matches, strict=True will work.
    # If it doesn't match perfectly, we fall back to strict=False.
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Loaded checkpoint with strict=True")
    except RuntimeError as e:
        print("⚠️ strict=True failed, trying strict=False")
        print("   (This is okay if only small naming differences exist.)")
        model.load_state_dict(state_dict, strict=False)

    return model


def main():
    print("Device:", DEVICE)
    print("Train dir:", TRAIN_DIR)
    print("Val dir:", VAL_DIR)
    print("Start ckpt:", START_CKPT)

    # Data transforms (helps with real phone photos)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    class_names = train_ds.classes  # alphabetical by folder name
    print("Class order:", class_names)

    # Save class order for inference later
    with open(OUT_LABELS, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(num_classes=len(class_names))
    model = load_checkpoint(model, START_CKPT)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.3f}")
        print(f"Val acc:   {val_acc:.3f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_CKPT)
            print("✅ Saved new best model to", OUT_CKPT)

    print("\nDone.")
    print("Best val acc:", best_val_acc)
    print("Labels saved to:", OUT_LABELS)


if __name__ == "__main__":
    main()
