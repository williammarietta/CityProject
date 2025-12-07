# 05_evaluate.py
# Evaluates the saved model on data/val:
# - overall accuracy
# - per-class accuracy
# - confusion_matrix.png
# - predictions.csv (path, true, pred, correct, confidence)
from pathlib import Path
import csv, json
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from inference import load_model, predict_pil

ROOT = Path(__file__).resolve().parent
VAL_DIR = ROOT / "data" / "val"
CKPT = ROOT / "models" / "recycler_v1.pt"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

# safety checks
if not CKPT.exists():
    raise SystemExit("❌ models/recycler_v1.pt not found. Run: python train.py")
if not VAL_DIR.exists():
    raise SystemExit("❌ data/val not found. Finish Day 2 split first.")

# load model & class mapping
model, preprocess, idx_to_class = load_model(CKPT)
classes = [idx_to_class[i] for i in range(len(idx_to_class))]  # in index order

# collect all val images with ground-truth labels
samples = []
for ci, cname in enumerate(classes):
    cdir = VAL_DIR / cname
    if not cdir.exists():
        print(f"⚠️ missing class folder in val: {cname}")
        continue
    for p in cdir.glob("*"):
        if p.is_file():
            samples.append((str(p), cname))

if not samples:
    raise SystemExit("❌ No validation images found under data/val/*")

# run predictions
y_true, y_pred, y_conf = [], [], []
for path, true_lab in samples:
    img = Image.open(path).convert("RGB")
    pred_lab, conf = predict_pil(img, model, preprocess, idx_to_class)
    y_true.append(true_lab)
    y_pred.append(pred_lab)
    y_conf.append(conf)

# overall accuracy
acc = sum(int(t==p) for t,p in zip(y_true, y_pred)) / len(y_true)
print(f"\nOverall val accuracy: {acc:.3f}  (N={len(y_true)})")

# per-class accuracy
per_class = {}
for c in classes:
    idxs = [i for i, t in enumerate(y_true) if t == c]
    correct = sum(1 for i in idxs if y_pred[i] == c)
    per_class[c] = (correct, len(idxs), correct / max(1,len(idxs)))
print("\nPer-class accuracy:")
for c,(k,n,pc) in per_class.items():
    print(f"  {c:9s}: {pc:.3f}  ({k}/{n})")

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(5.5,5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
ax.set_yticks(range(len(classes)), classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
cm_path = OUT_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {cm_path}")

# classification report (precision/recall/F1)
report = classification_report(y_true, y_pred, labels=classes, digits=3, zero_division=0)
(OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
print("Saved:", OUT_DIR / "classification_report.txt")

# CSV with all predictions
csv_path = OUT_DIR / "predictions.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path","true","pred","correct","confidence"])
    for path, t, p, c in zip((s[0] for s in samples), y_true, y_pred, y_conf):
        w.writerow([path, t, p, int(t==p), f"{c:.4f}"])
print("Saved:", csv_path)
