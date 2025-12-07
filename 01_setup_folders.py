from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = ROOT / "data"

# Where you will first drop photos
incoming = data / "incoming"
# Final training folders
train = data / "train"
val = data / "val"

classes = ["mixed", "cardboard", "trash"]

for base in [incoming, train, val]:
    for c in classes:
        (base / c).mkdir(parents=True, exist_ok=True)

print("✅ Folders ready:")
print(f"- {incoming}\\(mixed|cardboard|trash)")
print(f"- {train}\\(mixed|cardboard|trash)")
print(f"- {val}\\(mixed|cardboard|trash)")
