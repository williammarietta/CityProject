import torch
from pathlib import Path

CKPTS = [
    Path("models/recycler_v1.pt"),
    Path("models/recycler_balanced.pt"),
]

def show_tensor_keys(sd):
    keys = list(sd.keys())
    print(f"Total tensor keys: {len(keys)}")
    print("First 25 keys:")
    for k in keys[:25]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
        print(" ", k, "->", shape)

    print("\nLast 25 keys:")
    for k in keys[-25:]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
        print(" ", k, "->", shape)

    # Try to guess classifier-like weights:
    candidates = []
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        if len(v.shape) == 2:  # weight matrices
            out_dim, in_dim = v.shape
            if out_dim <= 12:   # likely a small classifier head
                candidates.append((k, v.shape))
    if candidates:
        print("\nPossible final-layer candidates (2D weights with small out_dim):")
        for k, s in candidates[:30]:
            print(" ", k, "->", s, "(out_dim =", s[0], ")")
    else:
        print("\nNo obvious small classifier weights found.")

for ckpt_path in CKPTS:
    print("\n==============================")
    print("Checkpoint:", ckpt_path)
    if not ckpt_path.exists():
        print("❌ Missing.")
        continue

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Loaded type:", type(ckpt))

    if isinstance(ckpt, torch.nn.Module):
        print("This checkpoint is a full nn.Module.")
        sd = ckpt.state_dict()
        show_tensor_keys(sd)
        continue

    if not isinstance(ckpt, dict):
        print("Unknown checkpoint format (not dict/module).")
        continue

    print("Top-level keys:", list(ckpt.keys())[:20])

    # Try common containers
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        print("Using ckpt['state_dict']")
        sd = ckpt["state_dict"]
    elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        print("Using ckpt['model_state']")
        sd = ckpt["model_state"]
    elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        print("Using ckpt directly as state_dict")
        sd = ckpt
    else:
        # Last resort: find the biggest dict of tensors inside
        sd = None
        for k, v in ckpt.items():
            if isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
                sd = v
                print(f"Using nested tensor dict at ckpt['{k}']")
                break
        if sd is None:
            print("Could not locate a tensor state_dict inside this checkpoint.")
            continue

    show_tensor_keys(sd)
