import torch
from pathlib import Path

CKPTS = [
    Path("models/recycler_v1.pt"),
    Path("models/recycler_balanced.pt"),
]

def infer_num_classes(state_dict):
    candidates = []
    for k, v in state_dict.items():
        if k.endswith(("fc.weight", "fc.bias", "classifier.weight", "classifier.bias")):
            if hasattr(v, "shape") and len(v.shape) > 0:
                candidates.append((k, v.shape))
    return candidates

for ckpt_path in CKPTS:
    print("\n===", ckpt_path, "===")
    if not ckpt_path.exists():
        print("Missing.")
        continue

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Handle different save formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    else:
        print("Unknown checkpoint structure. Keys:", list(ckpt.keys())[:10])
        continue

    shapes = infer_num_classes(sd)
    if not shapes:
        print("Could not find final-layer weights automatically.")
    else:
        print("Possible final layers:")
        for k, s in shapes:
            print(" ", k, "->", s)
            # Usually first dim = number of classes
            print("   inferred classes =", s[0])
