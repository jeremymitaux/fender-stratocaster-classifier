"""Classify a guitar listing image into one of the trained make/model classes.

The class list is read from ``models/class_names.json`` (written at train time),
so this works for any trained class set (e.g. stratocaster / telecaster).

Usage::

    strat-predict path/to/image.jpg [more.jpg ...]
    python -m strat_classifier.inference path/to/image.jpg
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from strat_classifier.data import build_transforms
from strat_classifier.model import build_model

MODEL_DIR = Path("models")

# Reuse the package's deterministic eval transform so inference preprocessing
# (resize + ImageNet normalization) can never silently drift from what the
# model saw at train time.
_, TRANSFORM = build_transforms()


def load_model(model_dir: Path = MODEL_DIR):
    """Load the trained checkpoint and its class-name list."""
    class_names = json.loads((model_dir / "class_names.json").read_text())
    model = build_model(num_classes=len(class_names), pretrained=False)
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, class_names


@torch.no_grad()
def predict(model: nn.Module, class_names: list, image_path: str) -> dict:
    """Return prediction, confidence, and per-class probabilities for one image."""
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    logits = model(tensor)[0]
    probs = torch.softmax(logits, dim=0)
    top_idx = probs.argmax().item()
    return {
        "prediction": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "scores": {c: float(probs[i]) for i, c in enumerate(class_names)},
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Usage: strat-predict <image_path> [<image_path> ...]")
        sys.exit(1)

    model, class_names = load_model(MODEL_DIR)
    print(f"Model loaded  |  classes: {class_names}\n")

    for path in argv:
        result = predict(model, class_names, path)
        print(f"{path}")
        print(f"  Prediction : {result['prediction'].upper()}  ({result['confidence']:.1%})")
        for cls, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"  {cls:<10} {score:.1%}  {bar}")
        print()


if __name__ == "__main__":
    main()
