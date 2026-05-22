"""
Classify a Fender Stratocaster image as american / mexican / japanese.

Usage:
    python inference.py path/to/image.jpg
    python inference.py path/to/image.jpg path/to/another.jpg
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image

MODEL_DIR = Path("models")


def load_model(model_dir: Path):
    class_names = json.loads((model_dir / "class_names.json").read_text())
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, class_names


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(model: nn.Module, class_names: list, image_path: str) -> dict:
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [<image_path> ...]")
        sys.exit(1)

    model, class_names = load_model(MODEL_DIR)
    print(f"Model loaded  |  classes: {class_names}\n")

    for path in sys.argv[1:]:
        result = predict(model, class_names, path)
        print(f"{path}")
        print(f"  Prediction : {result['prediction'].upper()}  ({result['confidence']:.1%})")
        for cls, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"  {cls:<10} {score:.1%}  {bar}")
        print()


if __name__ == "__main__":
    main()
