"""Gradio demo for the Strat vs. Tele guitar classifier.

Upload a guitar photo and get the predicted make/model with per-class
confidence. Model loading and preprocessing are reused from the
``strat_classifier`` package — this file is only the UI glue.

Run::

    python demo/app.py        # serves on http://localhost:7860

Requires the ``demo`` extra: ``pip install -e .[demo]``.

Deploying to Hugging Face Spaces later would mean little more than copying
this file plus ``models/`` and listing ``gradio`` in requirements — left out
for now since this is local-only.
"""

from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from strat_classifier.inference import TRANSFORM, load_model

# Paths are resolved relative to the project root (demo/ -> repo root), so the
# app works regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = ROOT / "data" / "sample_images"


def _select_device() -> torch.device:
    """Prefer Apple's MPS backend on Mac, fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device()

# Load once at startup. ``load_model`` reads class_names.json + best_model.pt
# from the package; we only move it onto the chosen device.
MODEL, CLASS_NAMES = load_model(MODEL_DIR)
MODEL.to(DEVICE)


@torch.no_grad()
def classify(image: Image.Image):
    """Return per-class confidences for one uploaded PIL image.

    The package's ``predict`` keeps its tensor on CPU, so for device support we
    reuse its ``TRANSFORM`` and run a device-aware forward pass here.
    """
    if image is None:
        return None, "Upload a guitar photo to classify."

    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(MODEL(tensor)[0], dim=0)

    scores = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
    top = max(scores, key=scores.get)
    summary = f"**{top.capitalize()}** — {scores[top]:.1%} confidence"
    return scores, summary


def build_interface() -> gr.Blocks:
    examples = sorted(str(p) for p in SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []

    with gr.Blocks(title="Strat vs. Tele Classifier") as demo:
        gr.Markdown(
            "# 🎸 Strat vs. Tele Classifier\n"
            f"Upload a guitar photo to classify it as **{' or '.join(CLASS_NAMES)}**. "
            f"Running on `{DEVICE.type}`."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Guitar image", sources=["upload", "clipboard"])
                classify_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                summary_out = gr.Markdown(label="Prediction")
                label_out = gr.Label(num_top_classes=len(CLASS_NAMES), label="Confidence")

        if examples:
            gr.Examples(examples=examples, inputs=image_in, label="Sample images")

        classify_btn.click(classify, inputs=image_in, outputs=[label_out, summary_out])
        image_in.change(classify, inputs=image_in, outputs=[label_out, summary_out])

    return demo


if __name__ == "__main__":
    build_interface().launch(server_name="127.0.0.1", server_port=7860)
