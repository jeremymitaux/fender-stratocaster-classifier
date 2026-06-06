"""Train the guitar make/model classifier (e.g. stratocaster / telecaster).

Dataloaders and transforms live in :mod:`strat_classifier.data`; the model
lives in :mod:`strat_classifier.model`. This module only owns the training
loop, evaluation, and result plotting.

Outputs:
    models/best_model.pt    — best checkpoint (by val accuracy)
    models/last_model.pt    — final checkpoint
    models/class_names.json — class ordering used at train time
    results/                — confusion matrix + training curves

Run with the installed console script::

    strat-train

or as a module::

    python -m strat_classifier.train
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from strat_classifier.data import DEFAULT_DATA_DIR, make_dataloaders
from strat_classifier.model import build_model

# --- Defaults --------------------------------------------------------------
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")
EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 32
SEED = 42

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# --- Train / eval loops ----------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def get_preds(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# --- Plotting --------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved confusion matrix → {save_path}")


def plot_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved training curves → {save_path}")


# --- Main ------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the guitar make/model classifier.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Split  — train: {len(train_loader.dataset)}  "
          f"val: {len(val_loader.dataset)}  test: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.model_dir / "best_model.pt")
            flag = "  ← best"

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}{flag}")

    torch.save(model.state_dict(), args.model_dir / "last_model.pt")

    # Final evaluation on the held-out test set, using the best checkpoint.
    model.load_state_dict(torch.load(args.model_dir / "best_model.pt", map_location=DEVICE))
    y_true, y_pred = get_preds(model, test_loader, DEVICE)

    print("\n--- Test Set Results ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names, args.result_dir / "confusion_matrix.png")
    plot_curves(history, args.result_dir / "training_curves.png")

    (args.model_dir / "class_names.json").write_text(json.dumps(class_names))
    print(f"\nBest val acc: {best_val_acc:.3f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
