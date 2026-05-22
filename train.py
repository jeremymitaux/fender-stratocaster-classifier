"""
Train an EfficientNet-B0 classifier to distinguish Fender Stratocaster origin:
    american | mexican | japanese

Expects images in:
    data/images_labeled/
        american/  *.jpg *.jpeg *.png
        mexican/   ...
        japanese/  ...

Outputs:
    models/best_model.pt   — best checkpoint (by val accuracy)
    models/last_model.pt   — final checkpoint
    results/               — confusion matrix + training curves
"""

import os
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data/images_labeled")
MODEL_DIR  = Path("models")
RESULT_DIR = Path("results")

CLASSES    = ["american", "mexican", "japanese"]
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
SEED       = 42

VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Dataset split (stratified)
# ---------------------------------------------------------------------------
def build_splits(data_dir: Path):
    full_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
    targets = [s[1] for s in full_ds.samples]

    idx = list(range(len(full_ds)))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_SPLIT, stratify=targets, random_state=SEED
    )
    targets_trainval = [targets[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        stratify=targets_trainval,
        random_state=SEED,
    )

    train_ds = Subset(full_ds, idx_train)
    val_ds   = Subset(full_ds, idx_val)
    test_ds  = Subset(full_ds, idx_test)

    # Val/test use eval transforms
    eval_ds_base = datasets.ImageFolder(str(data_dir), transform=eval_tf)
    val_ds  = Subset(eval_ds_base, idx_val)
    test_ds = Subset(eval_ds_base, idx_test)

    class_names = full_ds.classes
    return train_ds, val_ds, test_ds, class_names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
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
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved training curves → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")

    train_ds, val_ds, test_ds, class_names = build_splits(DATA_DIR)
    print(f"Split  — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    print(f"Classes: {class_names}")

    # Weighted sampler to compensate for class imbalance
    train_labels = [train_ds.dataset.targets[i] for i in train_ds.indices]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=True)

    model = build_model(num_classes=len(class_names)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc = eval_epoch(model, val_loader,   criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            flag = "  ← best"

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}{flag}"
        )

    torch.save(model.state_dict(), MODEL_DIR / "last_model.pt")

    # --- Final evaluation on test set ---
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE))
    y_true, y_pred = get_preds(model, test_loader, DEVICE)

    print("\n--- Test Set Results ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names, RESULT_DIR / "confusion_matrix.png")
    plot_curves(history, RESULT_DIR / "training_curves.png")

    # Save class mapping for inference
    import json
    (MODEL_DIR / "class_names.json").write_text(json.dumps(class_names))
    print(f"\nBest val acc: {best_val_acc:.3f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
