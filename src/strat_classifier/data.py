"""Dataset and DataLoader construction for the Stratocaster origin classifier.

Images are expected on disk as an ``ImageFolder`` tree::

    data/images_labeled/
        american/  *.jpg *.jpeg *.png
        japanese/  ...
        mexican/   ...

The public entry point is :func:`make_dataloaders`, which returns stratified
train/val/test loaders plus the class-name list. Training uses a
``WeightedRandomSampler`` to compensate for class imbalance (American listings
outnumber Mexican and Japanese ones).
"""

from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

# --- Constants -------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_DATA_DIR = Path("data/images_labeled")


# --- Transforms ------------------------------------------------------------
def build_transforms(img_size: int = IMG_SIZE):
    """Return ``(train_tf, eval_tf)`` transform pipelines.

    Training adds light augmentation (flip / color jitter / rotation); eval is
    deterministic. Both resize to ``img_size`` and normalize with ImageNet
    statistics so the pretrained backbone sees inputs in its expected range.
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# --- Stratified splits -----------------------------------------------------
def build_splits(
    data_dir: Path = DEFAULT_DATA_DIR,
    img_size: int = IMG_SIZE,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
):
    """Build stratified train/val/test ``Subset``s and the class-name list.

    Train uses augmenting transforms; val/test use deterministic eval
    transforms. Splits are stratified on the class label so each split keeps
    the same american/japanese/mexican proportions.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Image folder not found: {data_dir}\n"
            "Run `strat-prepare` (or python -m strat_classifier.scraping.prepare) "
            "to build the labeled dataset."
        )

    train_tf, eval_tf = build_transforms(img_size)
    train_base = datasets.ImageFolder(str(data_dir), transform=train_tf)
    eval_base = datasets.ImageFolder(str(data_dir), transform=eval_tf)
    targets = [label for _, label in train_base.samples]

    idx = list(range(len(train_base)))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_split, stratify=targets, random_state=seed
    )
    targets_trainval = [targets[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_split / (1 - test_split),
        stratify=targets_trainval,
        random_state=seed,
    )

    train_ds = Subset(train_base, idx_train)   # augmented
    val_ds = Subset(eval_base, idx_val)        # deterministic
    test_ds = Subset(eval_base, idx_test)      # deterministic
    return train_ds, val_ds, test_ds, train_base.classes


# --- Sampler ---------------------------------------------------------------
def make_weighted_sampler(train_ds: Subset) -> WeightedRandomSampler:
    """Inverse-frequency sampler so minority classes are not drowned out."""
    labels = [train_ds.dataset.targets[i] for i in train_ds.indices]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# --- DataLoaders -----------------------------------------------------------
def make_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR,
    img_size: int = IMG_SIZE,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    pin_memory: bool | None = None,
):
    """Return ``(train_loader, val_loader, test_loader, class_names)``.

    The train loader draws samples with a ``WeightedRandomSampler``; val/test
    iterate in order without shuffling. Pass ``num_workers=0`` when calling
    from a notebook to avoid multiprocessing issues. ``pin_memory`` defaults to
    on only when CUDA is available (it has no benefit on CPU/MPS).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds, val_ds, test_ds, class_names = build_splits(
        data_dir, img_size=img_size, val_split=val_split,
        test_split=test_split, seed=seed,
    )
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, class_names
