"""Dataset and DataLoader construction for the guitar make/model classifier.

Images are expected on disk as an ``ImageFolder`` tree, one folder per model::

    data/images_labeled/
        stratocaster/  <listing_id>_<idx>.jpg ...
        telecaster/    ...
        les_paul/      ...

The public entry point is :func:`make_dataloaders`, which returns
**group-aware**, stratified train/val/test loaders plus the class-name list.

Every image filename is ``<listing_id>_<idx>.<ext>`` (written by
``scraping.prepare``), so multiple photos of the *same* listing share a
``listing_id`` prefix. The split groups on that id — all images of one listing
land in a single split — which prevents near-duplicate photos of one guitar
leaking across train/val/test and inflating accuracy. Training also uses a
``WeightedRandomSampler`` to compensate for residual class imbalance.
"""

import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
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


# --- Group-aware stratified splits -----------------------------------------
def listing_id(path: str) -> str:
    """Group key for an image: the ``<listing_id>`` from ``<listing_id>_<idx>.<ext>``.

    Filenames are written by ``scraping.prepare`` as ``{listing_id}_{idx}.{ext}``.
    Reverb ids are numeric, but rsplit on the *last* ``_`` is robust even if an
    id ever contained one. Falls back to the whole stem if there is no ``_``.
    """
    stem = Path(path).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def build_splits(
    data_dir: Path = DEFAULT_DATA_DIR,
    img_size: int = IMG_SIZE,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
):
    """Build group-aware, stratified train/val/test ``Subset``s + class names.

    Train uses augmenting transforms; val/test use deterministic eval
    transforms. The split is performed over **listings** (grouped on the
    ``listing_id`` filename prefix), not individual images, so every photo of a
    given guitar stays in one split. Within each class, listings are shuffled
    and partitioned by the requested fractions, which keeps the per-split class
    proportions close to the overall distribution (stratification).
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

    # Map each listing (group) to its label and the image indices it owns.
    group_label: dict[str, int] = {}
    group_indices: dict[str, list[int]] = defaultdict(list)
    for i, (path, label) in enumerate(train_base.samples):
        g = listing_id(path)
        group_label[g] = label
        group_indices[g].append(i)

    # Bucket listings by class, then split each class's listings by fraction so
    # the train/val/test proportions are preserved per class (stratified).
    by_label: dict[int, list[str]] = defaultdict(list)
    for g, label in group_label.items():
        by_label[label].append(g)

    rng = random.Random(seed)
    idx_train: list[int] = []
    idx_val: list[int] = []
    idx_test: list[int] = []
    for label, groups in by_label.items():
        groups = sorted(groups)          # deterministic before shuffle
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        test_g = groups[:n_test]
        val_g = groups[n_test:n_test + n_val]
        train_g = groups[n_test + n_val:]
        for g in test_g:
            idx_test += group_indices[g]
        for g in val_g:
            idx_val += group_indices[g]
        for g in train_g:
            idx_train += group_indices[g]

    # Safety: no listing may appear in more than one split.
    g_train = {listing_id(train_base.samples[i][0]) for i in idx_train}
    g_val = {listing_id(train_base.samples[i][0]) for i in idx_val}
    g_test = {listing_id(train_base.samples[i][0]) for i in idx_test}
    assert not (g_train & g_val) and not (g_train & g_test) and not (g_val & g_test), \
        "listing leaked across splits — group-aware split is broken"

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
