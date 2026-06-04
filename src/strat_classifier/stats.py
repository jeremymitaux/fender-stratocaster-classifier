"""Dataset statistics for the guitar make/model classifier.

Reports, per class and overall:
  - number of unique listings (grouped on the ``<listing_id>`` filename prefix),
  - number of images,
  - mean images per listing,
  - the group-aware train/val/test split sizes (listings *and* images).

Prints a Markdown table (paste-ready for the README) and writes it to
``results/dataset_stats.md``.

Run with::

    python -m strat_classifier.stats
    python -m strat_classifier.stats --data-dir data/images_labeled
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from strat_classifier.data import DEFAULT_DATA_DIR, build_splits, listing_id

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def scan(data_dir: Path):
    """Return ``{class: {"images": int, "listings": set}}`` for each class folder."""
    out: dict[str, dict] = {}
    for cls_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        listings = {listing_id(p.name) for p in imgs}
        out[cls_dir.name] = {"images": len(imgs), "listings": listings}
    return out


def split_breakdown(data_dir: Path, seed: int = 42):
    """Images + unique listings landing in each split, per class."""
    train_ds, val_ds, test_ds, class_names = build_splits(data_dir, seed=seed)
    base = train_ds.dataset  # the shared ImageFolder
    rows = {}
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        imgs = Counter()
        listings = defaultdict(set)
        for i in ds.indices:
            path, label = base.samples[i]
            cls = class_names[label]
            imgs[cls] += 1
            listings[cls].add(listing_id(Path(path).name))
        rows[name] = {"images": imgs, "listings": {c: len(s) for c, s in listings.items()}}
    return class_names, rows


def build_report(data_dir: Path, seed: int = 42) -> str:
    per_class = scan(data_dir)
    if not per_class:
        return f"No class folders found in {data_dir}."

    lines = ["# Dataset statistics", "", f"Source: `{data_dir}`", ""]

    # --- Per-class totals ---
    lines += [
        "## Per-class totals",
        "",
        "| Class | Listings | Images | Images/listing |",
        "|---|---:|---:|---:|",
    ]
    tot_listings = tot_images = 0
    for cls, d in sorted(per_class.items(), key=lambda x: -len(x[1]["listings"])):
        nl, ni = len(d["listings"]), d["images"]
        tot_listings += nl
        tot_images += ni
        ratio = ni / nl if nl else 0
        lines.append(f"| {cls} | {nl} | {ni} | {ratio:.1f} |")
    ratio = tot_images / tot_listings if tot_listings else 0
    lines.append(f"| **Total** | **{tot_listings}** | **{tot_images}** | **{ratio:.1f}** |")
    lines.append("")

    # --- Split breakdown ---
    class_names, rows = split_breakdown(data_dir, seed=seed)
    lines += [
        "## Group-aware split (by listing, seed=%d)" % seed,
        "",
        "All images of one listing stay in a single split (no leakage).",
        "",
        "| Split | " + " | ".join(class_names) + " | Listings | Images |",
        "|---|" + "|".join(["---:"] * (len(class_names) + 2)) + "|",
    ]
    for name in ("train", "val", "test"):
        imgs = rows[name]["images"]
        lst = rows[name]["listings"]
        per_cls = " | ".join(str(lst.get(c, 0)) for c in class_names)
        n_list = sum(lst.values())
        n_img = sum(imgs.values())
        lines.append(f"| {name} | {per_cls} | {n_list} | {n_img} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Report dataset statistics.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=Path("results/dataset_stats.md"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args.data_dir, seed=args.seed)
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
