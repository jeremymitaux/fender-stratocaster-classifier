"""
Prepare the dataset for guitar make/model classification.

1. Load every JSON file in data/json/ and de-duplicate listings by id.
2. Label each listing by MODEL, read from the structured ``make`` / ``model`` /
   ``title`` metadata (not the old origin keyword heuristic). A listing is kept
   only if it:
       - comes from an allowed guitar brand (drops part-sellers / "Unbranded"),
       - matches exactly ONE canonical model (drops ambiguous / unmatched),
       - is NOT a parts/accessory listing (backplate, tremolo arm, pickguard, …),
       - is priced at/above MIN_PRICE (cuts most remaining accessories).
3. Download up to MAX_IMAGES_PER_LISTING images per listing into:
       data/images_labeled/<model>/<listing_id>_<idx>.<ext>
   The ``<listing_id>_`` filename prefix is what the grouped train/val/test split
   in data.py keys on, so all photos of one guitar stay in a single split.
4. Print the per-class distribution (and what was dropped, and why).

Use ``--dry-run`` to print the label distribution without downloading anything.

Run with::

    strat-prepare                 # label + download
    strat-prepare --dry-run       # just show the class breakdown
"""

import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

JSON_DIR = Path("data/json")
OUT_DIR = Path("data/images_labeled")
MAX_IMAGES_PER_LISTING = 5
MIN_PRICE = 150.0           # USD — below this is almost always a part/accessory
REQUEST_DELAY = 0.25        # seconds between image downloads


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

# Canonical model -> regex patterns matched against (title + model), lowercased.
# Word boundaries keep short tokens (sg, 335) from matching inside other words.
MODEL_PATTERNS: dict[str, list[str]] = {
    "stratocaster":      [r"stratocaster", r"\bstrat\b"],
    "telecaster":        [r"telecaster", r"\btele\b"],
    "les_paul":          [r"les\s*paul"],
    "sg":                [r"\bsg\b"],
    "es_335":            [r"es[\s-]*335", r"\b335\b"],
    "jazzmaster_jaguar": [r"jazzmaster", r"jaguar"],
}

# A listing's `make` must contain one of these (drops "Unbranded",
# "Paradox Pickguards", part-seller names, etc.).
ALLOWED_BRANDS = ["fender", "squier", "gibson", "epiphone"]

# If any of these appear in title/model, the listing is a part/accessory, not a guitar.
PART_KEYWORDS = [
    "backplate", "back plate", "pickguard", "tremolo arm", "trem arm",
    "whammy", "vibrato arm", "tuners", "tuning machine", "bridge assembly",
    "cavity cover", "neck plate", "neckplate", "pickup set", "pickup cover",
    "loaded pickguard", "cover for", "arm for", "for fender", "for gibson",
    "replacement neck", "decal", "knob set", "set of",
]


def _text(item: dict) -> str:
    """Lowercased title + model, HTML stripped."""
    t = " ".join([str(item.get("title", "")), str(item.get("model", ""))]).lower()
    return re.sub(r"<[^>]+>", " ", t)


def _price(item: dict) -> float | None:
    p = item.get("price")
    raw = p.get("amount") if isinstance(p, dict) else p
    try:
        return float(str(raw).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def _is_part(text: str) -> bool:
    return any(k in text for k in PART_KEYWORDS)


def _brand_ok(item: dict) -> bool:
    make = str(item.get("make") or item.get("brand") or "").lower()
    return any(b in make for b in ALLOWED_BRANDS)


def assign_label(item: dict) -> tuple[str | None, str]:
    """Return ``(label, reason)``. ``label`` is None when the listing is dropped;
    ``reason`` explains why (for the dropped-counts summary)."""
    if not _brand_ok(item):
        return None, "bad_brand"

    text = _text(item)
    if _is_part(text):
        return None, "part"

    price = _price(item)
    if price is None:
        return None, "no_price"
    if price < MIN_PRICE:
        return None, "too_cheap"

    matches = [
        model for model, patterns in MODEL_PATTERNS.items()
        if any(re.search(p, text) for p in patterns)
    ]
    if len(matches) != 1:
        return None, "ambiguous" if matches else "no_model"
    return matches[0], "ok"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def image_urls(item: dict) -> list[str]:
    entries = item.get("images")
    if isinstance(entries, list):
        return [u for u in entries if isinstance(u, str) and u]
    return []


def download_image(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:
        print(f"  [skip] {url}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_listings(json_dir: Path) -> list[dict]:
    """Load every JSON file under ``json_dir`` and de-duplicate by listing id."""
    by_id: dict[str, dict] = {}
    n_files = 0
    for path in sorted(json_dir.glob("*.json")):
        n_files += 1
        items = json.load(open(path))
        if isinstance(items, dict):
            items = [items]
        for item in items:
            by_id[str(item.get("id"))] = item
    print(f"Loaded {len(by_id)} unique listings from {n_files} file(s) in {json_dir}")
    return list(by_id.values())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Label listings by make/model and download images.")
    p.add_argument("--json-dir", type=Path, default=JSON_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-images", type=int, default=MAX_IMAGES_PER_LISTING)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the class breakdown without downloading images.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    listings = load_listings(args.json_dir)

    labeled: list[tuple[dict, str]] = []
    dropped = Counter()
    for item in listings:
        label, reason = assign_label(item)
        if label is None:
            dropped[reason] += 1
        else:
            labeled.append((item, label))

    dist = Counter(lbl for _, lbl in labeled)
    print(f"\nKept {len(labeled)} listings across {len(dist)} classes:")
    for cls, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s} {n}")
    print(f"\nDropped {sum(dropped.values())} listings:")
    for reason, n in sorted(dropped.items(), key=lambda x: -x[1]):
        print(f"  {reason:20s} {n}")

    if args.dry_run:
        print("\n(dry run — no images downloaded)")
        return

    for cls in MODEL_PATTERNS:
        (args.out_dir / cls).mkdir(parents=True, exist_ok=True)

    saved = Counter()
    for i, (item, lbl) in enumerate(labeled):
        listing_id = item.get("id", f"item_{i}")
        for idx, url in enumerate(image_urls(item)[: args.max_images]):
            ext = (url.split("?")[0].rsplit(".", 1)[-1] or "jpg").lower()
            dest = args.out_dir / lbl / f"{listing_id}_{idx}.{ext}"
            if download_image(url, dest):
                saved[lbl] += 1
            time.sleep(REQUEST_DELAY)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(labeled)} listings  |  images saved: {dict(saved)}")

    print(f"\nDone. Images saved per class: {dict(saved)}")
    print(f"Output directory: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
