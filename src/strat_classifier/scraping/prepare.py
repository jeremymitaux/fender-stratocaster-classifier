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
import io
import json
import re
import ssl
import time
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image

# Reverb's CDN occasionally serves HEIC. PIL can't decode it without the
# pillow-heif plugin, so register it when available; otherwise HEIC downloads
# are skipped (logged) rather than crashing. JPEG/PNG/WebP/GIF need no plugin.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# Build an SSL context from certifi's CA bundle if available. python.org builds
# on macOS don't use the system trust store, so a bare urlopen() raises
# CERTIFICATE_VERIFY_FAILED on Reverb's image CDN; certifi fixes that cleanly.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:                       # certifi missing → fall back to system default
    _SSL_CTX = ssl.create_default_context()

JSON_DIR = Path("data/json")
OUT_DIR = Path("data/images_labeled")
MAX_IMAGES_PER_LISTING = 5
MIN_PRICE = 150.0           # USD — below this is almost always a part/accessory
REQUEST_DELAY = 0.25        # seconds between image downloads


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

# Canonical model -> regex patterns matched against (title + model), lowercased.
# Word boundaries keep short tokens from matching inside other words.
# Scope: a 2-class Fender Stratocaster vs Telecaster classifier. (Earlier
# iterations carried les_paul / sg / es_335 / jazzmaster_jaguar patterns for a
# planned 6-class set; that was dropped — see PROPOSAL.md / NEXT_STEPS.md.)
MODEL_PATTERNS: dict[str, list[str]] = {
    "stratocaster": [r"stratocaster", r"\bstrat\b"],
    "telecaster":   [r"telecaster", r"\btele\b"],
}

# A listing's `make` must contain one of these (drops "Unbranded",
# "Paradox Pickguards", part-seller names, etc.). Strat/Tele are Fender
# designs, so only Fender and its budget brand Squier are in scope.
ALLOWED_BRANDS = ["fender", "squier"]

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


def download_image(url: str, dest: Path) -> str:
    """Download ``url`` and write it to ``dest`` re-encoded as JPEG. Returns
    ``"exists"`` if already on disk (no network request), ``"ok"`` on a
    successful fetch, or ``"fail"``.

    Every image is normalized to RGB JPEG so the on-disk dataset is a single
    format (no stray HEIC/WebP/GIF that ``torchvision.ImageFolder`` would
    silently skip or that PIL can't open). ``dest`` is expected to end in
    ``.jpg``. Callers throttle only on a real fetch, so re-runs over an existing
    dataset are fast (idempotent)."""
    if dest.exists():
        return "exists"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = resp.read()
        # convert("RGB") flattens alpha/palette and takes the first frame of an
        # animated GIF, so JPEG encoding always succeeds.
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(dest, "JPEG", quality=90)
        return "ok"
    except Exception as exc:
        print(f"  [skip] {url}: {exc}")
        return "fail"


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

    # Only create folders for classes that actually have listings — an empty
    # folder would become a phantom zero-sample class in ImageFolder.
    for cls in dist:
        (args.out_dir / cls).mkdir(parents=True, exist_ok=True)

    saved = Counter()
    for i, (item, lbl) in enumerate(labeled):
        listing_id = item.get("id", f"item_{i}")
        for idx, url in enumerate(image_urls(item)[: args.max_images]):
            # Always .jpg: download_image re-encodes every image to JPEG, and a
            # fixed extension keeps the idempotent "skip if dest exists" check
            # working regardless of the source URL's format.
            dest = args.out_dir / lbl / f"{listing_id}_{idx}.jpg"
            status = download_image(url, dest)
            if status != "fail":
                saved[lbl] += 1
            if status == "ok":          # only throttle on a real network fetch
                time.sleep(REQUEST_DELAY)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(labeled)} listings  |  images saved: {dict(saved)}")

    print(f"\nDone. Images saved per class: {dict(saved)}")
    print(f"Output directory: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
