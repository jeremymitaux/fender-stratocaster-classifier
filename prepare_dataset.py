"""
Prepare dataset for Fender Stratocaster origin classification (American / Mexican / Japanese).

1. Parse all JSON files in data/json/
2. Label each listing by country of origin using title + model + description keywords
3. Download up to MAX_IMAGES_PER_LISTING images per listing into:
       data/images_labeled/<label>/<listing_id>_<idx>.<ext>
4. Print a summary of downloaded images per class
"""

import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

JSON_DIR = Path("data/json")
OUT_DIR = Path("data/images_labeled")
MAX_IMAGES_PER_LISTING = 3
REQUEST_DELAY = 0.25  # seconds between HTTP requests


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def assign_label(item: dict) -> str | None:
    text = " ".join([
        item.get("title", ""),
        item.get("model", ""),
        item.get("description", ""),
    ]).lower()

    # Strip HTML tags from description
    text = re.sub(r"<[^>]+>", " ", text)

    japanese_kws = [
        "made in japan", "crafted in japan", "mij",
        "fujigen", "japan", "cij",
    ]
    mexican_kws = [
        "made in mexico", "mim", "ensenada", "mexico",
        "player series", "player stratocaster",
    ]
    american_kws = [
        "made in usa", "made in u.s.a", "american",
        "usa", "u.s.a.", "corona, ca", "fullerton",
    ]

    is_japanese = any(k in text for k in japanese_kws)
    is_mexican  = any(k in text for k in mexican_kws)
    is_american = any(k in text for k in american_kws)

    matched = sum([is_japanese, is_mexican, is_american])
    if matched != 1:
        return None  # ambiguous or unlabeled — skip

    if is_japanese:
        return "japanese"
    if is_mexican:
        return "mexican"
    return "american"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

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

def main():
    # Load all listings
    all_items = []
    for path in sorted(JSON_DIR.glob("*.json")):
        with open(path) as f:
            all_items.extend(json.load(f))
    print(f"Loaded {len(all_items)} listings from {JSON_DIR}")

    # Label
    labeled = [(item, assign_label(item)) for item in all_items]
    labeled = [(item, lbl) for item, lbl in labeled if lbl is not None]
    dist = Counter(lbl for _, lbl in labeled)
    print(f"Labeled: {dict(dist)}  (dropped {len(all_items) - len(labeled)} ambiguous)")

    # Create output dirs
    for cls in ["american", "mexican", "japanese"]:
        (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    # Download
    total_saved = Counter()
    for i, (item, lbl) in enumerate(labeled):
        urls = item.get("images", [])[:MAX_IMAGES_PER_LISTING]
        listing_id = item.get("id", f"item_{i}")

        for idx, url in enumerate(urls):
            ext = url.split("?")[0].rsplit(".", 1)[-1] or "jpg"
            dest = OUT_DIR / lbl / f"{listing_id}_{idx}.{ext}"
            if download_image(url, dest):
                total_saved[lbl] += 1
            time.sleep(REQUEST_DELAY)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(labeled)} listings  |  saved so far: {dict(total_saved)}")

    print(f"\nDone. Images saved: {dict(total_saved)}")
    print(f"Output directory: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
