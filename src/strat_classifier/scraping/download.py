"""
Download listing images from data/strats.csv into data/images/<listing_id>/.

Run after scrape_strats.py has finished.
Images are the raw material for training a visual guitar ID model.
"""

import csv
import os
import time
import urllib.request
from pathlib import Path

CSV_PATH = Path("data/strats.csv")
IMG_DIR = Path("data/images")
MAX_IMAGES_PER_LISTING = 3  # first N images per listing
DELAY = 0.3                 # seconds between requests


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:
        print(f"  [skip] {url}: {exc}")
        return False


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    downloaded = 0

    for i, row in enumerate(rows):
        listing_id = row.get("id", f"row_{i}")
        urls = [u.strip() for u in row.get("images", "").split("|") if u.strip()]
        urls = urls[:MAX_IMAGES_PER_LISTING]

        if not urls:
            continue

        listing_dir = IMG_DIR / listing_id
        listing_dir.mkdir(exist_ok=True)

        for j, url in enumerate(urls):
            ext = url.split("?")[0].rsplit(".", 1)[-1] or "jpg"
            dest = listing_dir / f"{j}.{ext}"
            if download(url, dest):
                downloaded += 1
            time.sleep(DELAY)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total} listings processed, {downloaded} images saved")

    print(f"\nDone. {downloaded} images saved to {IMG_DIR}/")


if __name__ == "__main__":
    main()
