"""
Collect Fender Stratocaster listings from Reverb.com via the Apify scraper.

Outputs:
  data/strats_raw.jsonl   — one JSON object per line, full API response
  data/strats.csv         — flat CSV with key fields for ML training
"""

import csv
import json
import os
import time
from pathlib import Path

from apify_client import ApifyClient  # type: ignore

API_TOKEN = os.environ.get("APIFY_TOKEN", "")  # export APIFY_TOKEN=... before running
ACTOR_ID = "parseforge/reverb-com-scraper"

# How many listings to target. Raise for a bigger dataset.
TARGET_ITEMS = 2000
PER_PAGE = 100  # max allowed

CSV_FIELDS = [
    "id",
    "title",
    "price",
    "currency",
    "condition",
    "year",
    "make",
    "model",
    "finish",
    "country_of_origin",
    "seller_location",
    "listing_url",
    "images",          # pipe-separated image URLs
    "description",
]

SEARCH_QUERIES = [
    "fender stratocaster",
    "fender strat american",
    "fender strat mexico",
    "fender strat japan",
    "fender strat vintage",
]


def _href_from_photo(photo) -> str:
    """Pull a URL out of one photo entry, which may be a dict or a plain string."""
    if isinstance(photo, str):
        return photo
    if isinstance(photo, dict):
        return (photo.get("_links", {}).get("large_crop", {}).get("href", "")
                or photo.get("url", ""))
    return ""


def image_urls(item: dict) -> str:
    """Pipe-separated image URLs.

    Reverb returns both `photos` (list of dicts with crop links) and `images`
    (list of plain URL strings). Prefer `photos` for the larger crops, then fall
    back to `images`. Each entry is type-checked so a string never gets `.get()`.
    """
    for field in ("photos", "images"):
        entries = item.get(field)
        if not isinstance(entries, list):
            continue
        urls = [u for u in (_href_from_photo(e) for e in entries) if u]
        if urls:
            return "|".join(urls)
    return ""


def flatten(item: dict) -> dict:
    """Extract the fields we care about into a flat dict."""
    price_obj = item.get("price") or {}
    price = price_obj.get("amount") or item.get("price") or ""
    currency = price_obj.get("currency") or "USD"

    # `condition` may be a plain string ("Brand New") or a dict with display_name.
    condition = item.get("condition") or ""
    if isinstance(condition, dict):
        condition = condition.get("display_name", "")

    return {
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "price": price,
        "currency": currency,
        "condition": condition,
        "year": item.get("year", ""),
        "make": item.get("make") or item.get("brand", ""),
        "model": item.get("model", ""),
        "finish": item.get("finish", ""),
        "country_of_origin": item.get("country_of_origin", ""),
        "seller_location": item.get("shop_location", item.get("seller_location", "")),
        "listing_url": item.get("_links", {}).get("web", {}).get("href", item.get("url", "")),
        "images": image_urls(item),
        "description": (item.get("description") or "").replace("\n", " ").strip(),
    }


def run_scrape(client: ApifyClient, query: str, page: int) -> list[dict]:
    run_input = {
        "searchQuery": query,
        "maxItems": PER_PAGE,
        "perPage": PER_PAGE,
        "page": page,
    }
    try:
        run = client.actor(ACTOR_ID).call(run_input=run_input)
        return list(client.dataset(run.default_dataset_id).iterate_items())
    except Exception as exc:
        print(f"  [warn] page {page} failed: {exc}")
        return []


def main():
    if not API_TOKEN:
        raise SystemExit("Set the APIFY_TOKEN env var (export APIFY_TOKEN=...) before running.")
    Path("data").mkdir(exist_ok=True)
    client = ApifyClient(API_TOKEN)

    seen_ids: set[str] = set()
    all_rows: list[dict] = []

    jsonl_path = Path("data/strats_raw.jsonl")
    csv_path = Path("data/strats.csv")

    with jsonl_path.open("w") as jf, csv_path.open("w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for query in SEARCH_QUERIES:
            if len(all_rows) >= TARGET_ITEMS:
                break
            print(f"\nQuery: '{query}'")

            for page in range(1, 201):  # up to 200 pages × 100 = 20k per query
                if len(all_rows) >= TARGET_ITEMS:
                    break

                print(f"  page {page} — {len(all_rows)} listings so far", end="\r")
                items = run_scrape(client, query, page)

                if not items:
                    print(f"\n  no results on page {page}, moving on")
                    break

                new_items = 0
                for item in items:
                    item_id = str(item.get("id", ""))
                    if item_id and item_id in seen_ids:
                        continue
                    if item_id:
                        seen_ids.add(item_id)

                    jf.write(json.dumps(item) + "\n")
                    row = flatten(item)
                    writer.writerow(row)
                    all_rows.append(row)
                    new_items += 1

                if new_items == 0:
                    print(f"\n  all duplicates on page {page}, stopping query")
                    break

                time.sleep(1)  # polite delay between API calls

    print(f"\nDone. {len(all_rows)} listings saved.")
    print(f"  Raw JSON: {jsonl_path}")
    print(f"  CSV:      {csv_path}")

    # Quick summary
    makes = {}
    for row in all_rows:
        makes[row["make"]] = makes.get(row["make"], 0) + 1
    print("\nTop makes found:")
    for make, count in sorted(makes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {make or '(unknown)':30s} {count}")


if __name__ == "__main__":
    main()
