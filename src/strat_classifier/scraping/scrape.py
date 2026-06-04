"""
Collect electric-guitar listings from Reverb.com via the Apify scraper.

One search query is run per target model (Stratocaster, Telecaster, Les Paul, …)
so every class in the make/model classifier actually gets populated. Each query is
paginated until it runs dry, results are de-duplicated by listing id across all
queries, and the category filter restricts results to electric guitars (which cuts
out most of the parts/accessories noise that a plain text search returns).

Outputs:
  data/json/scraped_<timestamp>.json  — JSON array of raw listings (consumed by `prepare`)
  data/strats.csv                     — flat CSV with key fields (quick inspection)

Run with::

    export APIFY_TOKEN=...
    strat-scrape
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

# apify-client is an optional dependency (install with: pip install -e ".[scrape]").
# It is imported lazily inside main() so the rest of the package imports without it.

API_TOKEN = os.environ.get("APIFY_TOKEN", "")  # export APIFY_TOKEN=... before running
ACTOR_ID = "parseforge/reverb-com-scraper"

# Restrict results to electric guitars. The actor's input accepts `categorySlug`;
# this is the main lever for filtering out parts/accessories. If a run still comes
# back full of backplates/tremolo arms, the real filtering happens in prepare.py.
CATEGORY_SLUG = "electric-guitars"

PER_PAGE = 100        # page size (max allowed)
MAX_PAGES = 20        # safety cap per query — 20 × 100 = up to 2k per model
REQUEST_DELAY = 1.0   # polite delay between API calls (seconds)

# One query per target model. jazzmaster + jaguar both feed the offset-body class.
# Add/remove queries to match the class set you settle on in NEXT_STEPS.md §1.
SEARCH_QUERIES = [
    "fender stratocaster",
    "fender telecaster",
    "gibson les paul",
    "gibson sg",
    "gibson es-335",
    "fender jazzmaster",
    "fender jaguar",
]

CSV_FIELDS = [
    "id",
    "title",
    "price",
    "currency",
    "condition",
    "year",
    "make",
    "model",
    "seller",
    "listing_url",
    "images",          # pipe-separated image URLs
    "description",
]


# --- Field extraction -------------------------------------------------------
def _price(item: dict) -> tuple[str, str]:
    """Return ``(amount, currency)`` from the price dict (or plain string)."""
    p = item.get("price")
    if isinstance(p, dict):
        return str(p.get("amount", "")), p.get("currency", "USD")
    return str(p or ""), "USD"


def _condition(item: dict) -> str:
    c = item.get("condition") or ""
    if isinstance(c, dict):
        return c.get("display_name", "") or c.get("condition_display_name", "")
    return c


def _make(item: dict) -> str:
    return item.get("make") or item.get("brand") or ""


def _seller(item: dict) -> str:
    s = item.get("seller")
    if isinstance(s, dict):
        return s.get("username", "") or s.get("shop_name", "")
    return item.get("shop_name", "") or ""


def _image_urls(item: dict) -> list[str]:
    """List of image URLs. Reverb's export stores ``images`` as plain URL strings."""
    entries = item.get("images")
    if isinstance(entries, list):
        return [u for u in entries if isinstance(u, str) and u]
    return []


def flatten(item: dict) -> dict:
    """Extract the fields we care about into a flat, CSV-friendly dict."""
    amount, currency = _price(item)
    return {
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "price": amount,
        "currency": currency,
        "condition": _condition(item),
        "year": item.get("year") or "",
        "make": _make(item),
        "model": item.get("model", ""),
        "seller": _seller(item),
        "listing_url": item.get("url", ""),
        "images": "|".join(_image_urls(item)),
        "description": (item.get("description") or "").replace("\n", " ").strip(),
    }


# --- Scraping ---------------------------------------------------------------
def fetch_page(client, query: str, page: int) -> list[dict]:
    """Run the actor for one (query, page) and return the raw items."""
    run_input = {
        "searchQuery": query,
        "maxItems": PER_PAGE,
        "perPage": PER_PAGE,
        "page": page,
        "categorySlug": CATEGORY_SLUG,
        "condition": "",
    }
    try:
        run = client.actor(ACTOR_ID).call(run_input=run_input)
        return list(client.dataset(run.default_dataset_id).iterate_items())
    except Exception as exc:
        print(f"  [warn] query '{query}' page {page} failed: {exc}")
        return []


def main():
    if not API_TOKEN:
        raise SystemExit("Set the APIFY_TOKEN env var (export APIFY_TOKEN=...) before running.")
    try:
        from apify_client import ApifyClient
    except ImportError:
        raise SystemExit(
            "apify-client is not installed. Install the scraping extra with:\n"
            '    pip install -e ".[scrape]"'
        )

    client = ApifyClient(API_TOKEN)
    Path("data/json").mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    all_items: list[dict] = []      # raw, de-duplicated
    per_query: dict[str, int] = {}  # new (unique) items contributed per query

    for query in SEARCH_QUERIES:
        print(f"\nQuery: '{query}'")
        new_for_query = 0

        for page in range(1, MAX_PAGES + 1):
            print(f"  page {page} — {len(all_items)} unique so far", end="\r")
            items = fetch_page(client, query, page)

            if not items:
                print(f"\n  no results on page {page}, next query")
                break

            new_this_page = 0
            for item in items:
                item_id = str(item.get("id", ""))
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                all_items.append(item)
                new_this_page += 1

            new_for_query += new_this_page
            if new_this_page == 0:
                print(f"\n  page {page} all duplicates, next query")
                break

            time.sleep(REQUEST_DELAY)

        per_query[query] = new_for_query
        print(f"  '{query}': +{new_for_query} unique listings")

    # --- Write outputs ------------------------------------------------------
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = Path("data/json") / f"scraped_{stamp}.json"
    csv_path = Path("data/strats.csv")

    json_path.write_text(json.dumps(all_items, indent=2))

    with csv_path.open("w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in all_items:
            writer.writerow(flatten(item))

    print(f"\nDone. {len(all_items)} unique listings saved.")
    print(f"  Raw JSON: {json_path}")
    print(f"  CSV:      {csv_path}")

    print("\nUnique listings per query:")
    for query, count in sorted(per_query.items(), key=lambda x: -x[1]):
        print(f"  {query:24s} {count}")

    print("\nTop makes found:")
    makes: dict[str, int] = {}
    for item in all_items:
        m = _make(item)
        makes[m] = makes.get(m, 0) + 1
    for make, count in sorted(makes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {make or '(unknown)':30s} {count}")


if __name__ == "__main__":
    main()
