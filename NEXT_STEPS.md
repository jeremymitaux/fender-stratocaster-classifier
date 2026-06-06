# Next Steps to Complete the Project

**Direction (updated):** pivot from the Fender *origin* classifier
(American/Japanese/Mexican) to a **guitar make/model classifier** over a fixed set
of popular models, with labels read from Reverb's structured metadata. This is
closer to the original proposal ("identify make, model"), is a genuinely learnable
computer-vision problem, and uses cleaner labels.

Items are ordered roughly by impact. Tags: **[data]** = you can do this while
updating the dataset; **[code]** = pipeline/model changes; **[writeup]** = docs/notebooks.

---

## ⭐ Current status — where we are

**The 2-class project is complete.** Decision (2026-06-05): ship Stratocaster vs.
Telecaster as the final scope rather than buying more Apify credits for the other
4 classes. The full pipeline — scrape → metadata label → group-aware split →
train → evaluate → writeup — is built, run, and committed.

**🏁 Final result:** EfficientNet-B0, **95.2% test accuracy** (933 held-out
images), macro-F1 0.95, vs. a 64.2% majority-class baseline. Per-class F1:
stratocaster 0.96, telecaster 0.93. Artifacts: `models/best_model.pt`,
`models/class_names.json`, `results/{confusion_matrix,training_curves,example_predictions}.png`.

**✅ Done:**
1. **Scraper run** (§2) — `strat-scrape` pulled real data; labeling kept **1333 listings:
   862 stratocaster + 471 telecaster**. The other classes were cut off when **Apify
   credits ran out** (potentially need les_paul / sg / es_335 / jazzmaster_jaguar — see §1).
   Might just make this a 2 type classification if I don't want to buy more API credits tbd. 
2. **Labeler** (§3) — `prepare.py` confirmed working: kept 1333, dropped 278
   (104 too-cheap, 102 parts, 71 wrong-brand, 1 ambiguous). Also fixed so it only
   creates folders for classes that actually have listings (no phantom empty classes).
3. **Group-aware split** (§4) — ✅ **DONE.** `build_splits` now splits over *listings*
   (grouped on the `<listing_id>` filename prefix), stratified per class, with an
   assertion that no listing leaks across train/val/test. Validated: zero overlap.
4. **Dataset statistics** — new `stats.py` (`python -m strat_classifier.stats`) reports
   per-class listings/images + the split breakdown, writes `results/dataset_stats.md`.

**✅ Done (this session, 2026-06-05):**
1. **Image download** — `strat-prepare` finished all 1333 listings → 6195 images
   (3982 strat / 2213 tele). Costs **no API credits** (pulls from Reverb's CDN,
   not Apify). `python -m strat_classifier.stats` regenerated.
2. **[code] Training run** (§5): `strat-train` on the 2-class set → 95.2% test acc.
   `best_model.pt` / `class_names.json` / plots regenerated (the old 3-class
   origin model is gone).
3. **[code] Pruned to 2 classes** — `MODEL_PATTERNS` / `ALLOWED_BRANDS` in
   `prepare.py` and `SEARCH_QUERIES` in `scrape.py` now hold only Strat/Tele.
4. **[code] Image normalization** (§8) — caught that `ImageFolder` was silently
   dropping 65 `.heic` files (13 listings). All images re-encoded to JPEG;
   `prepare.py` now normalizes on download; `pillow-heif` added to extras.
5. **[writeup] README** (§6) results/stats filled + **evaluation notebook** (§7)
   executed with outputs.

**⏸️ Deferred (only if scope ever expands):**
- **[data] More scraping** (§1/§2): top up Apify credits and scrape the remaining
  4 classes (les_paul / sg / es_335 / jazzmaster_jaguar) to reach the full
  make/model set. The pipeline already supports N classes — just add the queries
  back to `scrape.py` and the patterns to `prepare.py`.

---

## 0. TL;DR — why this pivot, and the biggest gaps

**Why make/model > origin:**
- **Real visual signal.** Stratocaster vs Telecaster vs Les Paul vs SG have
  distinct body shapes, headstocks, and pickup layouts a CNN can actually learn.
  American-vs-Mexican Strats look identical, so the old classifier was likely
  learning photo style/background, not the guitar.
- **Clean labels.** `make` / `model` come straight from Reverb metadata — no
  fragile keyword guessing like the origin labeler.
- **Better fit for the flipping goal.** "What is this guitar?" is make/model.

**What we are dropping:** country-of-origin, and **year** (only 44% of listings had
a usable year, and they were messy free-text like `"2000s?"`).

**Biggest gaps at the time of the pivot — all now resolved** (see the
⭐ Current status block above):
1. ~~README is still the proposal~~ → final README written with every required
   section; the proposal lives in `PROPOSAL.md`.
2. ~~No Evaluation Notebook~~ → `notebooks/evaluation.ipynb` added and executed.
3. ~~Train/val/test split leaks images of the same listing~~ → group-aware split
   on `listing_id` with a no-leak assertion (§4).
4. ~~Dataset is small *and* duplicated~~ → full scrape → **1,333 unique listings /
   6,195 images** across the 2 classes.

---

## 1. Define the class set [data + code] — ✅ encoded

The class set is now encoded in `scraping/prepare.py` (`MODEL_PATTERNS`) and the
queries in `scraping/scrape.py` (`SEARCH_QUERIES`). Current 6 classes:
`stratocaster`, `telecaster`, `les_paul`, `sg`, `es_335`, `jazzmaster_jaguar`.
Add/remove by editing those two lists. Original guidance kept below for reference.

Pick a **tractable, visually-distinct set** of ~6–10 models, each with enough
listings to learn from. Suggested starting set (all strong, recognizable silhouettes):

| Class label            | Notes |
|------------------------|-------|
| `stratocaster`         | Fender double-cutaway |
| `telecaster`           | Fender single-cut, slab body |
| `les_paul`             | Gibson/Epiphone single-cut |
| `sg`                   | Gibson double horn |
| `es_335`               | semi-hollow, f-holes |
| `jazzmaster_jaguar`    | offset body (group these two — similar shape) |
| `prs_custom`           | optional, if you can get volume |
| `ibanez_rg`            | optional, superstrat |

Guidance:
- [ ] **Classify by model/body-shape, not brand.** An Epiphone Les Paul and a
      Gibson Les Paul look the same — brand isn't visually separable, model is.
- [ ] Start with the **5–6 classes you can get ≥300–500 listings each** for; add
      more only if the data supports it. An imbalanced 10-class set with 30 examples
      in the tail will hurt more than help.
- [ ] Decide the set *before* the big scrape so your search queries target it.

---

## 2. Collect the data right [data] — ✅ scraper rewritten, run it

`scrape.py` now does per-model queries, `categorySlug=electric-guitars`, dedupe by
`id`, and paginate-until-dry, writing `data/json/scraped_<timestamp>.json` (which
`prepare.py` reads) + `data/strats.csv`.

**To run:**
```bash
pip install -e ".[scrape]"     # if apify-client isn't installed
export APIFY_TOKEN=...
strat-scrape
```
- [ ] **Test run first:** in the Apify console, run one query with `maxItems: 50`,
      `categorySlug: "electric-guitars"`, and confirm the parts/backplates are gone.
- [ ] Then `strat-scrape` for the full multi-model pull.
- [ ] **Target ≈300–500 unique listings per class** (≈2,000–3,000 total), reasonably
      balanced — rarer classes need more search depth (raise `MAX_PAGES` / add queries).
- [ ] Commit the new `data/json/scraped_*.json` so the dataset is reproducible.

Knobs in `scrape.py`: `SEARCH_QUERIES`, `CATEGORY_SLUG`, `PER_PAGE`, `MAX_PAGES`,
`REQUEST_DELAY`. Images downloaded per listing is set in `prepare.py`
(`MAX_IMAGES_PER_LISTING = 5`).

---

## 3. Relabel from metadata, not keywords [code] — ✅ done

`prepare.py` now labels by **make/model** from metadata and filters out junk:
- matches canonical models via `MODEL_PATTERNS` (drops 0-match and >1-match listings);
- requires an allowed brand (`ALLOWED_BRANDS`) → drops part-sellers / "Unbranded";
- drops parts via `PART_KEYWORDS` (backplate, tremolo arm, pickguard, …);
- drops listings under `MIN_PRICE` ($150);
- writes `data/images_labeled/<model>/<listing_id>_<idx>.<ext>` (the `<listing_id>_`
  prefix is what the grouped split in §4 keys on);
- `--dry-run` prints the class breakdown + dropped-reason counts without downloading.

Still worth doing after the real scrape:
- [ ] **Audit ~30–50 labeled images per class by eye** and report the error rate in
      the README (supports the "labels are reasonable" judgment).
- [ ] Tune `MODEL_PATTERNS` / `PART_KEYWORDS` / `MIN_PRICE` if the dry-run shows
      mislabels or too-aggressive drops.

---

## 4. Fix the train/val/test split — group by listing [code] — ✅ DONE

`build_splits()` used to split **individual images**, but multiple photos come
from the **same listing** (`1607482_0/_1/_2` = same guitar, same background), so
near-duplicates could land in train *and* test and inflate accuracy.

- [x] **Group-aware split keyed on the listing id** (`listing_id()` parses the
      filename prefix before the trailing `_<idx>`). Within each class, listings are
      shuffled (seeded) and partitioned by the val/test fractions — so the split is
      both group-aware *and* stratified. An assertion guards against any listing
      appearing in more than one split.
- [x] Kept the `WeightedRandomSampler` for residual class imbalance.

---

## 5. Train & report (multiclass) [code + writeup]

The model code barely changes — EfficientNet-B0 with an N-way head (N = number of
classes). `train.py` already handles arbitrary `num_classes`.

- [ ] **First run is 2-class** (stratocaster vs telecaster) until the remaining
      classes are scraped. Expect high accuracy — call this out as a not-yet-final
      result, not the headline number.
- [ ] Re-train on the new dataset; the prior origin results (79%) are **obsolete** —
      regenerate `results/confusion_matrix.png` and `results/training_curves.png`.
- [ ] Report **overall accuracy + per-class precision/recall/F1** (the existing
      `classification_report` already does this) and the confusion matrix.
- [ ] Add a **baseline** for honesty: majority-class accuracy, so reviewers see the
      lift over guessing.
- [ ] State the **test-set size** explicitly and note the confidence is wider when
      it's small.
- [ ] Address **overfitting** if it appears (last run: train ~98% vs val ~79%):
      early stopping on val accuracy, freeze backbone then unfreeze, stronger
      augmentation (`RandomResizedCrop`, perspective), `label_smoothing=0.1`.

---

## 6. Rewrite the README [writeup, required — Completeness]

The current `README.md` is the 300-word proposal. **Move it to `PROPOSAL.md`** (so the
make/model→origin→make/model story is traceable), then write the final README (~800
words) with every rubric section:

- [ ] **Project Purpose** — classify a guitar listing photo into one of N models.
- [ ] **Scope vs proposal** — explain you returned to the proposal's make/model goal,
      narrowed to a fixed model set, and dropped year (data too sparse/messy). The
      "Novelty" criterion grades you against the proposal + addressing feedback, so
      make the reasoning explicit.
- [ ] **Dataset & how you built it** — Reverb via Apify → dedupe → metadata labeling →
      image download. Report final per-class unique-listing and image counts.
- [ ] **How to train** — `pip install -e .` then `strat-train` (+ key flags).
- [ ] **Results** — accuracy, per-class metrics, embedded confusion matrix + curves.
- [ ] **Metrics** — map the proposal's "how accurately it ascribes a model" to the
      accuracy + per-class F1 you actually report.
- [ ] **Prediction visualization** — example images with predicted model + confidence.
- [ ] **Limitations & use** — dataset size, label/keyword caveats, look-alike models
      (e.g. Epiphone vs Gibson Les Paul), intended use as a flip-finding assistant.
- [ ] **Data & weights locations** — paths to `data/images_labeled/`, `data/json/`,
      and `models/best_model.pt` / `last_model.pt` (committed in the repo — say so).

---

## 7. Add an Evaluation Notebook [writeup, required — Completeness]

Create `notebooks/evaluation.ipynb` that:

- [ ] Imports from the package: `from strat_classifier.inference import load_model, predict`
      and `from strat_classifier.data import make_dataloaders`.
- [ ] Loads `models/best_model.pt` + `models/class_names.json`.
- [ ] Runs the model over the **test split**, prints `classification_report`, and
      regenerates the confusion matrix inline.
- [ ] Shows a grid of example images with predicted-vs-true labels + confidence bars
      (doubles as the README prediction visualization).

The existing `milestone_dataloader.ipynb` satisfies the **Data Demo** requirement
(it imports `make_dataloaders` from the dataset module — good).

---


## 8. Repo / reproducibility cleanup [code]

- [x] **`scrape.py` schema fixed** — now reads the real Apify export (`images` list,
      `brand`, `price` dict) and writes `data/json/scraped_*.json` + a real CSV.
- [x] Deleted the stale `data/strats_raw.jsonl` (no longer in the repo).
- [x] **Normalize images to JPEG in `prepare.py`** — `download_image` re-encodes
      every image to RGB JPEG, so there are no stray HEIC/WebP/GIF files that
      `ImageFolder` would skip. The on-disk dataset is 100% `.jpg`.
- [x] **Rubric file mapping** documented in the README's Repository-layout table
      (`model.py` / `data.py` / `train.py` = `models.py` / `dataset.py` / `train_models.py`).
- [x] Removed the dead `scraping/download.py` (legacy unlabeled downloader,
      superseded by `prepare.py`).
- [ ] Confirm a **fresh-clone install** works: `pip install -e .` then
      `strat-train --epochs 1` on the committed dataset.

---

## Suggested order of work

See the **⭐ Current status** block at the top. In short: pipeline is built and
validated on 2 classes — what's left is the first training run (§5), scraping the
remaining 4 classes once credits are topped up (§1/§2), and the writeup (§6, §7) +
cleanup (§8).

✅ Done: class set (§1), scraper rewrite + run → 1333 listings/2 classes (§2),
metadata labeler + phantom-folder fix (§3), **group-aware split (§4)**,
dataset statistics (`stats.py`), scrape schema (§8).
