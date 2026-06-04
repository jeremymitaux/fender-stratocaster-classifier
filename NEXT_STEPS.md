# Next Steps to Complete the Project

**Direction (updated):** pivot from the Fender *origin* classifier
(American/Japanese/Mexican) to a **guitar make/model classifier** over a fixed set
of popular models, with labels read from Reverb's structured metadata. This is
closer to the original proposal ("identify make, model"), is a genuinely learnable
computer-vision problem, and uses cleaner labels.

Items are ordered roughly by impact. Tags: **[data]** = you can do this while
updating the dataset; **[code]** = pipeline/model changes; **[writeup]** = docs/notebooks.

---

## ⭐ Tomorrow — ordered checklist

The data-collection pipeline is **already rewired** (see ✅ below). The fastest path
to a finished project from here:

1. **Scrape the new data** (§2). Do the 50-item test run in the Apify console first
   to confirm `categorySlug` filters, then `strat-scrape` for the full multi-model pull.
2. **Build the dataset:** `strat-prepare --dry-run` to check the class breakdown,
   then `strat-prepare` to download images. *(labeler is done — §3)*
3. **[code] Group-aware split** in `data.py` (§4) — the one remaining pipeline fix.
4. **[code] Re-train** and regenerate results (§5): `strat-train`.
5. **[writeup] README rewrite** (§6) + **evaluation notebook** (§7).
6. **[code] Repo cleanup** (§8).

**Already done for you:**
- ✅ `scrape.py` — per-model queries, `categorySlug=electric-guitars`, paginate-until-dry,
  dedupe by id, real Apify schema (§2).
- ✅ `prepare.py` — labels by **make/model** from metadata, parts/brand/price filtering,
  `--dry-run` mode (§3). Dry-run on the *current* data kept 136 Stratocasters and dropped
  134 junk listings (50 wrong-brand, 47 parts, 37 under $150).

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

**Biggest remaining gaps (unchanged by the pivot):**
1. README is still the proposal — none of the required final sections exist.
2. No Evaluation Notebook (rubric requires one separate from the data demo).
3. Train/val/test split leaks images of the same listing across splits.
4. Dataset is small *and* duplicated — your last pull was 500 rows but only
   **270 unique listings**.

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

## 4. Fix the train/val/test split — group by listing [code, required]

`build_splits()` in `data.py` splits **individual images**, but multiple photos come
from the **same listing** (`1607482_0/_1/_2` = same guitar, same background). Those
near-duplicates can land in train *and* test, inflating accuracy.

- [ ] Switch to a **group-aware split keyed on the listing id** (the filename prefix
      before `_`) using `StratifiedGroupKFold` or `GroupShuffleSplit`, so all images
      of one listing stay in a single split.
- [ ] Keep the `WeightedRandomSampler` for any residual class imbalance.

---

## 5. Train & report (multiclass) [code + writeup]

The model code barely changes — EfficientNet-B0 with an N-way head (N = number of
classes). `train.py` already handles arbitrary `num_classes`.

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
- [ ] Delete the stale `data/strats_raw.jsonl` (1 line) once the new scrape lands.
- [ ] Normalize images to JPEG/PNG in `prepare.py` (one stray `.gif` exists; animated
      GIFs collapse to the first frame).
- [ ] Map each rubric file requirement to your file (`model.py`, `data.py`,
      `train.py` are valid "equivalents" of `models.py` / `dataset.py` /
      `train_models.py` — just say so in the README).
- [ ] Confirm a **fresh-clone install** works: `pip install -e .` then
      `strat-train --epochs 1` on the committed dataset.

---

## Suggested order of work

See the **⭐ Tomorrow** checklist at the top. In short: scrape (§2) → prepare (§3,
done — just run it) → group-aware split (§4, only remaining pipeline code) →
re-train (§5) → README + eval notebook (§6, §7) → cleanup (§8).

✅ Done: class set (§1), scraper rewrite (§2), metadata labeler (§3), scrape schema (§8).
