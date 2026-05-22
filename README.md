I had Claude push for me so this is stuff it said down here 


⏺ Everything is clean. Here's the full pipeline:

  ---
  What was built

  3 scripts, run in order:

  1. python prepare_dataset.py

  Downloads ~770 images into data/images_labeled/american|mexican|japanese/. Labels are extracted from title + model + description using keyword
  matching. 302 listings labeled (198 dropped as ambiguous), estimated ~3 min to download.

  2. python train.py

  - Model: EfficientNet-B0 pretrained on ImageNet, last layer replaced for 3 classes
  - Split: 70/15/15 stratified train/val/test
  - Imbalance fix: WeightedRandomSampler so American (171) doesn't dominate Mexican (60) and Japanese (71)
  - Augmentation: random flip, color jitter, rotation
  - Scheduler: cosine annealing LR over 20 epochs
  - Device: auto-detects MPS (you have Apple Silicon)
  - Saves models/best_model.pt + results/confusion_matrix.png + training curves

  3. python inference.py guitar.jpg

  Loads best checkpoint, prints prediction + probability bar for each class.

  ---
  Start with:
  python prepare_dataset.py   # ~3 min download
  python train.py             # ~15-30 min on MPS
