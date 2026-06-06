# Dataset statistics

Source: `data/images_labeled`

## Per-class totals

| Class | Listings | Images | Images/listing |
|---|---:|---:|---:|
| stratocaster | 862 | 3982 | 4.6 |
| telecaster | 471 | 2213 | 4.7 |
| **Total** | **1333** | **6195** | **4.6** |

## Group-aware split (by listing, seed=42)

All images of one listing stay in a single split (no leakage).

| Split | stratocaster | telecaster | Listings | Images |
|---|---:|---:|---:|---:|
| train | 604 | 329 | 933 | 4310 |
| val | 129 | 71 | 200 | 952 |
| test | 129 | 71 | 200 | 933 |
