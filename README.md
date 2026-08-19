# CloudMask Benchmark

## Comparing Machine Learning Models for Satellite Cloud Detection

**Project overview:**
This project trains and compares two machine learning approaches — a Random
Forest on spectral features and a compact U-Net for segmentation — for cloud
detection on Sentinel-2 imagery from the CloudSEN12 dataset. Both models are
evaluated against the Sen2Cor operational baseline on the same held-out test
scenes, including difficult cases such as thin cloud, snow, and bright
surfaces.

The CloudSEN12 scenes and the operational Sen2Cor mask are streamed on demand
from the remote TACO-format datasets via `tacoreader`; the project does not
keep a local copy of the imagery, so there is no `data/` directory.

---

## Repository Structure

```
CloudMask-Benchmark/
├── notebooks/
│   └── cloud_masking_report.ipynb   # Single narrative report: exploration + comparison
├── src/
│   ├── config.py                  # Centralized configuration
│   ├── preprocessing.py           # CloudSEN12 band/label preprocessing utilities
│   ├── cloudsen12_data.py         # Shared data access, labeling, features, metrics
│   ├── unet_model.py              # Compact U-Net architecture
│   ├── train_random_forest.py     # Trains the Random Forest, compares vs. Sen2Cor
│   ├── train_unet.py              # Trains the U-Net, compares vs. RF and Sen2Cor
│   └── evaluate_models.py         # Final comparison on held-out test scenes
├── outputs/
│   ├── figures/                          # Dataset visualizations and comparison figures
│   ├── class_info.json                   # Class definitions and distribution
│   ├── split_info.json                   # Reproducible train/validation/test split
│   ├── preprocessing_config.json         # Normalization pipeline config
│   ├── random_forest_model.joblib        # Trained Random Forest
│   ├── random_forest_results.json        # RF vs. Sen2Cor (validation scenes)
│   ├── random_forest_feature_importance.json
│   ├── unet_best.pt                      # Best U-Net checkpoint (by validation loss)
│   ├── unet_training_results.json        # U-Net vs. RF vs. Sen2Cor (validation scenes)
│   ├── unet_validation_predictions.npz   # Saved validation patches + predictions
│   └── model_comparison_results.json     # Final test-set comparison and difficult cases
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA is used automatically for U-Net training/inference when available, with
a CPU fallback.

---

## Pipeline

The project is organized as three sequential scripts, each reading the
previous stage's saved outputs. All of them accept `--help` for their CLI
options.

### 1. Train the Random Forest baseline comparison

```bash
python src/train_random_forest.py --train-scenes 10 --validation-scenes 5 --max-pixels-per-scene 2000
```

Streams a deterministic pixel sample from CloudSEN12, trains a Random Forest
on RGB + NIR reflectance plus NDVI/brightness/greenness, and evaluates it
against the Sen2Cor baseline on validation scenes. Writes
`random_forest_model.joblib`, `random_forest_results.json`, and
`random_forest_feature_importance.json` to `outputs/`.

### 2. Train the U-Net segmentation model

```bash
python src/train_unet.py --train-scenes 20 --validation-scenes 8 --epochs 8 --max-patches-per-scene 16 --batch-size 8
```

Extracts 128x128 patches (`--max-patches-per-scene 16` uses every patch in
each 512x512 scene), trains a compact U-Net while monitoring the
train/validation loss, saves the best checkpoint, and compares it with the
Random Forest and Sen2Cor on validation scenes. Writes `unet_best.pt`,
`unet_training_results.json`, and `unet_validation_predictions.npz`.

### 3. Evaluate and compare on held-out test scenes

```bash
python src/evaluate_models.py --test-scenes 50 --num-figures 4
```

Loads the frozen Random Forest and U-Net, evaluates both alongside Sen2Cor on
test scenes never used for training or validation, computes overall and
per-class Precision/Recall/F1/IoU, analyzes difficult cases (thin cloud,
snow, bright surfaces, cloud boundaries), and saves qualitative comparison
figures to `outputs/figures/model_comparison/`. Writes
`model_comparison_results.json`. A larger `--test-scenes` gives enough
pixels for the difficult-case statistics (thin cloud, snow, bright surfaces)
to be meaningful rather than near-empty.

### Report notebook

```bash
cd notebooks
jupyter notebook cloud_masking_report.ipynb
```

`cloud_masking_report.ipynb` is the single narrative report: it explores the
CloudSEN12 dataset, then loads and presents the results produced by the three
scripts above (no training happens in the notebook itself).

---

## Current Results

The Random Forest and U-Net are trained on a modest scene budget (10 and 20
training scenes respectively — the U-Net now uses all 16 patches per
training scene). The numbers below are from the **final comparison** on 50
held-out test scenes (13.1M pixels) — large enough that thin cloud, snow,
and bright-surface pixels appear in meaningful quantities.

| Model | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| Sen2Cor (operational baseline) | 0.664 | 0.549 | 0.574 | 0.442 |
| Random Forest | 0.448 | 0.431 | 0.433 | 0.321 |
| U-Net | 0.421 | 0.413 | 0.408 | 0.295 |

**Per-class IoU:**

| Class | Sen2Cor | Random Forest | U-Net |
|---|---:|---:|---:|
| Clear sky | 0.689 | 0.575 | 0.495 |
| Thick cloud | 0.646 | 0.526 | 0.518 |
| Thin cloud | 0.128 | 0.0001 | 0.000 |
| Cloud shadow | 0.305 | 0.184 | 0.169 |

**Difficult cases** (agreement rate with ground truth):

| Case | Pixels | Sen2Cor | Random Forest | U-Net |
|---|---:|---:|---:|---:|
| Thin cloud | 242,975 | 23.2% | 0.03% | 0.0% |
| Snow | 747,217 | 73.4% | 17.8% | 18.0% |
| Bright surfaces | 555,298 | 98.1% | 0.03% | 0.6% |
| Cloud boundaries | 532,122 | 37.8% | 40.2% | 35.1% |

Sen2Cor still wins overall, but the comparison is now fair and the failure
modes are clear:
- **Both trained models mislabel snow and bright surfaces as cloud almost
  every time** (false-cloud rate ≈ 99% for Random Forest and U-Net alike),
  while Sen2Cor gets these right 98%+ of the time. This is exactly the
  classic weakness of models trained mainly on spectral signal — snow and
  bright bare ground reflect similarly to cloud.
- **Thin cloud is the hardest class for every method.** Sen2Cor is weak
  (IoU 0.128) but still the best of the three; Random Forest and U-Net are
  essentially unable to detect it (IoU ≈ 0).
- **U-Net is competitive with Random Forest on thick cloud** (IoU 0.518 vs.
  0.526) and close on cloud shadow, even though it trained on fewer labeled
  pixels overall — a reasonable segmentation signal once it stopped
  collapsing to a single class.
- **All three models struggle at cloud boundaries** (35–40% agreement),
  confirming edge pixels are the hardest cases regardless of method.

None of the trained models beat the Sen2Cor operational mask yet. Both would
need more spectral/contextual information (e.g. SWIR bands, which
distinguish snow from cloud far better than RGB+NIR) and more training data
to close the gap, particularly on thin cloud and snow/bright-surface
confusion.

---

## Key Decisions

| Parameter | Decision |
|-----------|----------|
| Dataset | CloudSEN12 L2A (`tacofoundation:cloudsen12-l2a`), streamed remotely |
| Subset | 10% of full 50,247 scenes = 5,024 scenes |
| Train/Val/Test split | 60% / 20% / 20%, randomized with seed=42 to avoid geographic bias |
| Operational baseline | Sen2Cor (`tacofoundation:cloudsen12-extra` / `cloudmask_sen2cor`) |
| Input bands | Red (B4), Green (B3), Blue (B2), NIR (B8) |
| Normalization | x / 10000 → [0, 1] |
| Classes | Clear sky, Thick cloud, Thin cloud, Cloud shadow |
| No-data handling | CloudSEN12 label value 99 is excluded from training/evaluation |
| Evaluation metrics | Precision, Recall, F1-score, IoU (macro and per-class) |

---

## Technologies Used
- Python
- PyTorch (U-Net)
- scikit-learn (Random Forest, evaluation metrics)
- Rasterio (GeoTIFF I/O)
- tacoreader (remote CloudSEN12 access)
- NumPy / Pandas (data processing)
- Matplotlib (visualization)
