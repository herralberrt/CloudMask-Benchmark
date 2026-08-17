# CloudMask Benchmark

## Comparing Machine Learning Models for Satellite Cloud Detection

**Project Overview:**  
This project trains and compares different ML approaches (Random Forest and U-Net) for cloud detection in satellite imagery, using the CloudSEN12 dataset with Sentinel-2 multispectral data.

---

## Project Status

### Week 1: ✓ Dataset Exploration & Configuration (COMPLETE)

**Completed Tasks:**
- ✓ Dataset analysis: CloudSEN12 (Sentinel-2 L2A)
  - 50,247 total samples
  - 14 raster bands in L2A: 12 optical bands + AOT + WVP
  - 4 cloud classes: Clear sky, Thick cloud, Thin cloud, Cloud shadow
- ✓ Band selection: RGB + NIR (4 channels for optimal cloud discrimination)
- ✓ Preprocessing pipeline: Normalization by 10000 → [0, 1] range
- ✓ Data split: 60% train / 20% val / 20% test (5,024 samples subset)
- ✓ Visualization: Sample images and class distribution analysis

**Artifacts Generated:**
- `notebooks/01_cloudsen12_exploration.ipynb` - Complete Week 1 exploration
- `src/preprocessing.py` - Preprocessing utilities
- `src/config.py` - Centralized configuration
- `outputs/preprocessing_config.json` - Normalization pipeline config
- `outputs/split_info.json` - Train/val/test indices
- `outputs/class_info.json` - Class labels and distribution
- `outputs/figures/` - Dataset visualizations

**Key Decisions:**
| Parameter | Decision |
|-----------|----------|
| Dataset | CloudSEN12 (Sentinel-2 L2A) |
| Subset | 10% for Week 1 (5,024 samples) |
| Input Bands | Red (B4), Green (B3), Blue (B2), NIR (B8) |
| Normalization | x / 10000 → [0, 1] |
| Target Format | Multiclass segmentation (4 classes) |
| Evaluation Metrics | Precision, Recall, F1-score, IoU |

---

### Week 2: Baseline & Random Forest (INITIAL RUN COMPLETE)
- [x] Load CloudSEN12 L2A and the matching operational-mask dataset
- [x] Use Sen2Cor as the operational baseline
- [x] Train a Random Forest classifier with spectral features
- [x] Evaluate and compare on held-out scenes

The L2A asset does not contain an SCL band. The operational comparison uses
`cloudsen12-extra/cloudmask_sen2cor`, while label value `99` is treated as
no-data. The reproducible Week 2 pipeline is in `src/week2_pipeline.py`.

### Week 3-5: U-Net, Comparison & Analysis
- [ ] U-Net segmentation model
- [ ] Complete model evaluation
- [ ] Detailed failure analysis
- [ ] Final report and presentation

---

## Repository Structure

```
CloudMask-Benchmark/
├── notebooks/
│   ├── 01_cloudsen12_exploration.ipynb        [Week 1 ✓]
│   └── 02_baseline_and_rf.ipynb               [exploratory draft]
├── src/
│   ├── config.py            # Centralized configuration
│   ├── preprocessing.py     # Data preprocessing utilities
│   └── week2_pipeline.py    # Canonical Week 2 baseline and RF pipeline
├── data/
│   ├── raw/                 # Reserved for optional local downloads
│   └── processed/           # Reserved for generated local data
├── outputs/
│   ├── figures/             # Week 1 visualizations
│   ├── class_info.json      # Class definitions and distribution
│   ├── split_info.json      # Reproducible train/validation/test split
│   ├── week2_results.json   # RF and Sen2Cor metrics
│   ├── rf_feature_importance.json
│   └── random_forest.joblib
├── requirements.txt
└── README.md
```

Planned files for Weeks 3–5, such as the U-Net model, comparison notebook,
and final analysis notebook, will be added when those weeks are implemented.

---

## Setup & Usage

### Installation
```bash
pip install -r requirements.txt
```

### Run the Week 2 experiment
```bash
python src/week2_pipeline.py --train-scenes 10 --validation-scenes 5 --max-pixels-per-scene 2000
```

This streams a small, deterministic subset from CloudSEN12 and writes
`outputs/week2_results.json`, `outputs/random_forest.joblib`, and
`outputs/rf_feature_importance.json`. Increase `--train-scenes` and
`--validation-scenes` when a larger experiment is practical.

`notebooks/02_baseline_and_rf.ipynb` contains the original exploratory Week 2
draft. It assumes an SCL band inside the L2A raster and is not the canonical
pipeline for this dataset; use `src/week2_pipeline.py` for reproducible runs.

### Run Week 1 Exploration
```bash
cd notebooks
jupyter notebook 01_cloudsen12_exploration.ipynb
```

---

## Technologies Used
- Python 3.x
- PyTorch / TensorFlow (for U-Net)
- scikit-learn (for Random Forest)
- Rasterio (for GeoTIFF I/O)
- NumPy / Pandas (data processing)
- Matplotlib (visualization)
