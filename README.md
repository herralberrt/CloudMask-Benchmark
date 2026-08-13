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
  - 12 spectral bands (11 data + 1 SCL)
  - 4 cloud classes: Clear sky, Thin cloud, Thick cloud, Cloud shadow
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

### Week 2: Baseline & Random Forest (IN PROGRESS)
- [ ] Extract Sentinel-2 SCL baseline
- [ ] Implement Random Forest classifier
- [ ] Feature engineering (spectral indices)
- [ ] Evaluate and compare

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
│   ├── 02_baseline_and_rf.ipynb               [Week 2]
│   ├── 03_unet_training.ipynb                 [Week 3]
│   ├── 04_model_comparison.ipynb              [Week 4]
│   └── 05_final_analysis.ipynb                [Week 5]
├── src/
│   ├── preprocessing.py      # Data preprocessing utilities
│   ├── config.py            # Configuration management
│   ├── models.py            # Model definitions (Week 2+)
│   └── metrics.py           # Evaluation metrics (Week 2+)
├── data/
│   ├── raw/                 # CloudSEN12 data (via tacoreader)
│   └── processed/           # Preprocessed data (Week 2+)
├── outputs/
│   ├── figures/             # Visualizations
│   └── *.json              # Configuration files
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Installation
```bash
pip install -r requirements.txt
```

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
