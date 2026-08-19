"""Shared CloudSEN12 data access, labeling, and evaluation utilities.

This module centralizes everything that both the Random Forest and the
U-Net pipelines need: reading scenes from the remote CloudSEN12 datasets,
normalizing labels, deriving spectral features, extracting patches, and
computing metrics. Keeping this logic in one place avoids duplicating it
across training scripts.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio
import tacoreader.v1 as tacoreader
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

DATASET_NAME = "tacofoundation:cloudsen12-l2a"
EXTRA_DATASET_NAME = "tacofoundation:cloudsen12-extra"

# Red (B4), Green (B3), Blue (B2), NIR (B8) out of the 14 raw L2A bands.
BAND_INDICES = [3, 2, 1, 7]
FEATURE_NAMES = ["B4", "B3", "B2", "B8", "NDVI", "brightness", "greenness"]

NUM_CLASSES = 4
CLASS_NAMES = ["Clear sky", "Thick cloud", "Thin cloud", "Cloud shadow"]

NO_DATA_VALUE = 99  # CloudSEN12 target value meaning "no label"
IGNORE_INDEX = 255  # Sentinel used for no-data pixels in labels/loss
PATCH_SIZE = 128
SNOW_SCL_VALUE = 11  # Sen2Cor Scene Classification Layer: snow/ice


def load_datasets():
    """Load the CloudSEN12 L2A scenes and the matching operational-mask dataset."""
    return tacoreader.load(DATASET_NAME), tacoreader.load(EXTRA_DATASET_NAME)


def cloudsen12_labels(target: np.ndarray) -> np.ndarray:
    """Normalize CloudSEN12 labels: 99 (no-data) becomes -1, 0-3 stay valid."""
    labels = target.reshape(-1).astype(np.int16)
    labels[labels == NO_DATA_VALUE] = -1
    return labels


def sen2cor_labels(scl: np.ndarray) -> np.ndarray:
    """Map raw Sen2Cor SCL classes to CloudSEN12's clear/thick/thin/shadow classes."""
    scl = scl.reshape(-1)
    mapped = np.zeros_like(scl, dtype=np.int8)
    mapped[np.isin(scl, [8, 9])] = 1  # Thick cloud
    mapped[scl == 10] = 2  # Thin cloud
    mapped[scl == 3] = 3  # Cloud shadow
    return mapped


def read_scene(l2a_dataset, extra_dataset, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one scene: normalized 4-band image, 2D labels, and 2D Sen2Cor baseline."""
    sample = l2a_dataset.read(int(index))
    image_row = sample.index[sample["tortilla:id"] == "s2l2a"][0]
    target_row = sample.index[sample["tortilla:id"] == "target"][0]
    with rasterio.open(sample.read(int(image_row))) as src:
        image = src.read()
    with rasterio.open(sample.read(int(target_row))) as src:
        target = src.read(1)

    extra = extra_dataset.read(int(index))
    baseline_row = extra.index[extra["tortilla:id"] == "cloudmask_sen2cor"][0]
    with rasterio.open(extra.read(int(baseline_row))) as src:
        baseline = src.read(1)

    image = image[BAND_INDICES].astype(np.float32) / 10000.0
    labels = cloudsen12_labels(target).astype(np.int16).reshape(target.shape)
    labels[labels < 0] = IGNORE_INDEX
    return image, labels, sen2cor_labels(baseline).reshape(target.shape)


def read_raw_sen2cor(extra_dataset, index: int) -> np.ndarray:
    """Read the raw Sen2Cor SCL values (0-11), before remapping to CloudSEN12 classes."""
    extra = extra_dataset.read(int(index))
    baseline_row = extra.index[extra["tortilla:id"] == "cloudmask_sen2cor"][0]
    with rasterio.open(extra.read(int(baseline_row))) as src:
        return src.read(1)


def spectral_features(image: np.ndarray) -> np.ndarray:
    """Return RGB/NIR reflectance plus NDVI, brightness, and greenness per pixel.

    Args:
        image: normalized 4-band array of shape (4, H, W), as returned by `read_scene`.
    """
    pixels = image.transpose(1, 2, 0).reshape(-1, 4)
    red, green, blue, nir = pixels.T
    ndvi = (nir - red) / (nir + red + 1e-7)
    brightness = pixels.mean(axis=1)
    greenness = green / (red + green + blue + 1e-7)
    return np.column_stack((pixels, ndvi, brightness, greenness))


def extract_patches(
    image: np.ndarray,
    labels: np.ndarray,
    patch_size: int = PATCH_SIZE,
    max_patches: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract aligned non-overlapping patches, dropping all-no-data patches."""
    image_patches, label_patches = [], []
    height, width = labels.shape
    for top in range(0, height - patch_size + 1, patch_size):
        for left in range(0, width - patch_size + 1, patch_size):
            label_patch = labels[top : top + patch_size, left : left + patch_size]
            if np.all(label_patch == IGNORE_INDEX):
                continue
            image_patches.append(image[:, top : top + patch_size, left : left + patch_size])
            label_patches.append(label_patch)
    if max_patches is not None:
        image_patches = image_patches[:max_patches]
        label_patches = label_patches[:max_patches]
    return image_patches, label_patches


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute macro-averaged Precision, Recall, F1, and IoU over all four classes."""
    labels = np.arange(NUM_CLASSES)
    return {
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "iou": float(jaccard_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute Precision, Recall, F1, and IoU separately for each of the four classes."""
    labels = np.arange(NUM_CLASSES)
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    iou = jaccard_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        CLASS_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "iou": float(iou[i]),
        }
        for i in range(NUM_CLASSES)
    }
