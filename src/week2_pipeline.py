"""Week 2 CloudSEN12 baseline and Random Forest experiment."""

import json
import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import rasterio
import tacoreader.v1 as tacoreader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)


DATASET_NAME = "tacofoundation:cloudsen12-l2a"
EXTRA_DATASET_NAME = "tacofoundation:cloudsen12-extra"
BAND_INDICES = [3, 2, 1, 7]
NUM_CLASSES = 4
NO_DATA_VALUE = 99


def spectral_features(image: np.ndarray) -> np.ndarray:
    """Return RGB/NIR reflectance plus three spectral features per pixel."""
    image = image[:12].astype(np.float32) / 10000.0
    pixels = image[BAND_INDICES].transpose(1, 2, 0).reshape(-1, 4)
    red, green, blue, nir = pixels.T
    ndvi = (nir - red) / (nir + red + 1e-7)
    brightness = pixels.mean(axis=1)
    greenness = green / (red + green + blue + 1e-7)
    return np.column_stack((pixels, ndvi, brightness, greenness))


def cloudsen12_labels(target: np.ndarray) -> np.ndarray:
    """Normalize CloudSEN12 labels: 99 is no-data, labels 0-3 are valid."""
    labels = target.reshape(-1).astype(np.int8)
    labels[labels == NO_DATA_VALUE] = -1
    return labels


def sen2cor_labels(scl: np.ndarray) -> np.ndarray:
    """Map Sen2Cor SCL classes to CloudSEN12's clear/thick/thin/shadow classes."""
    scl = scl.reshape(-1)
    mapped = np.zeros_like(scl, dtype=np.int8)
    mapped[np.isin(scl, [8, 9])] = 1
    mapped[scl == 10] = 2
    mapped[scl == 3] = 3
    return mapped


def read_sample(
    l2a_dataset, extra_dataset, index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read spectral image, human labels, and Sen2Cor baseline for one index."""
    sample = l2a_dataset.read(index)
    image_path = sample.read(int(sample.index[sample["tortilla:id"] == "s2l2a"][0]))
    target_path = sample.read(int(sample.index[sample["tortilla:id"] == "target"][0]))

    extra = extra_dataset.read(index)
    baseline_row = extra.index[extra["tortilla:id"] == "cloudmask_sen2cor"][0]
    baseline_path = extra.read(int(baseline_row))

    with rasterio.open(image_path) as src:
        image = src.read()
    with rasterio.open(target_path) as src:
        target = src.read(1)
    with rasterio.open(baseline_path) as src:
        baseline = src.read(1)

    return spectral_features(image), cloudsen12_labels(target), sen2cor_labels(baseline)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute macro metrics over the four CloudSEN12 classes."""
    labels = np.arange(NUM_CLASSES)
    return {
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "iou": float(jaccard_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def collect_data(
    l2a_dataset,
    extra_dataset,
    indices: Iterable[int],
    max_pixels_per_scene: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect a bounded, deterministic pixel sample from each scene."""
    features, labels, baselines = [], [], []
    rng = np.random.default_rng(42)
    for index in indices:
        scene_features, scene_labels, scene_baseline = read_sample(
            l2a_dataset, extra_dataset, int(index)
        )
        valid = scene_labels >= 0
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) > max_pixels_per_scene:
            valid_indices = rng.choice(valid_indices, max_pixels_per_scene, replace=False)
        features.append(scene_features[valid_indices])
        labels.append(scene_labels[valid_indices])
        baselines.append(scene_baseline[valid_indices])
    return np.vstack(features), np.concatenate(labels), np.concatenate(baselines)


def run_experiment(
    train_indices: Iterable[int],
    validation_indices: Iterable[int],
    output_dir: Path,
    max_pixels_per_scene: int = 2000,
) -> Dict[str, object]:
    """Train RF and compare it with Sen2Cor on held-out scenes."""
    train_indices = list(train_indices)
    validation_indices = list(validation_indices)
    l2a_dataset = tacoreader.load(DATASET_NAME)
    extra_dataset = tacoreader.load(EXTRA_DATASET_NAME)

    x_train, y_train, _ = collect_data(
        l2a_dataset, extra_dataset, train_indices, max_pixels_per_scene
    )
    x_val, y_val, y_sen2cor = collect_data(
        l2a_dataset, extra_dataset, validation_indices, max_pixels_per_scene
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    y_rf = model.predict(x_val)

    results = {
        "seed": 42,
        "dataset": DATASET_NAME,
        "operational_baseline": "cloudmask_sen2cor",
        "train_scenes": len(train_indices),
        "validation_scenes": len(validation_indices),
        "max_pixels_per_scene": max_pixels_per_scene,
        "train_pixels": int(len(y_train)),
        "validation_pixels": int(len(y_val)),
        "features": ["B4", "B3", "B2", "B8", "NDVI", "brightness", "greenness"],
        "random_forest": metrics(y_val, y_rf),
        "sen2cor": metrics(y_val, y_sen2cor),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "week2_results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "rf_feature_importance.json").write_text(
        json.dumps(dict(zip(results["features"], model.feature_importances_.tolist())), indent=2)
    )
    joblib.dump(model, output_dir / "random_forest.joblib")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenes", type=int, default=10)
    parser.add_argument("--validation-scenes", type=int, default=5)
    parser.add_argument("--max-pixels-per-scene", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / "outputs" / "split_info.json").read_text())
    results = run_experiment(
        split["train_indices"][: args.train_scenes],
        split["val_indices"][: args.validation_scenes],
        root / "outputs",
        max_pixels_per_scene=args.max_pixels_per_scene,
    )
    print(json.dumps(results, indent=2))
