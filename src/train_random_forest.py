"""Train a Random Forest cloud classifier and compare it with the Sen2Cor baseline."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from .cloudsen12_data import (
        FEATURE_NAMES,
        IGNORE_INDEX,
        compute_metrics,
        load_datasets,
        read_scene,
        spectral_features,
    )
except ImportError:
    from cloudsen12_data import (
        FEATURE_NAMES,
        IGNORE_INDEX,
        compute_metrics,
        load_datasets,
        read_scene,
        spectral_features,
    )

DEFAULT_MAX_PIXELS_PER_SCENE = 2000


def collect_pixels(
    l2a_dataset,
    extra_dataset,
    indices: Iterable[int],
    max_pixels_per_scene: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect a bounded, deterministic pixel sample from each scene."""
    features, labels, baselines = [], [], []
    rng = np.random.default_rng(42)
    for index in indices:
        image, scene_labels, scene_baseline = read_scene(l2a_dataset, extra_dataset, index)
        scene_features = spectral_features(image)
        flat_labels = scene_labels.reshape(-1)
        flat_baseline = scene_baseline.reshape(-1)
        valid_indices = np.flatnonzero(flat_labels != IGNORE_INDEX)
        if len(valid_indices) > max_pixels_per_scene:
            valid_indices = rng.choice(valid_indices, max_pixels_per_scene, replace=False)
        features.append(scene_features[valid_indices])
        labels.append(flat_labels[valid_indices])
        baselines.append(flat_baseline[valid_indices])
    return np.vstack(features), np.concatenate(labels), np.concatenate(baselines)


def run_experiment(
    train_indices: Iterable[int],
    validation_indices: Iterable[int],
    output_dir: Path,
    max_pixels_per_scene: int = DEFAULT_MAX_PIXELS_PER_SCENE,
) -> dict:
    """Train a Random Forest and compare it with Sen2Cor on held-out scenes."""
    train_indices, validation_indices = list(train_indices), list(validation_indices)
    l2a_dataset, extra_dataset = load_datasets()

    x_train, y_train, _ = collect_pixels(l2a_dataset, extra_dataset, train_indices, max_pixels_per_scene)
    x_val, y_val, y_sen2cor = collect_pixels(l2a_dataset, extra_dataset, validation_indices, max_pixels_per_scene)

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
        "dataset": "tacofoundation:cloudsen12-l2a",
        "operational_baseline": "cloudmask_sen2cor",
        "train_scenes": len(train_indices),
        "validation_scenes": len(validation_indices),
        "max_pixels_per_scene": max_pixels_per_scene,
        "train_pixels": int(len(y_train)),
        "validation_pixels": int(len(y_val)),
        "features": FEATURE_NAMES,
        "random_forest": compute_metrics(y_val, y_rf),
        "sen2cor": compute_metrics(y_val, y_sen2cor),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "random_forest_results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "random_forest_feature_importance.json").write_text(
        json.dumps(dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())), indent=2)
    )
    joblib.dump(model, output_dir / "random_forest_model.joblib")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenes", type=int, default=10)
    parser.add_argument("--validation-scenes", type=int, default=5)
    parser.add_argument("--max-pixels-per-scene", type=int, default=DEFAULT_MAX_PIXELS_PER_SCENE)
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
