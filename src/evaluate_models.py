"""Freeze the trained models and evaluate them on the held-out test scenes.

This is the final comparison stage: the Sen2Cor operational baseline, the
Random Forest, and the U-Net are all evaluated on the same test scenes.
Besides overall and per-class Precision/Recall/F1/IoU, it analyses difficult
cases (thin cloud, snow, bright surfaces, cloud boundaries) and saves
qualitative side-by-side comparison figures.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .cloudsen12_data import (
        CLASS_NAMES,
        IGNORE_INDEX,
        NUM_CLASSES,
        PATCH_SIZE,
        SNOW_SCL_VALUE,
        compute_metrics,
        compute_per_class_metrics,
        load_datasets,
        read_raw_sen2cor,
        read_scene,
        spectral_features,
    )
    from .unet_model import UNet
    from .train_random_forest import run_experiment as train_rf_experiment
except ImportError:
    from cloudsen12_data import (
        CLASS_NAMES,
        IGNORE_INDEX,
        NUM_CLASSES,
        PATCH_SIZE,
        SNOW_SCL_VALUE,
        compute_metrics,
        compute_per_class_metrics,
        load_datasets,
        read_raw_sen2cor,
        read_scene,
        spectral_features,
    )
    from unet_model import UNet
    from train_random_forest import run_experiment as train_rf_experiment

BRIGHTNESS_THRESHOLD = 0.3  # Reflectance above which a "clear" pixel counts as a bright surface
BRIGHTNESS_FEATURE_COLUMN = 5  # Column index of "brightness" in spectral_features' output


def generate_split_indices(split_info: dict) -> tuple:
    """Regenerate train/val/test indices from split parameters using seed."""
    subset_size = split_info["subset_size"]
    train_size = split_info["train_size"]
    val_size = split_info["val_size"]
    seed = split_info["random_seed"]
    
    all_indices = np.arange(subset_size)
    np.random.RandomState(seed).shuffle(all_indices)
    
    train_indices = sorted(all_indices[:train_size].tolist())
    val_indices = sorted(all_indices[train_size:train_size + val_size].tolist())
    test_indices = sorted(all_indices[train_size + val_size:].tolist())
    
    return train_indices, val_indices, test_indices


def run_unet_full_scene(model, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Tile a full scene into non-overlapping patches and stitch U-Net predictions."""
    _, height, width = image.shape
    prediction = np.zeros((height, width), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for top in range(0, height, PATCH_SIZE):
            for left in range(0, width, PATCH_SIZE):
                patch = image[:, top : top + PATCH_SIZE, left : left + PATCH_SIZE]
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(tensor)
                prediction[top : top + PATCH_SIZE, left : left + PATCH_SIZE] = (
                    logits.argmax(dim=1).squeeze(0).cpu().numpy()
                )
    return prediction


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Flag ground-truth pixels that sit next to a class transition."""
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return boundary & (labels != IGNORE_INDEX)


def flatten(scenes: List[dict], field: str, mask_field: str = "valid") -> np.ndarray:
    """Concatenate one field across scenes, restricted to each scene's mask."""
    return np.concatenate([scene[field][scene[mask_field]] for scene in scenes])


def evaluate_case(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Compare each model's predictions with ground truth on a difficult-case subset."""
    result: Dict[str, object] = {"pixel_count": int(len(y_true))}
    if len(y_true) == 0:
        return result
    for name, y_pred in predictions.items():
        result[name] = {
            "agreement_rate": float(np.mean(y_pred == y_true)),
            "false_cloud_rate": float(np.mean(np.isin(y_pred, [1, 2]))),
        }
    return result


def summarize_strengths_and_weaknesses(per_class: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, dict]:
    """Report each model's best and worst class by IoU."""
    summary = {}
    for model_name, class_metrics in per_class.items():
        ranked = sorted(class_metrics.items(), key=lambda item: item[1]["iou"])
        summary[model_name] = {
            "weakest_class": ranked[0][0],
            "weakest_class_iou": ranked[0][1]["iou"],
            "strongest_class": ranked[-1][0],
            "strongest_class_iou": ranked[-1][1]["iou"],
        }
    return summary


def save_qualitative_figure(image: np.ndarray, scene: dict, path: Path) -> None:
    """Save a side-by-side comparison: RGB image, ground truth, baseline, RF, U-Net."""
    rgb = np.clip(image[:3].transpose(1, 2, 0) * 3.0, 0, 1)
    display_labels = np.where(scene["labels"] == IGNORE_INDEX, 0, scene["labels"])
    panels = [
        (rgb, "Sentinel-2 RGB", False),
        (display_labels, "Ground truth", True),
        (scene["baseline"], "Sen2Cor baseline", True),
        (scene["rf"], "Random Forest", True),
        (scene["unet"], "U-Net", True),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (data, title, is_mask) in zip(axes, panels):
        if is_mask:
            ax.imshow(data, cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1)
        else:
            ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(test_indices: Iterable[int], output_dir: Path, num_figures: int = 3) -> dict:
    """Evaluate the frozen Random Forest and U-Net against Sen2Cor on test scenes."""
    test_indices = list(test_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l2a_dataset, extra_dataset = load_datasets()

    # Auto-generate Random Forest model if it doesn't exist
    rf_model_path = output_dir / "random_forest_model.joblib"
    if not rf_model_path.exists():
        print("Random Forest model not found. Training it now...")
        split_info_path = output_dir / "split_info.json"
        if split_info_path.exists():
            split = json.loads(split_info_path.read_text())
            train_indices, val_indices, _ = generate_split_indices(split)
            train_rf_experiment(
                train_indices[:50],
                val_indices[:10],
                output_dir,
                max_pixels_per_scene=5000,
            )
            print("Random Forest model trained and saved.")
        else:
            raise FileNotFoundError(f"Cannot train RF model: split_info.json not found at {split_info_path}")

    rf = joblib.load(rf_model_path)
    unet = UNet().to(device)
    checkpoint = torch.load(output_dir / "unet_best.pt", map_location=device, weights_only=True)
    unet.load_state_dict(checkpoint["model_state"])
    unet.eval()

    figures_dir = output_dir / "figures" / "model_comparison"
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenes: List[dict] = []
    qualitative_examples = []
    for position, index in enumerate(test_indices):
        image, labels, baseline = read_scene(l2a_dataset, extra_dataset, index)
        raw_scl = read_raw_sen2cor(extra_dataset, index)
        features = spectral_features(image)
        brightness = features[:, BRIGHTNESS_FEATURE_COLUMN].reshape(labels.shape)
        rf_prediction = rf.predict(features).reshape(labels.shape)
        unet_prediction = run_unet_full_scene(unet, image, device)

        valid = labels != IGNORE_INDEX
        scene = {
            "index": index,
            "labels": labels,
            "baseline": baseline,
            "rf": rf_prediction,
            "unet": unet_prediction,
            "valid": valid,
            "snow": (raw_scl == SNOW_SCL_VALUE) & valid,
            "bright": (brightness > BRIGHTNESS_THRESHOLD) & (labels == 0) & valid,
            "boundary": boundary_mask(labels),
        }
        scenes.append(scene)

        if position < num_figures:
            figure_path = figures_dir / f"scene_{index}.png"
            save_qualitative_figure(image, scene, figure_path)
            qualitative_examples.append({
                "scene_index": int(index),
                "figure": str(figure_path.relative_to(output_dir.parent)),
            })

    y_true = flatten(scenes, "labels")
    predictions = {
        "sen2cor_baseline": flatten(scenes, "baseline"),
        "random_forest": flatten(scenes, "rf"),
        "unet": flatten(scenes, "unet"),
    }

    overall = {name: compute_metrics(y_true, y_pred) for name, y_pred in predictions.items()}
    per_class = {name: compute_per_class_metrics(y_true, y_pred) for name, y_pred in predictions.items()}

    difficult_cases = {}
    for case_name, mask_field in (("thin_cloud", None), ("snow", "snow"), ("bright_surfaces", "bright"), ("cloud_boundaries", "boundary")):
        if case_name == "thin_cloud":
            case_mask = y_true == CLASS_NAMES.index("Thin cloud")
            case_true = y_true[case_mask]
            case_predictions = {name: y_pred[case_mask] for name, y_pred in predictions.items()}
        else:
            case_true = flatten(scenes, "labels", mask_field)
            case_predictions = {name: flatten(scenes, field, mask_field) for name, field in (("sen2cor_baseline", "baseline"), ("random_forest", "rf"), ("unet", "unet"))}
        difficult_cases[case_name] = evaluate_case(case_true, case_predictions)

    results = {
        "test_scenes": len(test_indices),
        "test_pixels": int(len(y_true)),
        "patch_size": PATCH_SIZE,
        "overall": overall,
        "per_class": per_class,
        "difficult_cases": difficult_cases,
        "strengths_and_weaknesses": summarize_strengths_and_weaknesses(per_class),
        "qualitative_examples": qualitative_examples,
    }
    (output_dir / "model_comparison_results.json").write_text(json.dumps(results, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-scenes", type=int, default=8)
    parser.add_argument("--num-figures", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / "outputs" / "split_info.json").read_text())
    _, _, test_indices = generate_split_indices(split)
    print(json.dumps(run_evaluation(
        test_indices[: args.test_scenes],
        root / "outputs",
        num_figures=args.num_figures,
    ), indent=2))
