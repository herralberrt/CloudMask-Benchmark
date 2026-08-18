"""Week 3 CloudSEN12 U-Net training and first model comparison."""

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import rasterio
import tacoreader.v1 as tacoreader
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .unet_model import UNet
    from .week2_pipeline import (
        BAND_INDICES,
        DATASET_NAME,
        EXTRA_DATASET_NAME,
        NUM_CLASSES,
        cloudsen12_labels,
        metrics,
        sen2cor_labels,
        spectral_features,
    )
except ImportError:
    from unet_model import UNet
    from week2_pipeline import (
        BAND_INDICES,
        DATASET_NAME,
        EXTRA_DATASET_NAME,
        NUM_CLASSES,
        cloudsen12_labels,
        metrics,
        sen2cor_labels,
        spectral_features,
    )

PATCH_SIZE = 128
IGNORE_INDEX = 255
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_scene(l2a_dataset, extra_dataset, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read normalized four-band image, labels, and Sen2Cor baseline."""
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


def extract_patches(
    image: np.ndarray,
    labels: np.ndarray,
    patch_size: int = PATCH_SIZE,
    max_patches: int | None = None,
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


class PatchDataset(Dataset):
    """In-memory patch dataset for a small reproducible Week 3 experiment."""

    def __init__(self, images: List[np.ndarray], labels: List[np.ndarray]) -> None:
        self.images = torch.from_numpy(np.stack(images)).float()
        self.labels = torch.from_numpy(np.stack(labels)).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def build_dataset(l2a_dataset, extra_dataset, indices: Iterable[int], max_patches: int) -> PatchDataset:
    images, labels = [], []
    for index in indices:
        image, scene_labels, _ = read_scene(l2a_dataset, extra_dataset, int(index))
        scene_images, scene_masks = extract_patches(image, scene_labels, max_patches=max_patches)
        images.extend(scene_images)
        labels.extend(scene_masks)
    if not images:
        raise ValueError("No valid image patches were found")
    return PatchDataset(images, labels)


def evaluate_unet(model, loader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return average loss and flattened valid predictions/labels."""
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    model.eval()
    losses, predictions, labels = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            losses.append(float(criterion(logits, targets).item()))
            predicted = logits.argmax(dim=1)
            valid = targets != IGNORE_INDEX
            predictions.append(predicted[valid].cpu().numpy())
            labels.append(targets[valid].cpu().numpy())
    return float(np.mean(losses)), np.concatenate(labels), np.concatenate(predictions)


def evaluate_rf_and_sen2cor(l2a_dataset, extra_dataset, indices: Iterable[int]):
    """Evaluate saved RF and Sen2Cor on the same validation scenes."""
    import joblib

    root = Path(__file__).resolve().parents[1]
    rf = joblib.load(root / "outputs" / "random_forest.joblib")
    labels, rf_predictions, baseline_predictions = [], [], []
    for index in indices:
        image, scene_labels, baseline = read_scene(l2a_dataset, extra_dataset, int(index))
        pixels = image.transpose(1, 2, 0).reshape(-1, 4)
        red, green, blue, nir = pixels.T
        ndvi = (nir - red) / (nir + red + 1e-7)
        brightness = pixels.mean(axis=1)
        greenness = green / (red + green + blue + 1e-7)
        features = np.column_stack((pixels, ndvi, brightness, greenness))
        valid = scene_labels.reshape(-1) != IGNORE_INDEX
        labels.append(scene_labels.reshape(-1)[valid])
        rf_predictions.append(rf.predict(features[valid]))
        baseline_predictions.append(baseline.reshape(-1)[valid])
    y_true = np.concatenate(labels)
    return metrics(y_true, np.concatenate(rf_predictions)), metrics(y_true, np.concatenate(baseline_predictions))


def save_prediction_examples(
    model,
    l2a_dataset,
    extra_dataset,
    indices: Iterable[int],
    output_path: Path,
    device: torch.device,
) -> None:
    """Save aligned validation patches and predictions for qualitative analysis."""
    scene_ids, images, labels, unet_predictions, baselines = [], [], [], [], []
    model.eval()
    for index in indices:
        image, scene_labels, baseline = read_scene(l2a_dataset, extra_dataset, int(index))
        image_patches, label_patches = extract_patches(image, scene_labels)
        baseline_patches, _ = extract_patches(baseline[None, ...], scene_labels)
        for image_patch, label_patch, baseline_patch_image in zip(
            image_patches, label_patches, baseline_patches
        ):
            with torch.no_grad():
                logits = model(torch.from_numpy(image_patch).unsqueeze(0).to(device))
            scene_ids.append(int(index))
            images.append(image_patch)
            labels.append(label_patch)
            unet_predictions.append(logits.argmax(dim=1).squeeze(0).cpu().numpy())
            baselines.append(baseline_patch_image[0])
    np.savez_compressed(
        output_path,
        scene_indices=np.asarray(scene_ids),
        images=np.asarray(images, dtype=np.float32),
        ground_truth=np.asarray(labels, dtype=np.int16),
        unet=np.asarray(unet_predictions, dtype=np.int8),
        sen2cor=np.asarray(baselines, dtype=np.int8),
    )


def run_experiment(
    train_indices: Iterable[int],
    validation_indices: Iterable[int],
    output_dir: Path,
    epochs: int = 3,
    max_patches_per_scene: int = 4,
    batch_size: int = 4,
) -> dict:
    """Train U-Net and compare it with RF and Sen2Cor."""
    set_seed()
    train_indices, validation_indices = list(train_indices), list(validation_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l2a_dataset = tacoreader.load(DATASET_NAME)
    extra_dataset = tacoreader.load(EXTRA_DATASET_NAME)
    train_data = build_dataset(l2a_dataset, extra_dataset, train_indices, max_patches_per_scene)
    validation_data = build_dataset(l2a_dataset, extra_dataset, validation_indices, max_patches_per_scene)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    history, best_val_loss = [], float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "unet_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, y_val, y_unet = evaluate_unet(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
        history.append(row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, checkpoint_path)
        print(row)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    _, y_val, y_unet = evaluate_unet(model, validation_loader, device)
    rf_results, sen2cor_results = evaluate_rf_and_sen2cor(l2a_dataset, extra_dataset, validation_indices)
    prediction_path = output_dir / "week3_predictions.npz"
    save_prediction_examples(model, l2a_dataset, extra_dataset, validation_indices, prediction_path, device)
    results = {
        "seed": SEED,
        "device": str(device),
        "train_scenes": len(train_indices),
        "validation_scenes": len(validation_indices),
        "patch_size": PATCH_SIZE,
        "train_patches": len(train_data),
        "validation_patches": len(validation_data),
        "epochs": epochs,
        "history": history,
        "unet": metrics(y_val, y_unet),
        "random_forest": rf_results,
        "sen2cor": sen2cor_results,
        "prediction_artifact": str(prediction_path.name),
    }
    (output_dir / "week3_results.json").write_text(json.dumps(results, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenes", type=int, default=3)
    parser.add_argument("--validation-scenes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-patches-per-scene", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / "outputs" / "split_info.json").read_text())
    print(json.dumps(run_experiment(
        split["train_indices"][: args.train_scenes],
        split["val_indices"][: args.validation_scenes],
        root / "outputs",
        epochs=args.epochs,
        max_patches_per_scene=args.max_patches_per_scene,
        batch_size=args.batch_size,
    ), indent=2))
