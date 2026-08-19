"""Train a compact U-Net for CloudSEN12 cloud segmentation."""

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .cloudsen12_data import (
        IGNORE_INDEX,
        PATCH_SIZE,
        compute_metrics,
        extract_patches,
        load_datasets,
        read_scene,
        spectral_features,
    )
    from .unet_model import UNet
except ImportError:
    from cloudsen12_data import (
        IGNORE_INDEX,
        PATCH_SIZE,
        compute_metrics,
        extract_patches,
        load_datasets,
        read_scene,
        spectral_features,
    )
    from unet_model import UNet

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatchDataset(Dataset):
    """In-memory patch dataset for a small reproducible U-Net experiment."""

    def __init__(self, images, labels) -> None:
        self.images = torch.from_numpy(np.stack(images)).float()
        self.labels = torch.from_numpy(np.stack(labels)).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def build_dataset(l2a_dataset, extra_dataset, indices: Iterable[int], max_patches: int) -> PatchDataset:
    images, labels = [], []
    for index in indices:
        image, scene_labels, _ = read_scene(l2a_dataset, extra_dataset, index)
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


def evaluate_random_forest_and_baseline(l2a_dataset, extra_dataset, indices: Iterable[int], random_forest_path: Path):
    """Evaluate the saved Random Forest and Sen2Cor baseline on the same scenes."""
    rf = joblib.load(random_forest_path)
    labels, rf_predictions, baseline_predictions = [], [], []
    for index in indices:
        image, scene_labels, baseline = read_scene(l2a_dataset, extra_dataset, index)
        features = spectral_features(image)
        valid = scene_labels.reshape(-1) != IGNORE_INDEX
        labels.append(scene_labels.reshape(-1)[valid])
        rf_predictions.append(rf.predict(features[valid]))
        baseline_predictions.append(baseline.reshape(-1)[valid])
    y_true = np.concatenate(labels)
    return compute_metrics(y_true, np.concatenate(rf_predictions)), compute_metrics(y_true, np.concatenate(baseline_predictions))


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
        image, scene_labels, baseline = read_scene(l2a_dataset, extra_dataset, index)
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
    """Train U-Net and compare it with the Random Forest and Sen2Cor baseline."""
    set_seed()
    train_indices, validation_indices = list(train_indices), list(validation_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l2a_dataset, extra_dataset = load_datasets()
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    _, y_val, y_unet = evaluate_unet(model, validation_loader, device)
    rf_results, sen2cor_results = evaluate_random_forest_and_baseline(
        l2a_dataset, extra_dataset, validation_indices, output_dir / "random_forest_model.joblib"
    )
    prediction_path = output_dir / "unet_validation_predictions.npz"
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
        "unet": compute_metrics(y_val, y_unet),
        "random_forest": rf_results,
        "sen2cor": sen2cor_results,
        "prediction_artifact": prediction_path.name,
    }
    (output_dir / "unet_training_results.json").write_text(json.dumps(results, indent=2))
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
