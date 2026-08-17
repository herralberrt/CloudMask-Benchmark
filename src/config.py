"""
Configuration module for CloudSEN12 cloud detection project.

Stores all configuration parameters and experiment setup.
"""

import json
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for cloud detection experiment."""
    
    # Dataset configuration
    DATASET_NAME = "CloudSEN12-L2A"
    DATASET_SOURCE = "tacofoundation:cloudsen12-l2a"
    SATELLITE = "Sentinel-2"
    
    # Full dataset statistics
    FULL_DATASET_SIZE = 50247
    
    # Week 1 subset configuration
    SUBSET_FRACTION = 0.10  # Use 10% for Week 1
    SUBSET_SIZE = 5024
    
    # Train/Val/Test split
    TRAIN_FRACTION = 0.60
    VAL_FRACTION = 0.20
    TEST_FRACTION = 0.20
    
    TRAIN_SIZE = int(SUBSET_SIZE * TRAIN_FRACTION)
    VAL_SIZE = int(SUBSET_SIZE * VAL_FRACTION)
    TEST_SIZE = SUBSET_SIZE - TRAIN_SIZE - VAL_SIZE
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Preprocessing configuration
    SELECTED_BANDS = [3, 2, 1, 7]  # Red, Green, Blue, NIR
    SELECTED_BAND_NAMES = ['Red (B4)', 'Green (B3)', 'Blue (B2)', 'NIR (B8)']
    
    SCALE_FACTOR = 10000  # Sentinel-2 L2A normalization factor
    INPUT_RANGE = (0.0, 1.0)  # Normalized value range
    
    # Cloud classes
    CLASS_LABELS = {
        0: 'Clear sky',
        1: 'Thin cloud',
        2: 'Thick cloud',
        3: 'Cloud shadow'
    }
    NUM_CLASSES = 4
    
    # Evaluation metrics
    METRICS = ['precision', 'recall', 'f1_score', 'iou']
    
    # File paths
    DATA_DIR = Path("../data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUTS_DIR = Path("../outputs")
    FIGURES_DIR = OUTPUTS_DIR / "figures"
    NOTEBOOKS_DIR = Path("../notebooks")
    SRC_DIR = Path("../src")
    
    # Configuration file names
    DATASET_CONFIG_FILE = OUTPUTS_DIR / "dataset_config.json"
    PREPROCESSING_CONFIG_FILE = OUTPUTS_DIR / "preprocessing_config.json"
    CLASS_INFO_FILE = OUTPUTS_DIR / "class_info.json"
    SPLIT_INFO_FILE = OUTPUTS_DIR / "split_info.json"
    
    @classmethod
    def get_dict(cls) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            'dataset': {
                'name': cls.DATASET_NAME,
                'source': cls.DATASET_SOURCE,
                'satellite': cls.SATELLITE,
                'full_size': cls.FULL_DATASET_SIZE,
                'subset_fraction': cls.SUBSET_FRACTION,
                'subset_size': cls.SUBSET_SIZE,
            },
            'split': {
                'train_fraction': cls.TRAIN_FRACTION,
                'val_fraction': cls.VAL_FRACTION,
                'test_fraction': cls.TEST_FRACTION,
                'train_size': cls.TRAIN_SIZE,
                'val_size': cls.VAL_SIZE,
                'test_size': cls.TEST_SIZE,
            },
            'preprocessing': {
                'selected_bands': cls.SELECTED_BANDS,
                'selected_band_names': cls.SELECTED_BAND_NAMES,
                'scale_factor': cls.SCALE_FACTOR,
                'input_range': cls.INPUT_RANGE,
            },
            'classes': cls.CLASS_LABELS,
            'metrics': cls.METRICS,
            'random_seed': cls.RANDOM_SEED,
        }
    
    @classmethod
    def save_to_json(cls, filepath: Path) -> None:
        """Save configuration to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(cls.get_dict(), f, indent=2)
    
    @classmethod
    def print_summary(cls) -> None:
        """Print configuration summary."""
        print("\n" + "="*70)
        print("WEEK 1 - EXPERIMENTAL CONFIGURATION")
        print("="*70)
        
        print(f"\nDataset:")
        print(f"  Name: {cls.DATASET_NAME}")
        print(f"  Full size: {cls.FULL_DATASET_SIZE:,} samples")
        print(f"  Subset: {cls.SUBSET_SIZE:,} samples ({cls.SUBSET_FRACTION*100:.1f}%)")
        
        print(f"\nTrain/Val/Test Split:")
        print(f"  Training:   {cls.TRAIN_SIZE:,} samples ({cls.TRAIN_FRACTION*100:.0f}%)")
        print(f"  Validation: {cls.VAL_SIZE:,} samples ({cls.VAL_FRACTION*100:.0f}%)")
        print(f"  Test:       {cls.TEST_SIZE:,} samples ({cls.TEST_FRACTION*100:.0f}%)")
        
        print(f"\nInput Bands:")
        print(f"  Selected: {cls.SELECTED_BAND_NAMES}")
        print(f"  Indices: {cls.SELECTED_BANDS}")
        
        print(f"\nPreprocessing:")
        print(f"  Normalization: x / {cls.SCALE_FACTOR}")
        print(f"  Output range: {cls.INPUT_RANGE}")
        
        print(f"\nCloud Classes ({cls.NUM_CLASSES}):")
        for class_id, label in cls.CLASS_LABELS.items():
            print(f"  {class_id}: {label}")
        
        print(f"\nEvaluation Metrics:")
        print(f"  {', '.join(cls.METRICS)}")
        
        print(f"\nRandom Seed: {cls.RANDOM_SEED}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Print configuration when running as script
    Config.print_summary()
