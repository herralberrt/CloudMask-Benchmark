"""
CloudSEN12 Preprocessing Module

Handles data loading, normalization, and preprocessing for CloudSEN12 dataset.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Optional


class CloudSEN12Preprocessor:
    """
    Preprocessor for CloudSEN12 Sentinel-2 L2A data.
    
    Attributes:
        selected_bands (list): Indices of bands to extract [3, 2, 1, 7] (R, G, B, NIR)
        scale_factor (int): Divisor for normalization (10000 for Sentinel-2 L2A)
        output_range (tuple): Target range for normalized values (0, 1)
    """
    
    # Sentinel-2 L2A band information
    SENTINEL2_BANDS = {
        0: {'name': 'B1', 'description': 'Coastal aerosol', 'resolution': '60m'},
        1: {'name': 'B2', 'description': 'Blue', 'resolution': '10m'},
        2: {'name': 'B3', 'description': 'Green', 'resolution': '10m'},
        3: {'name': 'B4', 'description': 'Red', 'resolution': '10m'},
        4: {'name': 'B5', 'description': 'Vegetation Red Edge', 'resolution': '20m'},
        5: {'name': 'B6', 'description': 'Vegetation Red Edge', 'resolution': '20m'},
        6: {'name': 'B7', 'description': 'Vegetation Red Edge', 'resolution': '20m'},
        7: {'name': 'B8', 'description': 'NIR', 'resolution': '10m'},
        8: {'name': 'B8A', 'description': 'Vegetation Red Edge', 'resolution': '20m'},
        9: {'name': 'B11', 'description': 'SWIR', 'resolution': '20m'},
        10: {'name': 'B12', 'description': 'SWIR', 'resolution': '20m'},
        11: {'name': 'SCL', 'description': 'Scene Classification', 'resolution': '20m'}
    }
    
    # Cloud mask class labels
    CLASS_LABELS = {
        0: 'Clear sky',
        1: 'Thick cloud',
        2: 'Thin cloud',
        3: 'Cloud shadow'
    }
    
    def __init__(self, 
                 selected_bands: list = None,
                 scale_factor: int = 10000,
                 clip_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize preprocessor.
        
        Args:
            selected_bands (list): Band indices to extract. Default: [3, 2, 1, 7] (R, G, B, NIR)
            scale_factor (int): Normalization divisor. Default: 10000
            clip_range (tuple): Min/max values for clipping. Default: (0.0, 1.0)
        """
        if selected_bands is None:
            selected_bands = [3, 2, 1, 7]  # Red, Green, Blue, NIR
        
        self.selected_bands = selected_bands
        self.scale_factor = scale_factor
        self.clip_range = clip_range
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from GeoTIFF file.
        
        Args:
            image_path (str): Path to image GeoTIFF
            
        Returns:
            np.ndarray: Image array of shape (num_bands, H, W)
        """
        with rasterio.open(image_path) as src:
            image = src.read()
        return image
    
    def load_mask(self, mask_path: str) -> np.ndarray:
        """
        Load cloud mask from GeoTIFF file.
        
        Args:
            mask_path (str): Path to mask GeoTIFF
            
        Returns:
            np.ndarray: Mask array of shape (1, H, W) or (H, W)
        """
        with rasterio.open(mask_path) as src:
            mask = src.read()
        return mask
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values to [0, 1] range.
        
        Args:
            image (np.ndarray): Image array
            
        Returns:
            np.ndarray: Normalized image
        """
        # Scale by factor
        normalized = image.astype(np.float32) / self.scale_factor
        
        # Clip to range
        normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        
        return normalized
    
    def extract_bands(self, image: np.ndarray, band_indices: Optional[list] = None) -> np.ndarray:
        """
        Extract selected bands from image.
        
        Args:
            image (np.ndarray): Image array of shape (num_bands, H, W) or (H, W, num_bands)
            band_indices (list, optional): Band indices to extract. Defaults to selected_bands.
            
        Returns:
            np.ndarray: Extracted bands in the same channel layout as the input.
        """
        band_indices = self.selected_bands if band_indices is None else band_indices
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D image array, got shape {image.shape}")

        channel_count = len(self.SENTINEL2_BANDS)
        if image.shape[0] == channel_count and image.shape[-1] != channel_count:
            return image[band_indices, :, :]
        if image.shape[-1] == channel_count and image.shape[0] != channel_count:
            return image[:, :, band_indices]

        if image.shape[0] <= 16 and image.shape[-1] > 16:
            return image[band_indices, :, :]
        if image.shape[-1] <= 16 and image.shape[0] > 16:
            return image[:, :, band_indices]
        raise ValueError(
            "Cannot infer channel layout for image shape "
            f"{image.shape}; use channel-first or channel-last data with at most 16 bands."
        )
    
    def preprocess(self, image: np.ndarray, mask: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            image (np.ndarray): Raw image array
            mask (np.ndarray, optional): Cloud mask array
            
        Returns:
            tuple: (preprocessed_image, mask)
                - preprocessed_image: shape (4, H, W), values in [0, 1]
                - mask: shape (1, H, W) or (H, W), multiclass labels
        """
        # Step 1: Extract selected bands
        image_selected = self.extract_bands(image)
        
        # Step 2: Normalize to [0, 1]
        image_normalized = self.normalize_image(image_selected)
        
        # Step 3: Process mask if provided
        if mask is not None:
            # Ensure mask is 2D (remove channel dimension if present)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
        
        return image_normalized, mask
    
    def get_band_info(self, band_indices: list = None) -> dict:
        """
        Get information about selected bands.
        
        Args:
            band_indices (list): Band indices. Default: self.selected_bands
            
        Returns:
            dict: Band information
        """
        if band_indices is None:
            band_indices = self.selected_bands
        
        band_info = {}
        for idx in band_indices:
            band_info[idx] = self.SENTINEL2_BANDS[idx]
        
        return band_info


class CloudDetectionDataset:
    """
    Base class for cloud detection datasets.
    """
    
    CLASS_LABELS = CloudSEN12Preprocessor.CLASS_LABELS
    
    def __init__(self, samples: list, preprocessor: CloudSEN12Preprocessor = None):
        """
        Initialize dataset.
        
        Args:
            samples (list): List of (image_path, mask_path) tuples
            preprocessor (CloudSEN12Preprocessor): Preprocessor instance
        """
        self.samples = samples
        self.preprocessor = preprocessor or CloudSEN12Preprocessor()
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get sample by index.
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: {'image': np.ndarray, 'mask': np.ndarray}
        """
        image_path, mask_path = self.samples[idx]
        
        # Load data
        image = self.preprocessor.load_image(image_path)
        mask = self.preprocessor.load_mask(mask_path)
        
        # Preprocess
        image, mask = self.preprocessor.preprocess(image, mask)
        
        return {
            'image': image,
            'mask': mask,
            'image_path': image_path,
            'mask_path': mask_path
        }
