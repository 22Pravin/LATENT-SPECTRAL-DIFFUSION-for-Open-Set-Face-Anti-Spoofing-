"""
2D Discrete Wavelet Transform (DWT) module.
Performs Haar wavelet decomposition on face images to extract
LL, LH, HL, HH frequency sub-bands.

The HH band captures high-frequency details that are particularly
discriminative for detecting presentation attacks (printing artifacts,
moiré patterns, screen pixel grids).
"""

import numpy as np
import pywt
import torch
import cv2


def dwt2d_numpy(image: np.ndarray, wavelet: str = 'haar') -> dict:
    """
    Apply 2D DWT to an image (H, W, C) and return sub-bands.

    Args:
        image: Input image as numpy array (H, W, C), values in [0, 255] or [0, 1].
        wavelet: Wavelet family to use (default: 'haar').

    Returns:
        Dictionary with keys 'LL', 'LH', 'HL', 'HH', each of shape (H//2, W//2, C).
    """
    bands = {'LL': [], 'LH': [], 'HL': [], 'HH': []}

    # Process each channel independently
    for c in range(image.shape[2]):
        coeffs = pywt.dwt2(image[:, :, c], wavelet)
        cA, (cH, cV, cD) = coeffs

        bands['LL'].append(cA)  # Approximation (low-low)
        bands['LH'].append(cH)  # Horizontal detail (low-high)
        bands['HL'].append(cV)  # Vertical detail (high-low)
        bands['HH'].append(cD)  # Diagonal detail (high-high)

    # Stack channels: list of (H/2, W/2) → (H/2, W/2, C)
    for key in bands:
        bands[key] = np.stack(bands[key], axis=-1)

    return bands


def dwt2d_tensor(image_tensor: torch.Tensor, wavelet: str = 'haar') -> dict:
    """
    Apply 2D DWT to a batch of images (PyTorch tensor).

    Args:
        image_tensor: (B, C, H, W) tensor, values in [0, 1].
        wavelet: Wavelet family.

    Returns:
        Dictionary with 'LL', 'LH', 'HL', 'HH', each (B, C, H//2, W//2).
    """
    # Convert to numpy, process, convert back
    batch = image_tensor.detach().cpu().numpy()
    B, C, H, W = batch.shape

    results = {k: [] for k in ['LL', 'LH', 'HL', 'HH']}

    for b in range(B):
        # (C, H, W) → (H, W, C)
        img = np.transpose(batch[b], (1, 2, 0))
        bands = dwt2d_numpy(img, wavelet)

        for key in results:
            # (H/2, W/2, C) → (C, H/2, W/2)
            results[key].append(np.transpose(bands[key], (2, 0, 1)))

    # Stack batch: list of (C, H/2, W/2) → (B, C, H/2, W/2)
    for key in results:
        results[key] = torch.tensor(
            np.stack(results[key], axis=0),
            dtype=image_tensor.dtype,
            device=image_tensor.device
        )

    return results


def concatenate_dwt_bands(bands: dict) -> torch.Tensor:
    """
    Concatenate all 4 DWT sub-bands along channel dimension.

    Args:
        bands: Dict with 'LL', 'LH', 'HL', 'HH', each (B, C, H/2, W/2).

    Returns:
        Concatenated tensor of shape (B, 4*C, H/2, W/2).
        For RGB input: (B, 12, 64, 64).
    """
    return torch.cat([bands['LL'], bands['LH'], bands['HL'], bands['HH']], dim=1)


def normalize_band(band: np.ndarray) -> np.ndarray:
    """
    Normalize a wavelet band to [0, 1] range for visualization.

    Args:
        band: DWT sub-band array.

    Returns:
        Normalized array in [0, 1].
    """
    band_min = band.min()
    band_max = band.max()
    if band_max - band_min > 0:
        return (band - band_min) / (band_max - band_min)
    return np.zeros_like(band)


def visualize_dwt_bands(image: np.ndarray, wavelet: str = 'haar'):
    """
    Visualize the 4 DWT sub-bands of an image.
    Returns a figure-ready dictionary.

    Args:
        image: Input image (H, W, C) in [0, 255] uint8 or [0, 1] float.

    Returns:
        Dict with 'LL', 'LH', 'HL', 'HH' as normalized uint8 images.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    bands = dwt2d_numpy(image, wavelet)

    vis = {}
    for key in bands:
        normalized = normalize_band(bands[key])
        vis[key] = (normalized * 255).astype(np.uint8)

    return vis
