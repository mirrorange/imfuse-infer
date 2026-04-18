"""Preprocessing pipeline for BraTS-style multi-modal brain MRI volumes.

Ported from MV-IM-Fuse/preprocess.py — crop to brain bounding box and
per-modality z-score normalization on brain mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CropInfo:
    """Stores bounding-box crop coordinates for later reconstruction."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int
    original_shape: tuple[int, int, int]  # (H, W, D) before crop


def _ensure_min_128(lo: int, hi: int, dim_size: int) -> tuple[int, int]:
    """Ensure [lo, hi) spans at least 128 voxels, matching original sup_128."""
    if hi - lo < 128:
        gap = int((128 - (hi - lo)) / 2)
        hi = hi + gap + 1
        lo = lo - gap
    if lo < 0:
        hi -= lo
        lo = 0
    if hi > dim_size:
        hi = dim_size
    return lo, hi


def compute_crop(vol: np.ndarray) -> CropInfo:
    """Compute brain bounding box from a (4, H, W, D) or (H, W, D) volume.

    Returns a CropInfo with slicing coordinates [min, max) for each axis.
    Each axis span is guaranteed >= 128.
    """
    if vol.ndim == 4:
        mask_vol = np.amax(vol, axis=0)
    else:
        mask_vol = vol
    assert mask_vol.ndim == 3

    H, W, D = mask_vol.shape
    xs, ys, zs = np.where(mask_vol != 0)

    if len(xs) == 0:
        # No brain content — return center crop
        return CropInfo(
            x_min=max(0, H // 2 - 64),
            x_max=min(H, H // 2 + 64),
            y_min=max(0, W // 2 - 64),
            y_max=min(W, W // 2 + 64),
            z_min=max(0, D // 2 - 64),
            z_max=min(D, D // 2 + 64),
            original_shape=(H, W, D),
        )

    x_min, x_max = int(np.amin(xs)), int(np.amax(xs))
    y_min, y_max = int(np.amin(ys)), int(np.amax(ys))
    z_min, z_max = int(np.amin(zs)), int(np.amax(zs))

    x_min, x_max = _ensure_min_128(x_min, x_max, H)
    y_min, y_max = _ensure_min_128(y_min, y_max, W)
    z_min, z_max = _ensure_min_128(z_min, z_max, D)

    return CropInfo(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        original_shape=(H, W, D),
    )


def crop_volume(vol: np.ndarray, info: CropInfo) -> np.ndarray:
    """Crop a (C, H, W, D) or (H, W, D) volume using CropInfo."""
    if vol.ndim == 4:
        return vol[:, info.x_min : info.x_max, info.y_min : info.y_max, info.z_min : info.z_max]
    return vol[info.x_min : info.x_max, info.y_min : info.y_max, info.z_min : info.z_max]


def normalize(vol: np.ndarray) -> np.ndarray:
    """Per-modality z-score normalization on brain mask.

    Parameters
    ----------
    vol : ndarray (4, H, W, D)
        Cropped multi-modal volume (float32).

    Returns
    -------
    vol : ndarray (4, H, W, D)
        Normalized volume. Voxels outside the brain mask stay 0.
    """
    vol = vol.copy()
    mask = vol.sum(axis=0) > 0  # brain mask: any modality > 0
    for k in range(vol.shape[0]):
        x = vol[k]
        y = x[mask]
        if y.size == 0:
            continue
        mean, std = y.mean(), y.std()
        if std < 1e-8:
            vol[k] = 0.0
        else:
            vol[k] = (x - mean) / std
    return vol


def preprocess(
    modality_volumes: list[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, CropInfo]:
    """Full preprocessing pipeline for inference.

    Parameters
    ----------
    modality_volumes : list of 4 arrays or None
        [flair, t1ce, t1, t2], each (H, W, D) or None if missing.
        All present volumes must have the same spatial shape.

    Returns
    -------
    vol : ndarray (4, H', W', D') float32
        Preprocessed, cropped, normalized volume. Missing modalities are zeros.
    mask : ndarray (4,) bool
        True for each available modality.
    crop_info : CropInfo
        For mapping output back to original coordinates.
    """
    # Determine spatial shape from first available modality
    ref_shape = None
    mask = np.zeros(4, dtype=bool)
    for i, v in enumerate(modality_volumes):
        if v is not None:
            mask[i] = True
            if ref_shape is None:
                ref_shape = v.shape
            else:
                if v.shape != ref_shape:
                    raise ValueError(
                        f"Modality {i} shape {v.shape} != reference shape {ref_shape}"
                    )

    if ref_shape is None:
        raise ValueError("At least one modality must be provided")

    # Stack into (4, H, W, D)
    vol = np.zeros((4, *ref_shape), dtype=np.float32)
    for i, v in enumerate(modality_volumes):
        if v is not None:
            vol[i] = v.astype(np.float32)

    # Crop to brain bounding box
    crop_info = compute_crop(vol)
    vol = crop_volume(vol, crop_info)

    # Normalize
    vol = normalize(vol)

    return vol, mask, crop_info
