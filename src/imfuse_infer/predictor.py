"""IMFusePredictor — end-to-end NIfTI → segmentation NIfTI inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from imfuse_infer.io.checkpoint import load_checkpoint
from imfuse_infer.io.nifti import load_nifti, save_nifti
from imfuse_infer.model.imfuse import IMFuse
from imfuse_infer.preprocessing.normalize import CropInfo, preprocess

logger = logging.getLogger(__name__)

PATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Sliding window helpers
# ---------------------------------------------------------------------------

def _sliding_window_starts(size: int, window: int, overlap: float = 0.5) -> list[int]:
    """Return start indices for sliding windows along one axis."""
    if size <= window:
        return [0]
    stride = max(1, int(window * (1 - overlap)))
    last = size - window
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _sliding_window_inference(
    model: IMFuse,
    x: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    overlap: float = 0.5,
) -> torch.Tensor:
    """Run sliding-window inference with weighted averaging.

    Parameters
    ----------
    model : IMFuse (already on device, eval mode)
    x : (B, 4, H, W, D)  — padded so each dim >= patch_size
    mask : (B, 4)  bool
    patch_size : int
    overlap : float

    Returns
    -------
    pred : (B, C, H, W, D)  softmax probabilities
    """
    device = x.device
    B, _, H, W, D = x.shape
    num_cls = model.decoder_fuse.seg_layer.out_channels

    h_starts = _sliding_window_starts(H, patch_size, overlap)
    w_starts = _sliding_window_starts(W, patch_size, overlap)
    d_starts = _sliding_window_starts(D, patch_size, overlap)

    # Count overlapping patches per voxel
    one = torch.ones(1, 1, patch_size, patch_size, patch_size, device=device)
    weight = torch.zeros(1, 1, H, W, D, device=device)
    for h in h_starts:
        for w in w_starts:
            for d in d_starts:
                weight[:, :, h : h + patch_size, w : w + patch_size, d : d + patch_size] += one
    weight = weight.expand(B, num_cls, -1, -1, -1)

    # Accumulate predictions
    pred = torch.zeros(B, num_cls, H, W, D, device=device)
    for h in h_starts:
        for w in w_starts:
            for d in d_starts:
                patch = x[:, :, h : h + patch_size, w : w + patch_size, d : d + patch_size]
                out = model(patch, mask)
                pred[:, :, h : h + patch_size, w : w + patch_size, d : d + patch_size] += out

    pred = pred / weight
    return pred


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class IMFusePredictor:
    """High-level inference API for IM-Fuse brain tumor segmentation.

    Parameters
    ----------
    checkpoint : str | Path
        Path to .pth or .safetensors checkpoint.
    device : str
        ``"cuda"``, ``"cpu"`` or specific device.
    num_cls : int
        Number of segmentation classes.
    interleaved_tokenization : bool
        Interleaved (True) vs concatenated (False) tokenization.
    mamba_skip : bool
        Use Mamba fusion in skip connections.
    mamba_backend : str
        ``"auto"``, ``"mamba_ssm"`` or ``"mambapy"``.
    overlap : float
        Overlap ratio for sliding window (0.0–1.0).

    Example
    -------
    >>> predictor = IMFusePredictor("model_last.pth")
    >>> predictor.predict_nifti(
    ...     {"flair": "flair.nii.gz", "t1ce": "t1c.nii.gz", "t1": "t1n.nii.gz", "t2": "t2w.nii.gz"},
    ...     "output_seg.nii.gz",
    ... )
    """

    MODALITY_ORDER = ("flair", "t1ce", "t1", "t2")

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        num_cls: int = 4,
        interleaved_tokenization: bool = True,
        mamba_skip: bool = True,
        mamba_backend: str = "auto",
        overlap: float = 0.5,
    ):
        self.device = torch.device(device)
        self.overlap = overlap
        self.num_cls = num_cls

        # Build model
        self.model = IMFuse(
            num_cls=num_cls,
            interleaved_tokenization=interleaved_tokenization,
            mamba_skip=mamba_skip,
            mamba_backend=mamba_backend,
        )

        # Load weights
        sd = load_checkpoint(checkpoint, device="cpu")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_volume(
        self,
        modality_volumes: list[np.ndarray | None],
    ) -> np.ndarray:
        """Run inference on preprocessed or raw modality volumes.

        Parameters
        ----------
        modality_volumes : list of 4 ndarrays or None
            [flair, t1ce, t1, t2], each (H, W, D) float or None if missing.
            Raw (un-normalized) volumes — preprocessing is applied internally.

        Returns
        -------
        seg : ndarray (H, W, D) uint8
            Segmentation labels in original spatial coordinates.
            Voxels outside the brain bounding box are labeled 0.
        """
        vol, mask_np, crop_info = preprocess(modality_volumes)
        return self._infer_cropped(vol, mask_np, crop_info)

    @torch.no_grad()
    def predict_nifti(
        self,
        input_paths: dict[str, str | Path],
        output_path: str | Path,
    ) -> np.ndarray:
        """End-to-end NIfTI file inference.

        Parameters
        ----------
        input_paths : dict
            Keys from ``{"flair", "t1ce", "t1", "t2"}``.
            Values are paths to NIfTI files. Missing modalities can be omitted.
        output_path : str | Path
            Where to save the segmentation NIfTI.

        Returns
        -------
        seg : ndarray (H, W, D) uint8
            Segmentation labels in original coordinates.
        """
        # Load modalities
        modality_volumes: list[np.ndarray | None] = [None] * 4
        ref_affine = None
        ref_header = None
        for i, name in enumerate(self.MODALITY_ORDER):
            if name in input_paths and input_paths[name] is not None:
                data, hdr, affine = load_nifti(input_paths[name])
                modality_volumes[i] = data
                if ref_affine is None:
                    ref_affine = affine
                    ref_header = hdr

        if ref_affine is None:
            raise ValueError("At least one modality path must be provided")

        seg = self.predict_volume(modality_volumes)

        save_nifti(seg, output_path, affine=ref_affine, header=ref_header)
        logger.info("Saved segmentation to %s", output_path)
        return seg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _infer_cropped(
        self,
        vol: np.ndarray,
        mask_np: np.ndarray,
        crop_info: CropInfo,
    ) -> np.ndarray:
        """Sliding-window inference on a cropped, normalized volume.

        Returns segmentation in the original (pre-crop) coordinate space.
        """
        # To tensor  (1, 4, H, W, D)
        x = torch.from_numpy(vol).unsqueeze(0).float().to(self.device)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        _, _, H, W, D = x.shape

        # Pad to at least PATCH_SIZE per axis
        pad_h = max(0, PATCH_SIZE - H)
        pad_w = max(0, PATCH_SIZE - W)
        pad_d = max(0, PATCH_SIZE - D)
        if pad_h or pad_w or pad_d:
            x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))

        # Sliding window
        pred = _sliding_window_inference(
            self.model, x, mask, PATCH_SIZE, self.overlap,
        )

        # Remove padding
        pred = pred[:, :, :H, :W, :D]

        # Argmax → segmentation labels
        seg_cropped = pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Place back into original spatial coordinates
        oh, ow, od = crop_info.original_shape
        seg_full = np.zeros((oh, ow, od), dtype=np.uint8)
        seg_full[
            crop_info.x_min : crop_info.x_max,
            crop_info.y_min : crop_info.y_max,
            crop_info.z_min : crop_info.z_max,
        ] = seg_cropped[
            : crop_info.x_max - crop_info.x_min,
            : crop_info.y_max - crop_info.y_min,
            : crop_info.z_max - crop_info.z_min,
        ]
        return seg_full
