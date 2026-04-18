"""NIfTI I/O helpers using nibabel."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Header, np.ndarray]:
    """Load a NIfTI volume.

    Returns
    -------
    data : ndarray
        Volume data.
    header : Nifti1Header
        NIfTI header (for re-use when saving output).
    affine : ndarray (4, 4)
        Affine matrix.
    """
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.header, img.affine


def save_nifti(
    data: np.ndarray,
    path: str | Path,
    affine: np.ndarray | None = None,
    header: nib.Nifti1Header | None = None,
) -> None:
    """Save a volume as NIfTI.

    Parameters
    ----------
    data : ndarray
        Volume to save (e.g. uint8 segmentation map).
    path : str | Path
        Output path (.nii or .nii.gz).
    affine : ndarray (4, 4), optional
        Affine matrix. Defaults to identity.
    header : Nifti1Header, optional
        Header to copy metadata from.
    """
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine, header=header)
    nib.save(img, str(path))
