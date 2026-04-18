"""Checkpoint loading utilities — pth and safetensors support."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load model state_dict from a .pth or .safetensors checkpoint.

    Handles:
    - DataParallel ``module.`` prefix stripping
    - ``state_dict`` key extraction from training checkpoint dicts
    - safetensors format (requires safetensors package)

    Returns
    -------
    state_dict : OrderedDict[str, Tensor]
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".safetensors",):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install with: pip install safetensors"
            ) from exc
        sd = load_file(str(path), device=str(device))
    else:
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, OrderedDict):
            sd = ckpt
        else:
            sd = ckpt

    # Strip DataParallel 'module.' prefix
    sd = _strip_module_prefix(sd)
    return sd


def _strip_module_prefix(sd: dict[str, Any]) -> OrderedDict[str, Any]:
    """Remove ``module.`` prefix from keys if present."""
    new_sd: OrderedDict[str, Any] = OrderedDict()
    for k, v in sd.items():
        new_sd[k.removeprefix("module.")] = v
    return new_sd


def convert_to_safetensors(
    src: str | Path,
    dst: str | Path,
    device: str = "cpu",
) -> None:
    """Convert a .pth checkpoint to .safetensors format.

    Parameters
    ----------
    src : path to .pth checkpoint
    dst : output path (.safetensors)
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for conversion. "
            "Install with: pip install safetensors"
        ) from exc

    sd = load_checkpoint(src, device=device)
    save_file(sd, str(dst))
