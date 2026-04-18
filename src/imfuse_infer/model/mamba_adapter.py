"""Mamba backend adapter — unified API for mamba_ssm and mambapy backends."""

from __future__ import annotations

import torch.nn as nn

_BACKEND_CACHE: str | None = None


def _resolve_backend(backend: str) -> str:
    global _BACKEND_CACHE
    if backend != "auto":
        return backend
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE
    try:
        from mamba_ssm import Mamba as _  # noqa: F401

        _BACKEND_CACHE = "mamba_ssm"
    except ImportError:
        _BACKEND_CACHE = "mambapy"
    return _BACKEND_CACHE


def create_mamba(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    backend: str = "auto",
) -> nn.Module:
    """Create a single-layer Mamba SSM block.

    The returned module has the interface ``forward(x: (B,L,D)) -> (B,L,D)``
    and weight names identical to ``mamba_ssm.Mamba``, so state_dicts are
    interchangeable between backends.

    Parameters
    ----------
    d_model : int
        Input / output dimension.
    d_state : int
        SSM state dimension N.
    d_conv : int
        Local convolution kernel width.
    expand : int
        Inner expansion factor (d_inner = expand * d_model).
    backend : str
        ``"auto"`` (try mamba_ssm first, fall back to mambapy),
        ``"mamba_ssm"`` or ``"mambapy"``.
    """
    resolved = _resolve_backend(backend)

    if resolved == "mamba_ssm":
        from mamba_ssm import Mamba

        return Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    # mambapy backend
    from imfuse_infer.model._vendor_mambapy.mamba import MambaBlock, MambaConfig

    config = MambaConfig(
        d_model=d_model,
        n_layers=1,
        d_state=d_state,
        d_conv=d_conv,
        expand_factor=expand,
    )
    return MambaBlock(config)
