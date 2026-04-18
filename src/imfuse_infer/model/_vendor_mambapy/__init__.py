# Vendored from mamba.py (https://github.com/alxndrTL/mamba.py)
# MIT License - Copyright (c) 2024 Alexandre TL
#
# Only MambaBlock, MambaConfig, RMSNorm and pscan are included.
# Modifications:
# - Removed multi-layer Mamba/ResidualBlock (not needed for single-block usage)
# - Removed use_cuda fallback code
# - Adjusted import paths

from imfuse_infer.model._vendor_mambapy.mamba import MambaBlock, MambaConfig, RMSNorm

__all__ = ["MambaBlock", "MambaConfig", "RMSNorm"]
