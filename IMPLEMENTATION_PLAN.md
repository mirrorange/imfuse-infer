# Implementation Plan: imfuse-infer

## Stage 1: Project Skeleton & Dependencies
**Goal**: Set up project structure, dependencies, and vendor mambapy
**Success Criteria**:
- pyproject.toml with all dependencies configured
- Package installable with `uv pip install -e .`
- Vendored mambapy (MambaBlock + pscan) importable
**Tests**: `python -c "from imfuse_infer.model._vendor_mambapy.mamba import MambaBlock"`
**Status**: Complete

## Stage 2: Model Porting & Mamba Adapter
**Goal**: Port IMFuse model and create Mamba backend adapter
**Success Criteria**:
- IMFuse model instantiable with both backends
- Load baseline checkpoint successfully
- Forward pass produces correct shape output
**Tests**:
- Load checkpoint, verify no missing/unexpected keys
- Forward pass with random input → shape (B, 4, 128, 128, 128)
- mambapy backend matches mamba_ssm backend output (within tolerance)
**Status**: Complete

## Stage 3: IO & Preprocessing
**Goal**: Implement NIfTI I/O and preprocessing pipeline
**Success Criteria**:
- Read NIfTI, preprocess, output matches original pipeline
- Checkpoint load/convert for pth and safetensors
**Tests**:
- Load a BraTS2023 case, preprocess, compare with npy reference
- pth → safetensors → load → same state_dict
**Status**: Complete

## Stage 4: Predictor & Sliding Window Inference
**Goal**: Implement IMFusePredictor with sliding window inference
**Success Criteria**:
- End-to-end NIfTI → NIfTI inference working
- Results match baseline validation (architecture doc section 4/5)
**Tests**:
- Run on a test case, compare segmentation output with baseline
- Missing modality inference produces reasonable output
**Status**: Not Started

## Stage 5: CLI & Final Integration
**Goal**: CLI entry point, final tests, cleanup
**Success Criteria**:
- `imfuse-infer predict` CLI works end-to-end
- `imfuse-infer convert` CLI works
- All tests pass
**Tests**:
- CLI predict on a test case
- CLI convert pth → safetensors
**Status**: Not Started
