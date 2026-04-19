<div align="center">

# IM-Fuse Infer

**Standalone inference library for the [IM-Fuse](https://github.com/AImageLab-zip/IM-Fuse) brain tumor segmentation model**

[![PyPI](https://img.shields.io/pypi/v/imfuse-infer)](https://pypi.org/project/imfuse-infer/)
[![Python](https://img.shields.io/pypi/pyversions/imfuse-infer)](https://pypi.org/project/imfuse-infer/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](#english) · [中文](#中文)

</div>

---

## English

### Overview

IM-Fuse is a Mamba-based multi-modal MRI fusion architecture for brain tumor segmentation (BraTS). This package provides a clean, minimal inference-only library with both CLI and Python API.

### Features

- **Sliding-window inference** on 3D NIfTI volumes with configurable overlap
- **Automatic modality discovery** — point to a directory and filenames are matched automatically
- **Dual Mamba backend** — native `mamba_ssm` (CUDA) or vendored pure-PyTorch `mambapy` (CPU/fallback)
- **Checkpoint conversion** between `.pth` and `.safetensors` formats
- **Zero training dependencies** — only `torch`, `numpy`, `nibabel` required

### Installation

#### Basic

```bash
pip install imfuse-infer
```

#### With CUDA Mamba acceleration

```bash
pip install imfuse-infer[cuda]
```

#### With safetensors support

```bash
pip install imfuse-infer[safetensors]
```

#### Full installation

```bash
pip install imfuse-infer[all]
```

#### From source

```bash
git clone https://github.com/mirrorange/imfuse-infer.git
cd imfuse-infer
pip install -e .
```

> **Note**: Python >= 3.12 is required.

### Quick Start

#### CLI Usage

##### Predict

**Auto-discover modalities from a directory:**

```bash
imfuse-infer predict \
  --input-dir ./patient001 \
  --checkpoint model.pth \
  --output seg.nii.gz
```

The tool scans `--input-dir` and matches NIfTI filenames to modalities (FLAIR, T1ce, T1, T2) by common naming patterns (e.g. `*_flair.nii.gz`, `*_t1ce.nii.gz`).

**Specify modality files explicitly:**

```bash
imfuse-infer predict \
  --flair patient001_flair.nii.gz \
  --t1ce patient001_t1ce.nii.gz \
  --t1 patient001_t1.nii.gz \
  --t2 patient001_t2.nii.gz \
  --checkpoint model.pth \
  --output seg.nii.gz
```

**Mix both — explicit paths override auto-discovered ones:**

```bash
imfuse-infer predict \
  --input-dir ./patient001 \
  --t1ce /other/path/t1ce.nii.gz \
  --checkpoint model.pth \
  --output seg.nii.gz
```

**Full options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | — | Directory containing modality NIfTI files |
| `--flair` | — | Path to FLAIR NIfTI |
| `--t1ce` | — | Path to T1ce NIfTI |
| `--t1` | — | Path to T1 NIfTI |
| `--t2` | — | Path to T2 NIfTI |
| `-o`, `--output` | *required* | Output segmentation NIfTI path |
| `-c`, `--checkpoint` | *required* | Model checkpoint path (.pth or .safetensors) |
| `--device` | `cuda` | Compute device |
| `--num-cls` | `4` | Number of output classes |
| `--overlap` | `0.5` | Sliding window overlap ratio |
| `--no-interleaved` | — | Disable interleaved tokenization |
| `--no-mamba-skip` | — | Disable Mamba skip connections |
| `--mamba-backend` | `auto` | Mamba backend: `auto`, `mamba_ssm`, `mambapy` |
| `-v`, `--verbose` | — | Enable verbose logging |

##### Convert checkpoint

```bash
# .pth → .safetensors
imfuse-infer convert model.pth model.safetensors

# .safetensors → .pth
imfuse-infer convert model.safetensors model.pth
```

#### Python API

##### Basic prediction

```python
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor(
    checkpoint="model.pth",
    device="cuda",  # or "cpu"
)

# Predict from NIfTI files — returns segmentation as numpy array
seg = predictor.predict_nifti(
    input_paths={
        "flair": "patient001_flair.nii.gz",
        "t1ce": "patient001_t1ce.nii.gz",
        "t1": "patient001_t1.nii.gz",
        "t2": "patient001_t2.nii.gz",
    },
    output_path="seg.nii.gz",
)

print(seg.shape)  # (H, W, D)
```

##### Predict from numpy arrays

```python
import numpy as np
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor(checkpoint="model.pth", device="cuda")

# Load your own volumes as numpy arrays (H, W, D)
flair = np.load("flair.npy")
t1ce = np.load("t1ce.npy")
t1 = np.load("t1.npy")
t2 = np.load("t2.npy")

# Pass as list of 4 volumes: [flair, t1ce, t1, t2]
# Use None for missing modalities
seg = predictor.predict_volume([flair, t1ce, t1, t2])

print(seg.shape)  # (H, W, D)
```

##### Custom model configuration

```python
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor(
    checkpoint="model.safetensors",
    device="cuda",
    num_cls=4,
    interleaved_tokenization=True,
    mamba_skip=True,
    mamba_backend="auto",  # "auto" | "mamba_ssm" | "mambapy"
    overlap=0.5,
)
```

### Output Labels

| Label | Description |
|-------|-------------|
| 0 | Background |
| 1 | Necrotic tumor core (NCR) |
| 2 | Peritumoral edema (ED) |
| 3 | GD-enhancing tumor (ET) |

### Mamba Backend

The model uses [Mamba](https://github.com/state-spaces/mamba) state-space layers for multi-modal fusion. Two backends are supported:

| Backend | Requirements | Performance | Notes |
|---------|-------------|-------------|-------|
| `mamba_ssm` | CUDA GPU + `pip install imfuse-infer[cuda]` | Fast (custom CUDA kernels) | Recommended for production |
| `mambapy` | No extra deps (vendored) | Slower (pure PyTorch) | Works on CPU, good for testing |

With `--mamba-backend auto` (default), the library tries `mamba_ssm` first, then falls back to `mambapy`.

### References

> Vittorio Pipoli, Alessia Saporita, Kevin Marchesini, Costantino Grana, Elisa Ficarra, Federico Bolelli. *IM-Fuse: A Mamba-based Fusion Block for Brain Tumor Segmentation with Incomplete Modalities.* 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2025.

- Paper: https://federicobolelli.it/pub_files/2025miccai_imfuse.html
- Original code: https://github.com/AImageLab-zip/IM-Fuse

### License

[MIT](LICENSE)

---

## 中文

### 概述

IM-Fuse 是一种基于 Mamba 的多模态 MRI 融合架构，用于脑肿瘤分割（BraTS）。本包提供了简洁、轻量的推理库，支持命令行和 Python API 两种使用方式。

### 功能特点

- **滑动窗口推理** — 支持 3D NIfTI 体积的滑动窗口推理，可配置重叠率
- **自动模态发现** — 指定目录即可自动匹配文件名
- **双 Mamba 后端** — 原生 `mamba_ssm`（CUDA）或内置纯 PyTorch `mambapy`（CPU/回退）
- **检查点转换** — 支持 `.pth` 与 `.safetensors` 格式之间的转换
- **零训练依赖** — 仅需 `torch`、`numpy`、`nibabel`

### 安装

#### 基础安装

```bash
pip install imfuse-infer
```

#### 启用 CUDA Mamba 加速

```bash
pip install imfuse-infer[cuda]
```

#### 启用 safetensors 支持

```bash
pip install imfuse-infer[safetensors]
```

#### 完整安装

```bash
pip install imfuse-infer[all]
```

#### 从源码安装

```bash
git clone https://github.com/mirrorange/imfuse-infer.git
cd imfuse-infer
pip install -e .
```

> **注意**：需要 Python >= 3.12。

### 快速开始

#### 命令行使用

##### 预测

**从目录自动发现模态文件：**

```bash
imfuse-infer predict \
  --input-dir ./patient001 \
  --checkpoint model.pth \
  --output seg.nii.gz
```

该工具会扫描 `--input-dir` 并通过常见命名模式（如 `*_flair.nii.gz`、`*_t1ce.nii.gz`）自动匹配 NIfTI 文件到对应模态（FLAIR、T1ce、T1、T2）。

**显式指定模态文件：**

```bash
imfuse-infer predict \
  --flair patient001_flair.nii.gz \
  --t1ce patient001_t1ce.nii.gz \
  --t1 patient001_t1.nii.gz \
  --t2 patient001_t2.nii.gz \
  --checkpoint model.pth \
  --output seg.nii.gz
```

**混合使用 — 显式路径覆盖自动发现的路径：**

```bash
imfuse-infer predict \
  --input-dir ./patient001 \
  --t1ce /other/path/t1ce.nii.gz \
  --checkpoint model.pth \
  --output seg.nii.gz
```

**完整选项：**

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | — | 包含模态 NIfTI 文件的目录 |
| `--flair` | — | FLAIR NIfTI 文件路径 |
| `--t1ce` | — | T1ce NIfTI 文件路径 |
| `--t1` | — | T1 NIfTI 文件路径 |
| `--t2` | — | T2 NIfTI 文件路径 |
| `-o`, `--output` | *必填* | 输出分割结果路径 |
| `-c`, `--checkpoint` | *必填* | 模型检查点路径（.pth 或 .safetensors） |
| `--device` | `cuda` | 计算设备 |
| `--num-cls` | `4` | 输出类别数 |
| `--overlap` | `0.5` | 滑动窗口重叠率 |
| `--no-interleaved` | — | 禁用交错分词 |
| `--no-mamba-skip` | — | 禁用 Mamba 跳跃连接 |
| `--mamba-backend` | `auto` | Mamba 后端：`auto`、`mamba_ssm`、`mambapy` |
| `-v`, `--verbose` | — | 启用详细日志 |

##### 转换检查点

```bash
# .pth → .safetensors
imfuse-infer convert model.pth model.safetensors

# .safetensors → .pth
imfuse-infer convert model.safetensors model.pth
```

#### Python 接口

##### 基础预测

```python
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor(
    checkpoint="model.pth",
    device="cuda",  # 或 "cpu"
)

# 从 NIfTI 文件预测 — 返回分割结果 numpy 数组
seg = predictor.predict_nifti(
    input_paths={
        "flair": "patient001_flair.nii.gz",
        "t1ce": "patient001_t1ce.nii.gz",
        "t1": "patient001_t1.nii.gz",
        "t2": "patient001_t2.nii.gz",
    },
    output_path="seg.nii.gz",
)

print(seg.shape)  # (H, W, D)
```

##### 从 numpy 数组预测

```python
import numpy as np
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor(checkpoint="model.pth", device="cuda")

# 加载自己的体积数据为 numpy 数组 (H, W, D)
flair = np.load("flair.npy")
t1ce = np.load("t1ce.npy")
t1 = np.load("t1.npy")
t2 = np.load("t2.npy")

# 以 4 个体积的列表传入：[flair, t1ce, t1, t2]
# 缺失的模态使用 None
seg = predictor.predict_volume([flair, t1ce, t1, t2])

print(seg.shape)  # (H, W, D)
```

##### 自定义模型配置

```python
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor(
    checkpoint="model.safetensors",
    device="cuda",
    num_cls=4,
    interleaved_tokenization=True,
    mamba_skip=True,
    mamba_backend="auto",  # "auto" | "mamba_ssm" | "mambapy"
    overlap=0.5,
)
```

### 输出标签

| 标签 | 说明 |
|------|------|
| 0 | 背景 |
| 1 | 坏死肿瘤核心（NCR） |
| 2 | 瘤周水肿（ED） |
| 3 | 钆增强肿瘤（ET） |

### Mamba 后端

该模型使用 [Mamba](https://github.com/state-spaces/mamba) 状态空间层进行多模态融合。支持两种后端：

| 后端 | 要求 | 性能 | 说明 |
|------|------|------|------|
| `mamba_ssm` | CUDA GPU + `pip install imfuse-infer[cuda]` | 快速（自定义 CUDA 核） | 生产环境推荐 |
| `mambapy` | 无额外依赖（内置） | 较慢（纯 PyTorch） | 支持 CPU，适合测试 |

使用 `--mamba-backend auto`（默认值）时，库会优先尝试 `mamba_ssm`，失败后回退到 `mambapy`。

### 引用

> Vittorio Pipoli, Alessia Saporita, Kevin Marchesini, Costantino Grana, Elisa Ficarra, Federico Bolelli. *IM-Fuse: A Mamba-based Fusion Block for Brain Tumor Segmentation with Incomplete Modalities.* 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2025.

- 论文：https://federicobolelli.it/pub_files/2025miccai_imfuse.html
- 原始代码：https://github.com/AImageLab-zip/IM-Fuse

### 许可证

[MIT](LICENSE)
