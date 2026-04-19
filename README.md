# IM-Fuse Infer

**Standalone inference library for the [IM-Fuse](https://github.com/king-haw/IM-Fuse) brain tumor segmentation model.**

IM-Fuse is a Mamba-based multi-modal MRI fusion architecture for brain tumor segmentation (BraTS). This package provides a clean, minimal inference-only library with both CLI and Python API.

**IM-Fuse 脑肿瘤分割模型的独立推理库。**

IM-Fuse 是一种基于 Mamba 的多模态 MRI 融合架构，用于脑肿瘤分割（BraTS）。本包提供了简洁、轻量的推理库，支持命令行和 Python API 两种使用方式。

---

## Features / 功能特点

- **Sliding-window inference** on 3D NIfTI volumes with configurable overlap / 支持 3D NIfTI 体积的滑动窗口推理，可配置重叠率
- **Automatic modality discovery** — point to a directory and filenames are matched automatically / 自动模态发现 — 指定目录即可自动匹配文件名
- **Dual Mamba backend** — native `mamba_ssm` (CUDA) or vendored pure-PyTorch `mambapy` (CPU/fallback) / 双 Mamba 后端 — 原生 `mamba_ssm`（CUDA）或内置纯 PyTorch `mambapy`（CPU/回退）
- **Checkpoint conversion** between `.pth` and `.safetensors` formats / 支持 `.pth` 与 `.safetensors` 格式之间的检查点转换
- **Zero training dependencies** — only `torch`, `numpy`, `nibabel` required / 零训练依赖 — 仅需 `torch`、`numpy`、`nibabel`

---

## Installation / 安装

### Basic / 基础安装

```bash
pip install imfuse-infer
```

### With CUDA Mamba acceleration / 启用 CUDA Mamba 加速

```bash
pip install imfuse-infer[cuda]
```

### With safetensors support / 启用 safetensors 支持

```bash
pip install imfuse-infer[safetensors]
```

### Full installation / 完整安装

```bash
pip install imfuse-infer[all]
```

### From source / 从源码安装

```bash
git clone https://github.com/mirrorange/imfuse-infer.git
cd imfuse-infer
pip install -e .
```

> **Note / 注意**: Python >= 3.12 is required. / 需要 Python >= 3.12。

---

## Quick Start / 快速开始

### CLI Usage / 命令行使用

#### Predict / 预测

**Auto-discover modalities from a directory / 从目录自动发现模态文件：**

```bash
imfuse-infer predict \
  --input-dir ./patient001 \
  --checkpoint model.pth \
  --output seg.nii.gz
```

The tool scans `--input-dir` and matches NIfTI filenames to modalities (FLAIR, T1ce, T1, T2) by common naming patterns (e.g. `*_flair.nii.gz`, `*_t1ce.nii.gz`).

该工具会扫描 `--input-dir` 并通过常见命名模式（如 `*_flair.nii.gz`、`*_t1ce.nii.gz`）自动匹配 NIfTI 文件到对应模态（FLAIR、T1ce、T1、T2）。

**Specify modality files explicitly / 显式指定模态文件：**

```bash
imfuse-infer predict \
  --flair patient001_flair.nii.gz \
  --t1ce patient001_t1ce.nii.gz \
  --t1 patient001_t1.nii.gz \
  --t2 patient001_t2.nii.gz \
  --checkpoint model.pth \
  --output seg.nii.gz
```

**Mix both — explicit paths override auto-discovered ones / 混合使用 — 显式路径覆盖自动发现的路径：**

```bash
imfuse-infer predict \
  --input-dir ./patient001 \
  --t1ce /other/path/t1ce.nii.gz \
  --checkpoint model.pth \
  --output seg.nii.gz
```

**Full options / 完整选项：**

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | — | Directory containing modality NIfTI files / 包含模态 NIfTI 文件的目录 |
| `--flair` | — | Path to FLAIR NIfTI / FLAIR NIfTI 文件路径 |
| `--t1ce` | — | Path to T1ce NIfTI / T1ce NIfTI 文件路径 |
| `--t1` | — | Path to T1 NIfTI / T1 NIfTI 文件路径 |
| `--t2` | — | Path to T2 NIfTI / T2 NIfTI 文件路径 |
| `-o`, `--output` | *required* | Output segmentation NIfTI path / 输出分割结果路径 |
| `-c`, `--checkpoint` | *required* | Model checkpoint path (.pth or .safetensors) / 模型检查点路径 |
| `--device` | `cuda` | Compute device / 计算设备 |
| `--num-cls` | `4` | Number of output classes / 输出类别数 |
| `--overlap` | `0.5` | Sliding window overlap ratio / 滑动窗口重叠率 |
| `--no-interleaved` | — | Disable interleaved tokenization / 禁用交错分词 |
| `--no-mamba-skip` | — | Disable Mamba skip connections / 禁用 Mamba 跳跃连接 |
| `--mamba-backend` | `auto` | Mamba backend: `auto`, `mamba_ssm`, `mambapy` / Mamba 后端选择 |
| `-v`, `--verbose` | — | Enable verbose logging / 启用详细日志 |

#### Convert checkpoint / 转换检查点

```bash
# .pth → .safetensors
imfuse-infer convert model.pth model.safetensors

# .safetensors → .pth
imfuse-infer convert model.safetensors model.pth
```

---

### Python API / Python 接口

#### Basic prediction / 基础预测

```python
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor(
    checkpoint="model.pth",
    device="cuda",        # or "cpu"
)

# Predict from NIfTI files — returns segmentation as numpy array
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

#### Predict from numpy arrays / 从 numpy 数组预测

```python
import numpy as np
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor(checkpoint="model.pth", device="cuda")

# Load your own volumes as numpy arrays (H, W, D)
# 加载自己的体积数据为 numpy 数组 (H, W, D)
flair = np.load("flair.npy")
t1ce = np.load("t1ce.npy")
t1 = np.load("t1.npy")
t2 = np.load("t2.npy")

# Pass as list of 4 volumes: [flair, t1ce, t1, t2]
# Use None for missing modalities
# 以 4 个体积的列表传入：[flair, t1ce, t1, t2]
# 缺失的模态使用 None
seg = predictor.predict_volume([flair, t1ce, t1, t2])

print(seg.shape)  # (H, W, D)
```

#### Custom model configuration / 自定义模型配置

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

---

## Output Labels / 输出标签

The segmentation output contains the following labels:

分割输出包含以下标签：

| Label | Description |
|-------|-------------|
| 0 | Background / 背景 |
| 1 | Necrotic tumor core (NCR) / 坏死肿瘤核心 |
| 2 | Peritumoral edema (ED) / 瘤周水肿 |
| 3 | GD-enhancing tumor (ET) / 钆增强肿瘤 |

---

## Mamba Backend / Mamba 后端

The model uses [Mamba](https://github.com/state-spaces/mamba) state-space layers for multi-modal fusion. Two backends are supported:

该模型使用 [Mamba](https://github.com/state-spaces/mamba) 状态空间层进行多模态融合。支持两种后端：

| Backend | Requirements | Performance | Notes |
|---------|-------------|-------------|-------|
| `mamba_ssm` | CUDA GPU + `pip install imfuse-infer[cuda]` | Fast (custom CUDA kernels) / 快速（自定义 CUDA 核） | Recommended for production / 生产环境推荐 |
| `mambapy` | No extra deps (vendored) / 无额外依赖（内置） | Slower (pure PyTorch) / 较慢（纯 PyTorch） | Works on CPU, good for testing / 支持 CPU，适合测试 |

With `--mamba-backend auto` (default), the library tries `mamba_ssm` first, then falls back to `mambapy`.

使用 `--mamba-backend auto`（默认值）时，库会优先尝试 `mamba_ssm`，失败后回退到 `mambapy`。

---

## License / 许可证

[MIT](LICENSE)

---

## Acknowledgements / 致谢

This project is based on the [IM-Fuse](https://github.com/king-haw/IM-Fuse) architecture by king-haw.

本项目基于 king-haw 的 [IM-Fuse](https://github.com/king-haw/IM-Fuse) 架构。
