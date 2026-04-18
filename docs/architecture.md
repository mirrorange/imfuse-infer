# imfuse-infer 架构设计文档

## 1. 项目目标

创建一个独立的 Python 推理库 `imfuse-infer`，用于 IM-Fuse 脑肿瘤分割模型的推理。

**核心需求**：

- 输入：标准化 NIfTI（`.nii` / `.nii.gz`）图像，支持 1–4 个模态及任意缺失组合
- 输出：分割结果 NIfTI 图像
- 兼容原始 IM-Fuse 仓库的全部模型参数与权重
- 支持 `mamba_ssm`（CUDA）和 `mambapy`（纯 PyTorch，CPU/MPS 友好）两种 Mamba 后端
- 支持 `.pth` 和 `.safetensors` 格式的模型加载
- 提供 Python API 和 CLI 推理入口

## 2. 源码分析摘要

### 2.1 IM-Fuse 模型架构

| 组件 | 说明 |
|------|------|
| 4×Encoder | 模态独立的 5 层 3D 卷积编码器，通道数 8→16→32→64→128 |
| IntraFormer | 瓶颈层各模态独立 Self-Attention (dim=512) |
| MaskModal | 缺失模态掩码：编码后、融合前将缺失模态特征置零 |
| Tokenize/TokenizeSep | 将 4 模态特征序列化为 Mamba 输入 token |
| MambaFusionLayer/MambaFusionCatLayer | Mamba SSM 跨模态融合 |
| InterFormer | 瓶颈层跨模态 Mamba 融合 + Transformer |
| Decoder_fuse | 融合解码器，含 skip connection + 金字塔辅助预测 |

**关键构造参数**：

```python
IMFuse(num_cls=4, interleaved_tokenization=False, mamba_skip=False)
```

| 参数 | 作用 |
|------|------|
| `num_cls` | 分割类别数（BraTS2020/2023=4, BraTS2015=5） |
| `interleaved_tokenization` | 交错 token 排布（True）vs 拼接排布（False） |
| `mamba_skip` | skip connection 中是否使用 Mamba 融合 |

**基线检查点配置**：`interleaved_tokenization=True, mamba_skip=True`

### 2.2 Mamba 依赖分析

IM-Fuse 中所有 Mamba 使用均为：

```python
from mamba_ssm import Mamba
mamba = Mamba(d_model=channels, d_state=min(channels, 256), d_conv=4, expand=2)
# forward: (B, L, D) → (B, L, D)
```

#### mamba_ssm.Mamba（CUDA 后端）

- 单层 SSM block，构造参数为关键字参数
- `forward(hidden_states)` → `(B, L, D)`
- 依赖 CUDA 扩展（`selective_scan_cuda`, `causal_conv1d`）

#### mambapy.MambaBlock（纯 PyTorch 后端）

- 单层 SSM block，但构造函数接受 `MambaConfig` dataclass
- `forward(x)` → `(B, L, D)`
- **权重参数名和 shape 与 mamba_ssm 完全一致**（`in_proj`, `conv1d`, `x_proj`, `dt_proj`, `A_log`, `D`, `out_proj`）
- 不兼容处：构造函数 API 不同（`expand` vs `expand_factor`），`MambaBlock` 外无 `LayerNorm`/残差连接

**兼容性结论**：可通过薄适配层桥接，权重可直接互转。

### 2.3 预处理流程

```
NII 原始数据 (H_raw, W_raw, D_raw) per modality
  → medpy.io.load() 读取 4 模态
  → stack → (4, H, W, D) float32
  → crop to brain bounding box (每轴 ≥ 128)
  → per-modality z-score 归一化（仅在脑区 mask 上计算 μ/σ）
  → 存储为 (H', W', D', 4) channels-last npy
```

### 2.4 推理流程

```
加载 npy → (4, H', W', D') channels-first
  → pad 到每轴 ≥ 128
  → 滑窗（window=128, overlap=0.5）推理
  → 重叠区域加权平均
  → 裁回原始体积尺寸
  → argmax → 分割标签图
```

## 3. 架构设计

### 3.1 包结构

```
imfuse-infer/
├── pyproject.toml
├── main.py                          # CLI 入口
├── src/
│   └── imfuse_infer/
│       ├── __init__.py              # 公开 API：IMFusePredictor
│       ├── predictor.py             # 核心推理器（Python API）
│       ├── model/
│       │   ├── __init__.py
│       │   ├── imfuse.py            # IMFuse 模型定义（移植自原仓库）
│       │   ├── layers.py            # 基础层（general_conv3d_prenorm 等）
│       │   └── mamba_adapter.py     # Mamba 后端适配层
│       ├── io/
│       │   ├── __init__.py
│       │   ├── nifti.py             # NIfTI 读写
│       │   └── checkpoint.py        # 模型加载（pth / safetensors / DataParallel 兼容）
│       └── preprocessing/
│           ├── __init__.py
│           └── normalize.py         # 预处理：归一化 + crop + pad
└── tests/
    ├── test_mamba_adapter.py        # Mamba 适配层测试
    ├── test_preprocessing.py        # 预处理测试
    └── test_predictor.py            # 端到端推理测试
```

### 3.2 模块职责

#### `predictor.py` — IMFusePredictor（核心公开类）

```python
class IMFusePredictor:
    """IM-Fuse 推理器，提供端到端 NIfTI → NIfTI 推理能力。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",            # "auto" | "cuda" | "cpu" | "mps"
        mamba_backend: str = "auto",      # "auto" | "mamba_ssm" | "mambapy"
        num_cls: int = 4,
        interleaved_tokenization: bool = True,
        mamba_skip: bool = True,
        patch_size: int = 128,
        overlap: float = 0.5,
    ): ...

    def predict(
        self,
        inputs: dict[str, str | Path],   # {"flair": "path.nii.gz", "t1ce": ..., ...}
        output_path: str | Path | None = None,
        *,
        modality_mask: list[bool] | None = None,  # 显式覆盖缺失模态掩码
    ) -> np.ndarray:
        """
        执行推理。

        inputs: 模态名 → NIfTI 文件路径的映射。
                支持的模态名：flair, t1ce, t1, t2
                允许部分模态缺失，缺失模态自动从 mask 中排除。

        output_path: 若提供，将分割结果保存为 NIfTI 文件，
                     使用第一个输入模态的 affine 和 header。

        返回值: np.ndarray, shape (H, W, D), dtype uint8, 值为 0-3（或 0-4）
        """
        ...

    def predict_volume(
        self,
        volume: np.ndarray,               # (C, H, W, D) 或 (H, W, D, C)，已归一化
        modality_mask: list[bool],         # 长度 4 的 bool 列表
    ) -> np.ndarray:
        """
        对已预处理的体积直接推理（跳过 NIfTI 读取和预处理）。
        
        volume: 4 模态体积，channels-first (4, H, W, D) 或 channels-last (H, W, D, 4)
        modality_mask: [flair可用, t1ce可用, t1可用, t2可用]

        返回值: np.ndarray, shape (H, W, D), dtype uint8
        """
        ...
```

**设计决策**：

- `predict()` 接受模态名 → 文件路径的 dict，而非固定 4 文件列表。缺失模态只需不传入对应 key，自动推导 mask。
- `predict_volume()` 适用于已完成预处理的场景（如批量处理管线）。
- `device="auto"` 策略：CUDA 可用 → CUDA，MPS 可用 → MPS，否则 CPU。
- `mamba_backend="auto"` 策略：`mamba_ssm` 可导入 → 使用 `mamba_ssm`，否则回退到 `mambapy`。

#### `model/mamba_adapter.py` — Mamba 后端适配

```python
def create_mamba(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    backend: str = "auto",
) -> nn.Module:
    """
    创建 Mamba SSM block，统一 API。

    backend="auto": 尝试 mamba_ssm，失败则回退到 mambapy
    backend="mamba_ssm": 强制使用 mamba_ssm（需 CUDA）
    backend="mambapy": 强制使用 mambapy（纯 PyTorch）

    返回的 nn.Module 接口：forward(x: (B,L,D)) → (B,L,D)
    权重参数名与 mamba_ssm.Mamba 完全一致，可互相加载。
    """
    ...
```

**适配策略**：

1. 当 `backend="mamba_ssm"` 时，直接返回 `mamba_ssm.Mamba(d_model, d_state, d_conv, expand)`。
2. 当 `backend="mambapy"` 时，创建 `mambapy.mamba.MambaBlock(MambaConfig(d_model, n_layers=1, d_state, d_conv, expand_factor=expand))`。由于 `MambaBlock` 与 `mamba_ssm.Mamba` 的权重参数名一致，无需额外转换。
3. 当 `backend="auto"` 时，先尝试 `mamba_ssm`，`ImportError` 时回退到 `mambapy`。

**注意**：`mamba_ssm.Mamba` 的 `forward` 参数名为 `hidden_states`，`mambapy.MambaBlock` 的为 `x`，但这对调用方透明（位置参数）。

#### `model/imfuse.py` — 模型定义

从原仓库 `IMFuse.py` 和 `layers.py` 移植，做以下修改：

1. 将 `from mamba_ssm import Mamba` 替换为 `from .mamba_adapter import create_mamba`
2. 所有 `Mamba(d_model=..., d_state=..., d_conv=4, expand=2)` 调用替换为 `create_mamba(d_model=..., d_state=..., d_conv=4, expand=2, backend=self.mamba_backend)`
3. 构造函数增加 `mamba_backend` 参数，透传至 `create_mamba`
4. 移除训练专用组件（`Decoder_sep`、金字塔辅助输出），`forward` 仅返回 `fuse_pred`
5. 移除 `is_training` 状态管理，固定为推理模式

**模型常量保持不变**：

```python
basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8
```

#### `io/checkpoint.py` — 模型加载

```python
def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    strict: bool = True,
) -> None:
    """
    加载检查点到模型。

    支持格式：
    - .pth / .pt: PyTorch checkpoint（兼容 DataParallel 的 'module.' 前缀）
    - .safetensors: safetensors 格式

    处理逻辑：
    1. 检测文件格式
    2. 加载 state_dict：
       - pth: 从 checkpoint['state_dict'] 或直接 state_dict 加载
       - safetensors: 使用 safetensors.torch.load_file()
    3. 去除 DataParallel 'module.' 前缀
    4. 过滤训练专用权重（decoder_sep 等）
    5. 加载到模型
    """
    ...


def convert_checkpoint_to_safetensors(
    pth_path: str | Path,
    output_path: str | Path,
) -> None:
    """将 .pth 检查点转换为 .safetensors 格式。"""
    ...
```

#### `io/nifti.py` — NIfTI 读写

```python
def load_nifti(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Header, np.ndarray]:
    """
    读取 NIfTI 文件。

    返回: (data, header, affine)
    - data: np.ndarray, 原始体素数据
    - header: NIfTI header（用于保存输出时保持元信息）
    - affine: 4×4 仿射矩阵
    """
    ...


def save_nifti(
    data: np.ndarray,
    path: str | Path,
    affine: np.ndarray,
    header: nib.Nifti1Header | None = None,
) -> None:
    """保存 NIfTI 文件。"""
    ...
```

#### `preprocessing/normalize.py` — 预处理

```python
def preprocess_nifti_volumes(
    volumes: dict[str, np.ndarray],
    modality_order: tuple[str, ...] = ("flair", "t1ce", "t1", "t2"),
) -> tuple[np.ndarray, list[bool]]:
    """
    对 NIfTI 体积执行 IM-Fuse 标准预处理。

    volumes: 模态名 → 3D 体积的映射（原始体素值）
    modality_order: 模态排列顺序

    处理步骤：
    1. 对缺失模态填充全零体积
    2. Stack → (4, H, W, D)
    3. Crop to brain bounding box（每轴 ≥ 128）
    4. Per-modality z-score 归一化（仅在脑区上计算）
    5. 生成模态可用性 mask

    返回: (volume, mask)
    - volume: (4, H', W', D') float32, channels-first
    - mask: [bool × 4]，True = 该模态可用
    """
    ...
```

### 3.3 CLI 设计

```
imfuse-infer predict \
    --checkpoint model.pth \
    --flair flair.nii.gz \
    --t1ce t1ce.nii.gz \
    --t1 t1.nii.gz \
    --t2 t2.nii.gz \
    --output segmentation.nii.gz \
    [--device auto] \
    [--mamba-backend auto] \
    [--num-cls 4] \
    [--interleaved-tokenization] \
    [--no-interleaved-tokenization] \
    [--mamba-skip] \
    [--no-mamba-skip] \
    [--patch-size 128] \
    [--overlap 0.5]

imfuse-infer convert \
    --input model_last.pth \
    --output model_last.safetensors
```

- 模态参数均为可选，缺失模态只需不传对应 flag
- `--interleaved-tokenization` 和 `--mamba-skip` 默认为 True（匹配基线检查点）

CLI 入口位于 `main.py`，通过 `pyproject.toml` 的 `[project.scripts]` 注册为 `imfuse-infer` 命令。

### 3.4 依赖关系

```toml
[project]
dependencies = [
    "torch>=2.0",
    "numpy",
    "nibabel",
]

[project.optional-dependencies]
cuda = ["mamba-ssm>=2.0", "causal-conv1d"]
safetensors = ["safetensors"]
all = ["mamba-ssm>=2.0", "causal-conv1d", "safetensors"]
```

- `mambapy` 作为内部代码嵌入（vendor），不作为外部依赖。原因：该包无 PyPI 发布，且只需 `MambaBlock` + `pscan` 两个文件。
- `mamba_ssm` 作为可选依赖，仅在 CUDA 环境下安装。
- `nibabel` 替代原仓库的 `medpy`，更轻量且是 NIfTI 的标准库。

### 3.5 mambapy Vendor 策略

从 `mamba.py` 仓库提取必要文件，放置于：

```
src/imfuse_infer/model/_vendor_mambapy/
    __init__.py
    mamba.py      # MambaBlock, MambaConfig, RMSNorm
    pscan.py      # 并行扫描实现
```

修改内容：
- 仅保留 `MambaBlock`、`MambaConfig`、`RMSNorm`、`pscan`
- 移除 `Mamba`（多层容器）、`ResidualBlock`（不需要）
- 移除 `use_cuda` 相关代码（CUDA 路径由 `mamba_ssm` 后端负责）
- 调整 import 路径

## 4. 数据流

### 4.1 端到端 Python API 调用

```python
from imfuse_infer import IMFusePredictor

predictor = IMFusePredictor("model.safetensors", device="cuda")

seg = predictor.predict(
    inputs={
        "flair": "patient001_flair.nii.gz",
        "t1ce": "patient001_t1ce.nii.gz",
        # t1 和 t2 缺失
    },
    output_path="patient001_seg.nii.gz",
)
# seg: np.ndarray, shape (H, W, D), dtype uint8
```

### 4.2 内部处理流程

```
1. load_nifti() × N 个可用模态
   → {modality_name: (H, W, D) ndarray} + affine/header

2. preprocess_nifti_volumes()
   → 填充缺失模态为全零
   → stack (4, H, W, D)
   → crop to brain bbox (每轴 ≥ 128)
   → per-modality z-score on brain mask
   → 返回 volume (4, H', W', D'), mask [bool×4]

3. sliding_window_inference()
   → pad 到每轴 ≥ 128
   → 滑窗 (128³, overlap=0.5)
   → model.forward(patch, mask) → (B, num_cls, 128, 128, 128)
   → 重叠区域加权平均
   → 裁回原始体积尺寸
   → argmax → (H', W', D') uint8

4. uncrop to original volume size
   → 嵌入回原始 NIfTI 空间

5. save_nifti() → 输出 .nii.gz
```

### 4.3 体积空间处理细节

crop-to-brain 操作会裁剪原始体积，推理后需要将分割结果嵌回原始空间：

```
原始 NIfTI: (H_raw, W_raw, D_raw)
  → crop bbox: (x_min:x_max, y_min:y_max, z_min:z_max)
  → 裁剪后: (H', W', D')
  → 推理得到 seg: (H', W', D')
  → 嵌回: result = zeros(H_raw, W_raw, D_raw); result[x_min:x_max, y_min:y_max, z_min:z_max] = seg
```

## 5. 权重兼容性

### 5.1 DataParallel 前缀处理

原始检查点通过 `nn.DataParallel` 保存，所有 key 带 `module.` 前缀：

```python
# 原始: module.flair_encoder.e1_c1.conv.weight
# 目标: flair_encoder.e1_c1.conv.weight
state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
```

### 5.2 训练专用权重过滤

推理模型不包含 `Decoder_sep`，需过滤以下 key 前缀：

```python
skip_prefixes = [
    "decoder_sep.",
    "flair_decode_conv.", "t1ce_decode_conv.", "t1_decode_conv.", "t2_decode_conv.",
    # 金字塔辅助预测器（推理模型中可省略，但保留也无害）
]
```

### 5.3 Mamba 权重互换

`mamba_ssm.Mamba` 和 `mambapy.MambaBlock` 的权重 key 完全一致：

| Key | Shape |
|-----|-------|
| `in_proj.weight` | `(2*d_inner, d_model)` |
| `conv1d.weight` | `(d_inner, 1, d_conv)` |
| `conv1d.bias` | `(d_inner,)` |
| `x_proj.weight` | `(dt_rank + 2*d_state, d_inner)` |
| `dt_proj.weight` | `(d_inner, dt_rank)` |
| `dt_proj.bias` | `(d_inner,)` |
| `A_log` | `(d_inner, d_state)` |
| `D` | `(d_inner,)` |
| `out_proj.weight` | `(d_model, d_inner)` |

**无需任何权重转换即可在两个后端间切换。**

## 6. 可行性分析

### 6.1 可行

| 项目 | 评估 |
|------|------|
| 模型移植 | IMFuse 模型为标准 PyTorch，可直接移植 |
| Mamba 后端切换 | 权重名一致，API 差异可通过适配层消除 |
| NIfTI I/O | nibabel 成熟稳定 |
| 滑窗推理 | 逻辑明确，已在原仓库验证 |
| safetensors 支持 | 标准库操作 |
| CPU/MPS 推理 | mambapy 为纯 PyTorch，天然支持 |

### 6.2 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| mambapy pscan 要求序列长度 pad 到 2 的幂 | 3D 医学图像 token 序列长度通常非 2 的幂（如 128³=2097152），pad 内存开销大 | 瓶颈层 8³=512 是 2 的幂（OK）；skip 层如 128³ 已是 2 的幂（OK）；64³=262144 是 2 的幂（OK）；32³=32768 是 2 的幂（OK）；16³=4096 是 2 的幂（OK）。**所有层的 token 序列长度恰好均为 2 的幂，无额外 pad 开销** |
| mambapy CPU 推理性能 | 大体积 3D 推理慢 | 可接受：推理场景非实时性要求；可通过减小 overlap、增大 patch 优化 |
| 预处理归一化差异 | 原仓库使用 medpy 读取，我们改用 nibabel | medpy 底层也调用 nibabel，数据值一致；关键是归一化算法一致 |

### 6.3 不纳入本库的功能

- 训练（`Decoder_sep`、损失函数、数据增强）
- 评估指标计算（Dice score）
- `IMFuse_no1skip` 变体（原仓库基线未使用）
- `BidirectionalMamba.py` / `UnidirectionalMamba.py`（原仓库 IMFuse 未使用，属于独立模块）

## 7. 测试策略

| 测试 | 内容 |
|------|------|
| `test_mamba_adapter.py` | 验证 mambapy 后端与 mamba_ssm 后端在相同权重下输出一致（需 CUDA 环境） |
| `test_preprocessing.py` | 验证预处理与原仓库 `preprocess.py` 输出一致 |
| `test_predictor.py` | 端到端推理测试：加载基线检查点，对测试样本推理，与基线结果对比 |
| `test_checkpoint.py` | pth/safetensors 加载、DataParallel 前缀去除、权重过滤 |

## 8. 已确认事项

1. **mambapy vendor 许可**：mamba.py 仓库使用 MIT 协议，可自由嵌入。需在代码中保留许可声明。
2. **不需要支持 `IMFuse_no1skip` 变体**：仅支持基线检查点对应的 `IMFuse`（完整版）。
3. **输出标签不做映射**：BraTS 2023 已将 ET 标签值从 4 改为 3，模型输出 `{0,1,2,3}` 即为最终标签。
4. **输出 NIfTI 使用第一个输入模态的 affine 和 header**：已确认。
