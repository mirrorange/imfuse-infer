"""IMFuse model — ported from MV-IM-Fuse/IMFuse.py for inference only.

Changes from original:
- ``from mamba_ssm import Mamba`` replaced with ``create_mamba`` adapter
- Constructor accepts ``mamba_backend`` parameter
- Training-only components (Decoder_sep, is_training branching) removed
- ``forward()`` always returns inference output only
- ``torch.cuda.amp.autocast`` replaced with ``torch.amp.autocast``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from imfuse_infer.model.layers import fusion_prenorm, general_conv3d_prenorm
from imfuse_infer.model.mamba_adapter import create_mamba

# ---------- constants (identical to original) ----------
basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8  # bottleneck spatial resolution
input_patch_size = 128


# ---------- Mamba wrappers ----------

class MambaTrans(nn.Module):
    def __init__(self, channels: int, mamba_backend: str = "auto"):
        super().__init__()
        self.mamba = create_mamba(
            d_model=channels,
            d_state=min(channels, 256),
            d_conv=4,
            expand=2,
            backend=mamba_backend,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mamba(self.norm1(x)) + x
        x = self.head(self.norm2(x)) + x
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim: int, mamba_backend: str = "auto"):
        super().__init__()
        self.dim = dim
        self.mamba = MambaTrans(dim, mamba_backend=mamba_backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        return self.mamba(x)


# ---------- Fusion layers ----------

class MambaFusionLayer(nn.Module):
    def __init__(self, dim: int, num_tokens_fused_representation: int, mamba_backend: str = "auto"):
        super().__init__()
        self.dim = dim
        self.num_tokens_fused_representation = num_tokens_fused_representation
        self.fused_tokens = nn.Parameter(
            torch.randn(1, num_tokens_fused_representation, dim)
        )
        self.mamba_layer = MambaLayer(dim, mamba_backend=mamba_backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        fused_tokens = self.fused_tokens.expand(B, -1, -1)
        x_fused = torch.cat([x, fused_tokens], dim=1)
        x_mamba = self.mamba_layer(x_fused)
        return x_mamba[:, -self.num_tokens_fused_representation :, :]


class MambaFusionCatLayer(nn.Module):
    def __init__(self, dim: int, num_tokens_fused_representation: int, mamba_backend: str = "auto"):
        super().__init__()
        self.dim = dim
        self.num_tokens_fused_representation = num_tokens_fused_representation
        self.fused_tokens = nn.Parameter(
            torch.randn(1, num_tokens_fused_representation, dim)
        )
        self.mamba_layer = MambaLayer(dim, mamba_backend=mamba_backend)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        B = x[0].size(0)
        fused_tokens = self.fused_tokens.expand(B, -1, -1)
        x = torch.stack([*x, fused_tokens], dim=2)  # (B, N, 5, D)
        x = x.view(B, -1, self.dim)  # (B, 5N, D)
        x = self.mamba_layer(x)
        return x[:, 4::5, :]  # take every 5th token starting from index 4


# ---------- Tokenizers ----------

class Tokenize(nn.Module):
    def __init__(self, dims: int, num_modals: int = 4):
        super().__init__()
        self.dims = dims
        self.num_modals = num_modals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.num_modals, dim=1)
        tokens = [
            c.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims)
            for c in chunks
        ]
        return torch.cat(tokens, dim=1)


class TokenizeSep(nn.Module):
    def __init__(self, dims: int, num_modals: int = 4):
        super().__init__()
        self.dims = dims
        self.num_modals = num_modals

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = torch.chunk(x, self.num_modals, dim=1)
        return tuple(
            c.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims)
            for c in chunks
        )


# ---------- Encoder ----------

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1_c1 = nn.Conv3d(1, basic_dims, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type="reflect")
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type="reflect")

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type="reflect")
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type="reflect")
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type="reflect")

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type="reflect")
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type="reflect")
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type="reflect")

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type="reflect")
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type="reflect")
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type="reflect")

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type="reflect")
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type="reflect")
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type="reflect")

    def forward(self, x: torch.Tensor):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5


# ---------- Decoder (fuse only, inference) ----------

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls: int = 4, mamba_skip: bool = False):
        super().__init__()
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type="reflect")

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect")

        self.seg_d4 = nn.Conv3d(basic_dims * 16, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(basic_dims * 8, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(basic_dims * 4, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(basic_dims * 2, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_cls=1 if mamba_skip else num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_cls=1 if mamba_skip else num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_cls=1 if mamba_skip else num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=1 if mamba_skip else num_cls)
        self.mamba_skip = mamba_skip

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)
        return pred


# ---------- Transformer ----------

class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, qkv_bias: bool = False, qk_scale=None, dropout_rate: float = 0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim: int, dropout_rate: float, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim: int, depth: int, heads: int, mlp_dim: int, dropout_rate: float = 0.1, n_levels: int = 1, n_points: int = 4):
        super().__init__()
        self.depth = depth
        self.cross_attention_list = nn.ModuleList()
        self.cross_ffn_list = nn.ModuleList()
        for _ in range(depth):
            self.cross_attention_list.append(
                Residual(PreNormDrop(embedding_dim, dropout_rate, SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate)))
            )
            self.cross_ffn_list.append(
                Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate)))
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


# ---------- Mask ----------

class MaskModal(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y.view(B, -1, H, W, Z)


# ---------- Main model ----------

class IMFuse(nn.Module):
    """IM-Fuse inference model.

    Parameters
    ----------
    num_cls : int
        Number of segmentation classes (4 for BraTS 2020/2023, 5 for BraTS 2015).
    interleaved_tokenization : bool
        Use interleaved (True) vs concatenated (False) tokenization.
    mamba_skip : bool
        Use Mamba fusion in skip connections.
    mamba_backend : str
        ``"auto"``, ``"mamba_ssm"`` or ``"mambapy"``.
    """

    def __init__(
        self,
        num_cls: int = 4,
        interleaved_tokenization: bool = True,
        mamba_skip: bool = True,
        mamba_backend: str = "auto",
    ):
        super().__init__()
        self.interleaved_tokenization = interleaved_tokenization
        self.mamba_skip = mamba_skip

        # --- 4 modality-independent encoders ---
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        if interleaved_tokenization:
            TokenizerClass = TokenizeSep
            MambaFusionLayerClass = MambaFusionCatLayer
        else:
            TokenizerClass = Tokenize
            MambaFusionLayerClass = MambaFusionLayer

        # --- IntraFormer ---
        self.flair_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        # NOTE: decode_conv weights exist in the checkpoint (used during training).
        # We keep them so load_state_dict(strict=False) doesn't lose them silently.
        self.flair_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1ce_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.fused_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))

        self.flair_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1ce_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)

        # --- InterFormer ---
        self.mamba_fusion_layer = MambaFusionLayer(
            dim=transformer_basic_dims,
            num_tokens_fused_representation=patch_size ** 3,
            mamba_backend=mamba_backend,
        )
        self.multimodal_transformer = Transformer(
            embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals,
        )
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16 * num_modals, kernel_size=1, padding=0)

        self.masker = MaskModal()

        # --- Skip connection tokenizers + Mamba fusion ---
        _mb = mamba_backend
        self.tokenize = nn.ModuleList([
            TokenizerClass(dims=8, num_modals=num_modals),
            TokenizerClass(dims=16, num_modals=num_modals),
            TokenizerClass(dims=32, num_modals=num_modals),
            TokenizerClass(dims=64, num_modals=num_modals),
            TokenizerClass(dims=512, num_modals=num_modals),
        ])
        self.mamba_fusion_layers = nn.ModuleList([
            MambaFusionLayerClass(dim=8, num_tokens_fused_representation=128 ** 3, mamba_backend=_mb),
            MambaFusionLayerClass(dim=16, num_tokens_fused_representation=64 ** 3, mamba_backend=_mb),
            MambaFusionLayerClass(dim=32, num_tokens_fused_representation=32 ** 3, mamba_backend=_mb),
            MambaFusionLayerClass(dim=64, num_tokens_fused_representation=16 ** 3, mamba_backend=_mb),
            MambaFusionLayerClass(dim=512, num_tokens_fused_representation=8 ** 3, mamba_backend=_mb),
        ])

        # --- Decoder ---
        self.decoder_fuse = Decoder_fuse(num_cls=num_cls, mamba_skip=mamba_skip)

        # Training-only decoder — kept for checkpoint weight compatibility
        self.decoder_sep = Decoder_sep_stub(num_cls=num_cls)

        # Weight initialization (matches original)
        self.apply(_init_weights_he)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, 4, H, W, D)
            4-modality input volume.
        mask : Tensor (B, 4)
            Boolean mask — True = modality available.

        Returns
        -------
        pred : Tensor (B, num_cls, H, W, D)
            Softmax probability map.
        """
        B = x.size(0)

        # --- Modality-independent encoding ---
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4])

        # --- IntraFormer (bottleneck self-attention per modality) ---
        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, transformer_basic_dims)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, transformer_basic_dims)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, transformer_basic_dims)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, transformer_basic_dims)

        flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
        t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

        flair_intra_x5 = flair_intra_token_x5.view(B, patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(B, patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(B, patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(B, patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        # --- Mask missing modalities ---
        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)

        # --- Mamba Skip fusion ---
        if self.mamba_skip:
            x1 = self.tokenize[-5](x1)
            x1 = self.mamba_fusion_layers[-5](x1)
            x1 = x1.view(B, input_patch_size, input_patch_size, input_patch_size, basic_dims).permute(0, 4, 1, 2, 3).contiguous()

            x2 = self.tokenize[-4](x2)
            x2 = self.mamba_fusion_layers[-4](x2)
            x2 = x2.view(B, input_patch_size // 2, input_patch_size // 2, input_patch_size // 2, basic_dims * 2).permute(0, 4, 1, 2, 3).contiguous()

            x3 = self.tokenize[-3](x3)
            x3 = self.mamba_fusion_layers[-3](x3)
            x3 = x3.view(B, input_patch_size // 4, input_patch_size // 4, input_patch_size // 4, basic_dims * 4).permute(0, 4, 1, 2, 3).contiguous()

            x4 = self.tokenize[-2](x4)
            x4 = self.mamba_fusion_layers[-2](x4)
            x4 = x4.view(B, input_patch_size // 8, input_patch_size // 8, input_patch_size // 8, basic_dims * 8).permute(0, 4, 1, 2, 3).contiguous()

        # --- MambaFusion + InterFormer ---
        multimodal_token_x5 = self.tokenize[-1](x5_intra)
        fused_multimodal = self.mamba_fusion_layers[-1](multimodal_token_x5)
        multimodal_pos = self.fused_pos.expand(B, -1, -1)
        multimodal_inter_token_x5 = self.multimodal_transformer(fused_multimodal, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(
            multimodal_inter_token_x5.view(B, patch_size, patch_size, patch_size, transformer_basic_dims)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        x5_inter = multimodal_inter_x5

        fuse_pred = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        return fuse_pred


# ---------- Stub for training-only decoder (for weight loading compatibility) ----------

class Decoder_sep_stub(nn.Module):
    """Stub that holds decoder_sep weights for checkpoint compatibility but is never called."""

    def __init__(self, num_cls: int = 4):
        super().__init__()
        self.d4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type="reflect")

        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect")

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)


# ---------- Weight initialization ----------

def _init_weights_he(module: nn.Module, neg_slope: float = 1e-2) -> None:
    if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=neg_slope)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
