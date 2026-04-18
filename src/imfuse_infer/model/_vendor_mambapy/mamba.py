# Vendored from mamba.py (https://github.com/alxndrTL/mamba.py)
# MIT License - Copyright (c) 2024 Alexandre TL
#
# Single-block Mamba SSM implementation in pure PyTorch.
# Only MambaBlock (single layer), MambaConfig, and RMSNorm are included.
# The multi-layer Mamba/ResidualBlock wrappers are not needed.
#
# Original: https://github.com/alxndrTL/mamba.py/blob/main/mambapy/mamba.py

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from imfuse_infer.model._vendor_mambapy.pscan import pscan


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int = 1
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N
    expand_factor: int = 2  # E
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False

    pscan: bool = True  # use parallel scan mode or sequential mode

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(
                self.config.dt_rank, config.rms_norm_eps
            )
            self.B_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps
            )
            self.C_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps
            )
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  # depthwise convolution over time
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x, z)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z):
        # x : (B, L, ED)
        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = (
            self.dt_proj.weight @ delta.transpose(1, 2)
        )  # (ED, dt_rank) @ (B, L, dt_rank)^T -> (B, ED, L)

        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + self.dt_proj.bias)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(
            x.size(0),
            self.config.d_inner,
            self.config.d_state,
            device=deltaA.device,
        )  # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED)

        y = y + D * x

        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output
