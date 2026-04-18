"""Basic 3D convolution layers — ported from MV-IM-Fuse/layers.py."""

import torch
import torch.nn as nn


def normalization(planes: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    elif norm == "gn":
        return nn.GroupNorm(4, planes)
    elif norm == "in":
        return nn.InstanceNorm3d(planes)
    else:
        raise ValueError(f"normalization type {norm} is not supported")


class general_conv3d_prenorm(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "zeros",
        norm: str = "in",
        is_training: bool = True,
        act_type: str = "lrelu",
        relufactor: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )
        self.norm = normalization(out_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class general_conv3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "zeros",
        norm: str = "in",
        is_training: bool = True,
        act_type: str = "lrelu",
        relufactor: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )
        self.norm = normalization(out_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class fusion_prenorm(nn.Module):
    def __init__(self, in_channel: int = 64, num_cls: int = 4):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d_prenorm(
                in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1
            ),
            general_conv3d_prenorm(
                in_channel, in_channel, k_size=3, padding=1, stride=1
            ),
            general_conv3d_prenorm(
                in_channel, in_channel, k_size=1, padding=0, stride=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_layer(x)
