"""
Self-contained nnU-Net + SeedSegmentor architecture for inference.

Copied from seedModel project to avoid external dependency.
Only requires torch and torch.nn.
"""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual convolutional block: Conv3x3 -> IN -> LeakyReLU -> Conv3x3 -> IN + skip."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class nnUNetEncoder(nn.Module):
    """nnU-Net encoder with strided conv downsampling."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32, num_stages: int = 5):
        super().__init__()
        self.num_stages = num_stages
        channels = [base_channels * (2 ** i) for i in range(num_stages)]

        self.stages = nn.ModuleList()
        self.stages.append(ConvBlock(in_channels, channels[0]))

        self.downsamples = nn.ModuleList()
        for i in range(1, num_stages):
            self.downsamples.append(
                nn.Conv2d(channels[i - 1], channels[i - 1], kernel_size=2, stride=2, bias=False)
            )
            self.stages.append(ConvBlock(channels[i - 1], channels[i]))

        self.channels = channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        out = self.stages[0](x)
        features.append(out)

        for i in range(1, self.num_stages):
            out = self.downsamples[i - 1](out)
            out = self.stages[i](out)
            features.append(out)

        return features


class nnUNetDecoder(nn.Module):
    """nnU-Net decoder with transposed conv upsampling and skip connections."""

    def __init__(self, encoder_channels: List[int], deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        num_decoder_stages = len(encoder_channels) - 1

        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()

        for i in range(num_decoder_stages):
            low_ch = encoder_channels[-(i + 1)]
            skip_ch = encoder_channels[-(i + 2)]

            self.upsamples.append(
                nn.ConvTranspose2d(low_ch, skip_ch, kernel_size=2, stride=2, bias=False)
            )
            self.stages.append(ConvBlock(skip_ch * 2, skip_ch))

        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for i in range(num_decoder_stages - 1):
                out_ch = encoder_channels[-(i + 2)]
                self.ds_heads.append(nn.Conv2d(out_ch, 1, kernel_size=1))

    def forward(
        self, features: List[torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = features[-1]
        ds_outputs = []
        num_stages = len(self.stages)

        for i in range(num_stages):
            skip = features[-(i + 2)]
            x = self.upsamples[i](x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.stages[i](x)

            if self.deep_supervision and i < num_stages - 1:
                ds_outputs.append(self.ds_heads[i](x))

        if self.deep_supervision:
            ds_outputs.reverse()
            return x, ds_outputs

        return x, None


class nnUNet2D(nn.Module):
    """nnU-Net 2D segmentation model."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        num_stages: int = 5,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.encoder = nnUNetEncoder(in_channels, base_channels, num_stages)
        self.decoder = nnUNetDecoder(self.encoder.channels, deep_supervision)
        self.seg_head = nn.Conv2d(self.encoder.channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        features = self.encoder(x)
        decoded, ds_outputs = self.decoder(features)
        logits = self.seg_head(decoded)

        if self.training and self.deep_supervision and ds_outputs is not None:
            return [logits] + ds_outputs

        return logits


class SeedSegmentor(nn.Module):
    """Seed point-based vessel segmentor (nnU-Net only, for inference)."""

    def __init__(
        self,
        in_channels: int = 2,
        nnunet_cfg: dict | None = None,
    ):
        super().__init__()
        self.model_type = "nnunet"
        nnunet_cfg = nnunet_cfg or {}
        self.net = nnUNet2D(
            in_channels=in_channels,
            base_channels=nnunet_cfg.get("base_channels", 32),
            num_stages=nnunet_cfg.get("num_stages", 5),
            deep_supervision=False,
        )

    def forward(
        self,
        image: torch.Tensor,
        seed_map: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([image, seed_map], dim=1)
        return self.net(x)
