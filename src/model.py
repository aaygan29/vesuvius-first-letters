"""3D->2D U-Net for ink detection.

Input:  (B, 1, Z, H, W) surface-volume tile (Z = layers around the papyrus surface)
Output: (B, 1, H, W) per-pixel ink logits

A single 3D conv mixes information across the Z (depth) layers, then we collapse
depth with a max over Z and run a standard 2D U-Net. This mirrors the canonical
Vesuvius ink-detection notebook, with a `base` width knob so it fits in the
limited memory of a laptop GPU.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, base: int = 32, in_z_channels: int = 1):
        super().__init__()
        b = base
        self.conv_z = nn.Conv3d(in_z_channels, b, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.enc1 = self._double_conv(b, b)
        self.enc2 = self._double_conv(b, b * 2)
        self.enc3 = self._double_conv(b * 2, b * 4)
        self.enc4 = self._double_conv(b * 4, b * 8)
        self.enc5 = self._double_conv(b * 8, b * 16)

        self.up1 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec1 = self._double_conv(b * 16, b * 8)
        self.up2 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec2 = self._double_conv(b * 8, b * 4)
        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec3 = self._double_conv(b * 4, b * 2)
        self.up4 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec4 = self._double_conv(b * 2, b)

        self.out_conv = nn.Conv2d(b, 1, kernel_size=1)

    @staticmethod
    def _double_conv(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_z(x)          # (B, base, Z, H, W)
        x, _ = torch.max(x, dim=2)  # collapse depth -> (B, base, H, W)

        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2))
        x5 = self.enc5(F.max_pool2d(x4, 2))

        x = self.dec1(torch.cat([self.up1(x5), x4], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec4(torch.cat([self.up4(x), x1], dim=1))
        return self.out_conv(x)


def initialize_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
