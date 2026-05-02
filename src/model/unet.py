import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """Blind residual denoising U-Net. Predicts noise and subtracts it from the input."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        bilinear: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._checkpointing = False

        c = base_channels
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, 2 * c)
        self.down2 = Down(2 * c, 4 * c)
        self.down3 = Down(4 * c, 8 * c)
        self.down4 = Down(8 * c, 16 * c // factor)
        self.up1 = Up(16 * c, 8 * c // factor, bilinear)
        self.up2 = Up(8 * c, 4 * c // factor, bilinear)
        self.up3 = Up(4 * c, 2 * c // factor, bilinear)
        self.up4 = Up(2 * c, c, bilinear)
        self.outc = OutConv(c, out_channels)

    def _run(self, module, *args):
        if self._checkpointing and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._run(self.inc, x)
        x2 = self._run(self.down1, x1)
        x3 = self._run(self.down2, x2)
        x4 = self._run(self.down3, x3)
        x5 = self._run(self.down4, x4)
        y = self._run(self.up1, x5, x4)
        y = self._run(self.up2, y, x3)
        y = self._run(self.up3, y, x2)
        y = self._run(self.up4, y, x1)
        return (x - self.outc(y)).clamp(0.0, 1.0)

    def use_checkpointing(self):
        self._checkpointing = True
