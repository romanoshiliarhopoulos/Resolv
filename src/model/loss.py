import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM, averaged over batch/channels. Inputs are expected in [0, 1]."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, channels: int = 3, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.data_range = data_range
        self.register_buffer("_kernel", torch.empty(0), persistent=False)

    def _get_kernel(self, x: torch.Tensor) -> torch.Tensor:
        if self._kernel.numel() == 0 or self._kernel.shape[0] != x.shape[1] \
                or self._kernel.device != x.device or self._kernel.dtype != x.dtype:
            self._kernel = _gaussian_kernel(
                self.window_size, self.sigma, x.shape[1], x.device, x.dtype
            )
        return self._kernel

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernel = self._get_kernel(pred)
        pad = self.window_size // 2
        groups = pred.shape[1]

        mu_p = F.conv2d(pred, kernel, padding=pad, groups=groups)
        mu_t = F.conv2d(target, kernel, padding=pad, groups=groups)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p2 = F.conv2d(pred * pred, kernel, padding=pad, groups=groups) - mu_p2
        sigma_t2 = F.conv2d(target * target, kernel, padding=pad, groups=groups) - mu_t2
        sigma_pt = F.conv2d(pred * target, kernel, padding=pad, groups=groups) - mu_pt

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
            (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
        )
        return 1.0 - ssim_map.mean()


class DenoiseLoss(nn.Module):
    """L1 + alpha * (1 - SSIM). Standard combo for image restoration."""

    def __init__(self, alpha: float = 0.15, channels: int = 3):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(channels=channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target) + self.alpha * self.ssim(pred, target)
