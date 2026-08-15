import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import _gaussian_window_2d


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred, target):
        kernel = _gaussian_window_2d(self.window_size, self.sigma, pred.device, pred.dtype)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, kernel, padding=pad)
        mu2 = F.conv2d(target, kernel, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred ** 2, kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, kernel, padding=pad) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=pad) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class CompositeLoss(nn.Module):
    def __init__(self, charbonnier_weight=1.0, ssim_weight=0.3, l1_weight=0.1):
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0
        if self.charbonnier_weight > 0:
            loss = loss + self.charbonnier_weight * self.charbonnier(pred, target)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * self.ssim(pred, target)
        if self.l1_weight > 0:
            loss = loss + self.l1_weight * self.l1(pred, target)
        return loss