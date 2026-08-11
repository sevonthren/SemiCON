import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyRestorationModel(nn.Module):
    """
    A lightweight baseline model that uses bicubic upsampling + a small Conv layer.
    Used for testing the pipeline end-to-end.
    """
    def __init__(self, scale_factor=2):
        super(DummyRestorationModel, self).__init__()
        self.scale_factor = scale_factor
        # A tiny conv layer to simulate model weights
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # 1. Bicubic upsample by 2x: (B, 1, H, W) -> (B, 1, 2*H, 2*W)
        x_upsampled = F.interpolate(
            x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False
        )
        # 2. Pass through conv and clamp to valid range [0, 1]
        out = self.conv(x_upsampled)
        return torch.clamp(out, 0.0, 1.0)