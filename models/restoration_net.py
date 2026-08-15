import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """Split channels in half, multiply them together -- non-linear gating."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class HINConvBlock(nn.Module):
    """Half Instance Normalization to preserve texture on unnormalized channel half."""
    def __init__(self, channels):
        super().__init__()
        assert channels % 2 == 0
        self.norm = nn.InstanceNorm2d(channels // 2, affine=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.norm(x1)
        return torch.cat([x1, x2], dim=1)


class DilatedGateBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 3, padding=dilation, dilation=dilation)
        self.hin = HINConvBlock(channels * 2)
        self.gate = SimpleGate()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.hin(out)
        out = self.gate(out)
        out = self.conv2(out)
        return x + out


class ResidualDenseGroup(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 5)):
        super().__init__()
        self.blocks = nn.ModuleList([DilatedGateBlock(channels, d) for d in dilations])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class NonLocalBlock(nn.Module):
    """Bounded adaptive pooled attention to prevent OOM errors at high resolutions."""
    def __init__(self, channels, pool_size=16):
        super().__init__()
        inter = max(channels // 2, 1)
        self.pool_size = pool_size
        self.theta = nn.Conv2d(channels, inter, 1)
        self.phi = nn.Conv2d(channels, inter, 1)
        self.g = nn.Conv2d(channels, inter, 1)
        self.project = nn.Conv2d(inter, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        ph, pw = min(self.pool_size, h), min(self.pool_size, w)
        x_small = F.adaptive_avg_pool2d(x, (ph, pw))

        theta = self.theta(x_small).view(b, -1, ph * pw).permute(0, 2, 1)
        phi = self.phi(x_small).view(b, -1, ph * pw)
        g = self.g(x_small).view(b, -1, ph * pw).permute(0, 2, 1)

        attn = torch.softmax(torch.bmm(theta, phi), dim=-1)
        y_small = torch.bmm(attn, g).permute(0, 2, 1).view(b, -1, ph, pw)
        y = F.interpolate(y_small, size=(h, w), mode="bilinear", align_corners=False)
        return x + self.project(y)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale ** 2), 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class RestorationNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_groups=4, scale_factor=2):
        super().__init__()
        self.scale = scale_factor

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.downsample = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)

        self.groups = nn.ModuleList([ResidualDenseGroup(base_channels) for _ in range(num_groups)])
        self.non_local = NonLocalBlock(base_channels)

        self.upsample_back = PixelShuffleUpsample(base_channels, scale=2)
        self.skip_merge = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)

        self.sr_upsample = PixelShuffleUpsample(base_channels, scale=scale_factor)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        baseline = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        stem_feat = self.stem(x)
        feat = self.downsample(stem_feat)
        for group in self.groups:
            feat = group(feat)
        feat = self.non_local(feat)

        feat = self.upsample_back(feat)
        if feat.shape[-2:] != stem_feat.shape[-2:]:
            feat = F.interpolate(feat, size=stem_feat.shape[-2:], mode="bilinear", align_corners=False)
        merged = self.skip_merge(torch.cat([stem_feat, feat], dim=1))

        feat = self.sr_upsample(merged)
        detail = self.out_conv(feat)
        return baseline + detail