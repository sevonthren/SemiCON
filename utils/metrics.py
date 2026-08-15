import torch
import torch.nn.functional as F


def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-8:
        return torch.tensor(60.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian_window_2d(window_size, sigma, device, dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size // 2)
    g = torch.exp(-coords ** 2 / (2 * (sigma ** 2)))
    g = g / g.sum()
    return torch.outer(g, g).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)


def calculate_ssim(pred, target, window_size=11, sigma=1.5):
    kernel = _gaussian_window_2d(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad)
    mu2 = F.conv2d(target, kernel, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()