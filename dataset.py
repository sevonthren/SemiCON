import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np

def percentile_normalize(img_np, p_min=1, p_max=99):
    """
    Percentile-based normalization for NoisyLR numpy arrays.
    Clamps extreme outliers and scales values to [0, 1].
    """
    v_min, v_max = np.percentile(img_np, (p_min, p_max))
    if v_max == v_min:
        return np.zeros_like(img_np, dtype=np.float32)
    img_norm = (img_np - v_min) / (v_max - v_min)
    img_norm = np.clip(img_norm, 0.0, 1.0)
    return img_norm.astype(np.float32)

class RestorationDataset(Dataset):
    def __init__(self, lr_dir, gt_dir, patch_size_lr=64, scale_factor=2, is_train=True):
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.patch_size_lr = patch_size_lr
        self.scale_factor = scale_factor
        self.is_train = is_train

        # Load file lists while explicitly filtering out hidden/macOS metadata files (._*)
        self.lr_filenames = sorted([
            os.path.join(lr_dir, f) for f in os.listdir(lr_dir) 
            if f.endswith('.npy') and not f.startswith('._')
        ])
        self.gt_filenames = sorted([
            os.path.join(gt_dir, f) for f in os.listdir(gt_dir) 
            if f.endswith('.npy') and not f.startswith('._')
        ])
        
        assert len(self.lr_filenames) > 0, f"No valid .npy files found in {lr_dir}"
        assert len(self.lr_filenames) == len(self.gt_filenames), \
            f"Mismatch: {len(self.lr_filenames)} LR files vs {len(self.gt_filenames)} GT files."

    def __len__(self):
        return len(self.lr_filenames)

    def __getitem__(self, idx):
        lr_path = self.lr_filenames[idx]
        gt_path = self.gt_filenames[idx]

        # Load numpy arrays
        lr_img = np.load(lr_path).astype(np.float32)
        gt_img = np.load(gt_path).astype(np.float32)

        # Handle dimension ordering if array is (H, W, C) vs (C, H, W)
        if lr_img.ndim == 2:  # Grayscale (H, W) -> (1, H, W)
            lr_img = np.expand_dims(lr_img, axis=0)
            gt_img = np.expand_dims(gt_img, axis=0)
        elif lr_img.ndim == 3 and lr_img.shape[-1] in [1, 3]:  # (H, W, C) -> transpose to (C, H, W)
            lr_img = np.transpose(lr_img, (2, 0, 1))
            gt_img = np.transpose(gt_img, (2, 0, 1))

        # Percentile-based scaling on NoisyLR
        lr_img = percentile_normalize(lr_img)
        gt_img = np.clip(gt_img, 0.0, 1.0)  # GT strictly in [0, 1]

        lr_tensor = torch.from_numpy(lr_img)
        gt_tensor = torch.from_numpy(gt_img)

        # Training-time cropping and augmentations
        if self.is_train:
            _, h_lr, w_lr = lr_tensor.shape
            p_lr = self.patch_size_lr
            p_gt = p_lr * self.scale_factor

            max_h = h_lr - p_lr
            max_w = w_lr - p_lr
            h_start = torch.randint(0, max_h + 1, (1,)).item() if max_h > 0 else 0
            w_start = torch.randint(0, max_w + 1, (1,)).item() if max_w > 0 else 0

            # Crop paired patches
            lr_tensor = lr_tensor[:, h_start:h_start + p_lr, w_start:w_start + p_lr]
            
            h_start_gt, w_start_gt = h_start * self.scale_factor, w_start * self.scale_factor
            gt_tensor = gt_tensor[:, h_start_gt:h_start_gt + p_gt, w_start_gt:w_start_gt + p_gt]

            # Augmentations (Horizontal flip & rotations)
            if torch.rand(1).item() > 0.5:
                lr_tensor = TF.hflip(lr_tensor)
                gt_tensor = TF.hflip(gt_tensor)

            rot_k = torch.randint(0, 4, (1,)).item()
            if rot_k > 0:
                lr_tensor = torch.rot90(lr_tensor, rot_k, [1, 2])
                gt_tensor = torch.rot90(gt_tensor, rot_k, [1, 2])

        return lr_tensor, gt_tensor