import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class RestorationDataset(Dataset):
    def __init__(self, lr_dir, gt_dir, file_list=None, patch_size_lr=64,
                 scale_factor=2, is_train=True, stats_path="dataset_stats.json"):
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.patch_size_lr = patch_size_lr
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.stats_path = stats_path

        if file_list is not None:
            self.filenames = file_list
        else:
            self.filenames = sorted([
                f for f in os.listdir(lr_dir)
                if f.endswith(".npy") and not f.startswith("._")
            ])
        assert len(self.filenames) > 0, f"No .npy files found in {lr_dir}"

        if is_train:
            self.mean, self.std = self.compute_stats()
            with open(self.stats_path, "w") as f:
                json.dump({"mean": float(self.mean), "std": float(self.std)}, f)
            print(f"Dataset stats -- mean: {self.mean:.4f}, std: {self.std:.4f}")
        else:
            try:
                with open(self.stats_path, "r") as f:
                    stats = json.load(f)
                    self.mean, self.std = stats["mean"], stats["std"]
            except Exception:
                self.mean, self.std = 0.0, 1.0

    def compute_stats(self):
        all_pixels = []
        max_files = min(len(self.filenames), 100)
        for fname in self.filenames[:max_files]:
            lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
            gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
            all_pixels.append(lr.flatten())
            all_pixels.append(gt.flatten())
        all_pixels = np.concatenate(all_pixels)
        return float(np.mean(all_pixels)), float(np.std(all_pixels))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr_img = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
        gt_img = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        lr_img = (lr_img - self.mean) / self.std
        gt_img = (gt_img - self.mean) / self.std

        lr_tensor = torch.from_numpy(lr_img).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0)

        if self.is_train:
            _, h_lr, w_lr = lr_tensor.shape
            p_lr = self.patch_size_lr
            p_gt = p_lr * self.scale_factor

            if h_lr > p_lr and w_lr > p_lr:
                h_start = torch.randint(0, h_lr - p_lr + 1, (1,)).item()
                w_start = torch.randint(0, w_lr - p_lr + 1, (1,)).item()
            else:
                h_start, w_start = 0, 0

            lr_tensor = lr_tensor[:, h_start:h_start + p_lr, w_start:w_start + p_lr]
            h_start_gt, w_start_gt = h_start * self.scale_factor, w_start * self.scale_factor
            gt_tensor = gt_tensor[:, h_start_gt:h_start_gt + p_gt, w_start_gt:w_start_gt + p_gt]

            if torch.rand(1).item() > 0.5:
                lr_tensor, gt_tensor = TF.hflip(lr_tensor), TF.hflip(gt_tensor)
            if torch.rand(1).item() > 0.5:
                lr_tensor, gt_tensor = TF.vflip(lr_tensor), TF.vflip(gt_tensor)
            rot_k = torch.randint(0, 4, (1,)).item()
            if rot_k > 0:
                lr_tensor = torch.rot90(lr_tensor, rot_k, [1, 2])
                gt_tensor = torch.rot90(gt_tensor, rot_k, [1, 2])

        return lr_tensor, gt_tensor


def get_train_val_loaders(data_root, batch_size=16, patch_size_lr=64,
                          scale_factor=2, val_ratio=0.1):
    train_lr_dir = os.path.join(data_root, "train", "NoisyLR")
    train_gt_dir = os.path.join(data_root, "train", "GT")

    all_files = sorted([f for f in os.listdir(train_lr_dir)
                        if f.endswith(".npy") and not f.startswith("._")])
    random.shuffle(all_files)

    val_size = int(len(all_files) * val_ratio)
    train_files, val_files = all_files[val_size:], all_files[:val_size]

    train_dataset = RestorationDataset(train_lr_dir, train_gt_dir, train_files,
                                       patch_size_lr, scale_factor, is_train=True)
    val_dataset = RestorationDataset(train_lr_dir, train_gt_dir, val_files,
                                     patch_size_lr, scale_factor, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, train_dataset.mean, train_dataset.std