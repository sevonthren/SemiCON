import numpy as np
import os

# Update these paths to match where your .npy files are located
lr_folder = r"C:\Users\kgowt\Downloads\SemiCON\data\train\NoisyLR"
gt_folder = r"C:\Users\kgowt\Downloads\SemiCON\data\train\GT"

# Pick the first file from each folder
lr_files = [f for f in os.listdir(lr_folder) if f.endswith('.npy') and not f.startswith('._')]
gt_files = [f for f in os.listdir(gt_folder) if f.endswith('.npy') and not f.startswith('._')]

sample_lr_path = os.path.join(lr_folder, lr_files[0])
sample_gt_path = os.path.join(gt_folder, gt_files[0])

# Load numpy arrays
sample_lr = np.load(sample_lr_path)
sample_gt = np.load(sample_gt_path)

print("--- Data Check Results ---")
print(f"LR File: {lr_files[0]}")
print(f"LR Shape: {sample_lr.shape}, Min: {sample_lr.min():.4f}, Max: {sample_lr.max():.4f}, Dtype: {sample_lr.dtype}")
print("-" * 30)
print(f"GT File: {gt_files[0]}")
print(f"GT Shape: {sample_gt.shape}, Min: {sample_gt.min():.4f}, Max: {sample_gt.max():.4f}, Dtype: {sample_gt.dtype}")
