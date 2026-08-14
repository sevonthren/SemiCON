import os
import argparse
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import torch

# Optional: try importing lpips if installed
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def calculate_metrics(pred_dir, gt_dir, calc_lpips=False):
    """
    Computes average PSNR, SSIM, and (optional) LPIPS across all paired .npy files in pred_dir and gt_dir.
    """
    pred_files = sorted([
        f for f in os.listdir(pred_dir) 
        if f.endswith('.npy') and not f.startswith('._')
    ])
    gt_files = sorted([
        f for f in os.listdir(gt_dir) 
        if f.endswith('.npy') and not f.startswith('._')
    ])

    assert len(pred_files) == len(gt_files), \
        f"Mismatch: {len(pred_files)} prediction files vs {len(gt_files)} GT files."

    psnr_list = []
    ssim_list = []
    lpips_list = []

    # Initialize LPIPS model if requested
    lpips_fn = None
    if calc_lpips:
        if LPIPS_AVAILABLE:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # VGG backbone is standard for LPIPS evaluation
            lpips_fn = lpips.LPIPS(net='vgg').to(device)
            lpips_fn.eval()
        else:
            print("Warning: 'lpips' library not installed. Skipping LPIPS calculation.")

    print(f"Calculating metrics across {len(pred_files)} images...")

    for i in range(len(pred_files)):
        pred_path = os.path.join(pred_dir, pred_files[i])
        gt_path = os.path.join(gt_dir, gt_files[i])

        # Load numpy arrays
        pred_img = np.load(pred_path).astype(np.float32)
        gt_img = np.load(gt_path).astype(np.float32)

        # Clip values to [0, 1] for safe calculation
        pred_img = np.clip(pred_img, 0.0, 1.0)
        gt_img = np.clip(gt_img, 0.0, 1.0)

        # 1. PSNR Calculation (data_range=1.0 since images are normalized [0, 1])
        psnr_val = compute_psnr(gt_img, pred_img, data_range=1.0)
        psnr_list.append(psnr_val)

        # 2. SSIM Calculation
        ssim_val = compute_ssim(gt_img, pred_img, data_range=1.0)
        ssim_list.append(ssim_val)

        # 3. LPIPS Calculation (Optional)
        if lpips_fn is not None:
            # LPIPS expects 3-channel tensors in range [-1, 1]
            pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0)      # (1, 1, H, W)

            # Convert 1-channel grayscale to 3-channel RGB for VGG
            pred_tensor = pred_tensor.repeat(1, 3, 1, 1)
            gt_tensor = gt_tensor.repeat(1, 3, 1, 1)

            # Map range [0, 1] -> [-1, 1]
            pred_tensor = pred_tensor * 2.0 - 1.0
            gt_tensor = gt_tensor * 2.0 - 1.0

            with torch.no_grad():
                lpips_val = lpips_fn(pred_tensor.to(device), gt_tensor.to(device)).item()
                lpips_list.append(lpips_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list) if lpips_list else None

    # Print Summary Results
    print("\n" + "="*35)
    print("      EVALUATION RESULTS         ")
    print("="*35)
    print(f" Average PSNR:  {avg_psnr:.4f} dB  (Higher is better)")
    print(f" Average SSIM:  {avg_ssim:.4f}     (Higher is better)")
    if avg_lpips is not None:
        print(f" Average LPIPS: {avg_lpips:.4f}     (Lower is better)")
    print("="*35 + "\n")

    return {"psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, and LPIPS metrics for .npy predictions")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to directory containing predicted .npy files")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to directory containing Ground Truth .npy files")
    parser.add_argument("--calc_lpips", action="store_true", help="Include LPIPS calculation (requires GPU/torch)")

    args = parser.parse_args()

    calculate_metrics(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        calc_lpips=args.calc_lpips
    )