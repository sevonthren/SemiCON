# KLA Semiconductor Image Restoration & Denoising Engine (SemiCON)

A high-performance deep learning pipeline designed to execute state-of-the-art speckle denoising and 2× super-resolution for semiconductor manufacturing inspection. Built for the Semicon India Hackathon 2026 (KLA Track PS01) , this project deploys a custom HINet-based `RestorationNet` augmented with half-resolution processing and non-local attention to restore heavily degraded electron microscopy scans with extreme fidelity.

---

##  Repository Structure

The project is organised into modular, self-contained pipeline directories and execution scripts:

```
SemiCON/
├── models/
│   ├── restoration_net.py   # Main RestorationNet architecture (HINet-based)
│   └── dummy_model.py       # Simple baseline for pipeline testing
├── utils/
│   ├── metrics.py           # PSNR, SSIM, LPIPS implementations
│   └── losses.py            # Composite loss (Charbonnier + SSIM + L1)
├── dataset.py                # Z-score normalised dataset & loaders
├── train.py                  # Training loop with mixed precision & early stopping
├── evaluate.py                # Standalone inference script with argparse
├── check_data.py              # Quick data shape/range sanity check
├── requirements.txt           # Python dependencies
├── dataset_stats.json         # Pre-computed mean/std (optional)
├── best_model.pth             # Final trained model weights (~20.1 MB)
├── submission_outputs/        # Restored test outputs (generated at inference)
└── README.md                  # This file
```

---

##  Dataset

**The raw dataset is NOT included in this repository** due to file size constraints.
You must obtain the KLA semiconductor dataset from the official i4C hackathon portal.

Once downloaded, place the `.npy` files in the following folder topology:

```
SemiCON/
└── data/
    ├── train/
    │   ├── NoisyLR/   # 128×128 degraded input arrays
    │   └── GT/        # 256×256 clean target arrays
    └── val/
        ├── NoisyLR/   # (optional) validation input
        └── GT/        # (optional) validation ground truth
```

> If your dataset uses a different nesting (e.g., `train/train/NoisyLR`), adjust the `--data_root` argument accordingly when running `train.py`.

---

##  Getting Started (How to Run)

Follow these steps to set up, train, and evaluate the restoration pipeline.

### 1. Clone the Repository

```bash
git clone https://github.com/sevonthren/SemiCON.git
cd SemiCON
```

### 2. Environment Setup

Create and activate a virtual environment:

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

##  Pipeline Execution Steps

Execute the training, inference, and benchmarking pipeline sequentially:

### Step A: Reproduce the Training Pipeline (Optional)

Train the model from scratch using the provided architecture and your dataset:

```bash
python train.py --data_root ./data --epochs 50 --batch_size 16 --lr 1e-4
```

**Recommended hyperparameters for best performance:**

- `--batch_size 12`
- `--lr 3e-5`
- `--patch_size 96`

The best checkpoint will be saved as `best_model.pth` in the current directory.

### Step B: Run Automated Inference & Generate Restored Outputs

Run the benchmarking script to process noisy test inputs and save the cleaned `.npy` arrays:

```bash
python evaluate.py --test_dir ./data/val/NoisyLR --output_dir ./submission_outputs --model_path best_model.pth --batch_size 16
```

> The `--output_dir` will be created if it doesn't exist, and all restored arrays will be written there.

---

##  Model Architecture: RestorationNet

Our custom CNN-based network is inspired by **HINet** and **NAFNet**, designed for lightweight and fast inference while preserving fine semiconductor details.

| Component | Purpose |
| --- | --- |
| **Half-Instance Normalization (HIN)** | Stabilises training while preserving texture |
| **SimpleGate** | Replaces heavy activations with channel-wise multiplication |
| **Dilated Convolutions** (coprime rates 1, 2, 5) | Expands receptive field without gridding artifacts |
| **Non-Local Block** (pooled) | Mimics classical non-local means denoising |
| **PixelShuffle Upsampling** | Clean 2× super-resolution without checkerboard artifacts |
| **Global Residual** | Learns only the missing details on top of a bicubic baseline |

**Parameter count:** ~1.75M — ultra-lightweight and edge-deployable.

---

##  Loss & Metrics

### Loss Function

Composite loss combining three complementary objectives:

```
L = L_Charbonnier + 0.3 * L_SSIM + 0.1 * L_L1
```

- **Charbonnier Loss** – robust L1 variant for handling outliers
- **SSIM Loss** – structural similarity for edge preservation
- **L1 Loss** – pixel-wise absolute difference

### Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) – higher is better
- **SSIM** (Structural Similarity) – closer to 1 is better
- **LPIPS** (Learned Perceptual Similarity) – lower is better (computed with AlexNet)

---



##  Testing the Pipeline

You can quickly verify the pipeline using the dummy model:

```bash
python train.py --model_class DummyRestorationModel --epochs 1
```

Or inspect your data files with:

```bash
python check_data.py
```

---

##  Dependencies

See `requirements.txt` for the full list. Key packages:

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.20.0
opencv-python>=4.7.0
matplotlib>=3.7.0
pillow>=9.5.0
tqdm>=4.65.0
lpips
```

---

##  Important Notes

- All images are grayscale (1-channel).
- Input size: **128×128**, output size: **256×256** (2× upscaling).
- **Z-score normalisation** is applied (no clipping) to preserve speckle noise peaks – critical for meeting KLA's challenge requirements.
- The `lpips` package downloads pretrained AlexNet weights at first use – internet access is required.
- The final checkpoint `best_model.pth` is only **~20.1 MB**, making it suitable for local deployment.

---

##  Acknowledgements

This work builds on outstanding research from:

- **NAFNet** (Megvii)
- **HINet** (NTIRE 2021 winner)
- **Restormer** (Transformer for restoration)



---

