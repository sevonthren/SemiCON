import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Import the model (we will swap DummyRestorationModel for Person B's model later)
from models.dummy_model import DummyRestorationModel

def percentile_normalize(img_np, p_min=1, p_max=99):
    """Percentile-based normalization for NoisyLR numpy arrays."""
    v_min, v_max = np.percentile(img_np, (p_min, p_max))
    if v_max == v_min:
        return np.zeros_like(img_np, dtype=np.float32)
    img_norm = (img_np - v_min) / (v_max - v_min)
    img_norm = np.clip(img_norm, 0.0, 1.0)
    return img_norm.astype(np.float32)

class TestDataset(Dataset):
    """Dataset class specifically for unlabelled inference data in evaluate.py."""
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.filenames = sorted([
            f for f in os.listdir(input_dir) 
            if f.endswith('.npy') and not f.startswith('._')
        ])
        assert len(self.filenames) > 0, f"No valid .npy files found in {input_dir}"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        fpath = os.path.join(self.input_dir, fname)
        
        lr_img = np.load(fpath).astype(np.float32)
        lr_img = percentile_normalize(lr_img)
        
        # Shape: (128, 128) -> (1, 128, 128)
        lr_tensor = torch.from_numpy(lr_img).unsqueeze(0)
        return lr_tensor, fname

def run_evaluation(input_dir, output_dir, model_path=None, batch_size=8, device="cuda"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select hardware device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = DummyRestorationModel(scale_factor=2).to(device)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model checkpoint from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No checkpoint loaded. Running with un-trained model weights.")
        
    model.eval()

    # Load Test Data
    test_dataset = TestDataset(input_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Processing {len(test_dataset)} images from '{input_dir}'...")

    with torch.no_grad():
        for lr_tensors, fnames in test_loader:
            lr_tensors = lr_tensors.to(device)
            
            # Predict clean HR image
            preds = model(lr_tensors)  # Output shape: (B, 1, 256, 256)
            preds = torch.clamp(preds, 0.0, 1.0)
            
            # Convert back to NumPy and save
            preds_np = preds.squeeze(1).cpu().numpy()  # Shape: (B, 256, 256)
            
            for i in range(len(fnames)):
                out_path = os.path.join(output_dir, fnames[i])
                np.save(out_path, preds_np[i].astype(np.float32))

    print(f"Successfully saved all restored images to '{output_dir}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Evaluation Script for Image Restoration")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory containing NoisyLR .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory to save restored .npy files")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (.pth file)")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    
    args = parser.parse_args()
    
    run_evaluation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        batch_size=args.batch_size
    )