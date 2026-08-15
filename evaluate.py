import os
import time
import numpy as np
import torch
from torch.cuda.amp import autocast
from models.restoration_net import RestorationNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model_path, test_dir, output_dir, batch_size=16):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = RestorationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint.get("mean", 0.0)
    std = checkpoint.get("std", 1.0)

    test_files = sorted([f for f in os.listdir(test_dir)
                          if f.endswith(".npy") and not f.startswith("._")])
    total_time = 0
    
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i + batch_size]
        batch = []
        for fname in batch_files:
            lr = np.load(os.path.join(test_dir, fname)).astype(np.float32)
            lr = (lr - mean) / std
            batch.append(torch.from_numpy(lr).unsqueeze(0).unsqueeze(0))
            
        batch_tensor = torch.cat(batch, dim=0).to(device)

        start = time.time()
        with torch.no_grad():
            with autocast():
                out = model(batch_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - start

        for j, out_img in enumerate(out):
            out_np = out_img.squeeze().cpu().numpy() * std + mean
            out_np = np.clip(out_np, 0.0, 1.0)
            np.save(os.path.join(output_dir, batch_files[j]), out_np.astype(np.float32))

    fps = len(test_files) / total_time if total_time > 0 else 0
    print(f"Inference complete. Evaluated {len(test_files)} images at {fps:.2f} FPS.")


if __name__ == "__main__":
    run_inference(
        model_path="best_model.pth",
        test_dir="./data/test/NoisyLR",
        output_dir="./submission_outputs"
    )