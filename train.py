import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from models.restoration_net import RestorationNet
from dataset import get_train_val_loaders
from utils.metrics import calculate_psnr, calculate_ssim
from utils.losses import CompositeLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(model, val_loader, mean, std):
    model.eval()
    psnr_vals, ssim_vals = [], []
    with torch.no_grad():
        for lr, gt in val_loader:
            lr, gt = lr.to(device), gt.to(device)
            with autocast():
                out = model(lr)
            out = torch.clamp(out * std + mean, 0, 1)
            gt = torch.clamp(gt * std + mean, 0, 1)
            for i in range(out.shape[0]):
                psnr_vals.append(calculate_psnr(out[i:i + 1], gt[i:i + 1]).item())
                ssim_vals.append(calculate_ssim(out[i:i + 1], gt[i:i + 1]).item())
    return np.mean(psnr_vals), np.mean(ssim_vals)


def train_model(data_root, epochs=50, batch_size=16, lr=1e-4, save_path="best_model.pth"):
    train_loader, val_loader, mean, std = get_train_val_loaders(
        data_root, batch_size, patch_size_lr=64, scale_factor=2, val_ratio=0.1
    )

    model = RestorationNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    criterion = CompositeLoss(charbonnier_weight=1.0, ssim_weight=0.3, l1_weight=0.1).to(device)

    best_psnr = 0.0
    history = {"train_loss": [], "val_loss": [], "psnr": [], "ssim": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for lr_img, gt in train_loader:
            lr_img, gt = lr_img.to(device), gt.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(lr_img)
                loss = criterion(out, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_img, gt in val_loader:
                lr_img, gt = lr_img.to(device), gt.to(device)
                with autocast():
                    out = model(lr_img)
                    val_loss += criterion(out, gt).item()
        avg_val_loss = val_loss / len(val_loader)

        psnr, ssim = calculate_metrics(model, val_loader, mean, std)
        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["psnr"].append(psnr)
        history["ssim"].append(ssim)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

        if psnr > best_psnr:
            best_psnr = psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "psnr": best_psnr,
                "ssim": ssim,
                "mean": mean,
                "std": std,
            }, save_path)
            print(f"  Saved new best model -> {save_path} (PSNR: {best_psnr:.2f} dB)")

    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")
    return model, history


if __name__ == "__main__":
    train_model(data_root="./data", epochs=50)