#!/usr/bin/env python3
import os
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import (
    get_dataloaders,
    get_default_transform,
    compute_class_weights,
    weighted_mse,
    mse_to_rmse,
    mse_to_pct
)


class CNN_GRU_FUSION(nn.Module):
    def __init__(self):
        super().__init__()


        # CNN → images

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            self.cnn_out_size = self.cnn(dummy).shape[1]

        # One GRU with 2 features

        self.gru_dyn = nn.GRU(
            input_size=2,   # speed + deviation
            hidden_size=32,
            batch_first=True
        )

        # Final fusion
 
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_size + 32, 100),
            nn.ReLU(),
            nn.Linear(100, 2)   # steer, throttle
        )

    def forward(self, images, speeds, deviations):

        B, T, C, H, W = images.shape


        # CNN per frame

        x = images.view(B * T, C, H, W)
        feats_img = self.cnn(x)
        feats_img = feats_img.view(B, T, -1)

        # last frame
        feats_img = feats_img[:, -1, :]   # (B, cnn_out)


        # GRU with 2 features speed and deviation
   
        dyn = torch.cat([speeds, deviations], dim=2)   # (B, T, 2)
        out_dyn, _ = self.gru_dyn(dyn)
        feats_dyn = out_dyn[:, -1, :]  # (B, 32)


        # Fusion
  
        fused = torch.cat([feats_img, feats_dyn], dim=1)

        return self.fc(fused)

def main():

    num_epochs = 15
    batch_size = 32
    lr = 3e-4
    delta = 1e-4
    seq_len = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=exp_dir)

    transform = get_default_transform()

    train_dataset, val_dataset, test_dataset, \
    train_loader, val_loader, test_loader = get_dataloaders(
        "train.pt",
        "val.pt",
        "test.pt",
        batch_size=batch_size,
        transform=transform
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    print("\n===== DATA DEBUG =====")
    sample = train_dataset[0]
    print("Seq imgs shape:", sample[0].shape)      # (T, C, H, W)
    print("Seq speeds shape:", sample[1].shape)   # (T, 1)
    print("Label:", sample[2])                    # (2,)
    print("Estado:", sample[3])
    print("======================\n")

    w_global = compute_class_weights(train_dataset.estados, device)
    print("Weights:", w_global)

    model = CNN_GRU_FUSION().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()

    best_val = float("inf")
    best_model = deepcopy(model)

    print("\n*********** Training Started ************")

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0

        for i, (seq_imgs, seq_speeds, seq_deviations, labels, estados) in enumerate(train_loader):

            if epoch == 0 and i == 0:
                print("\n===== BATCH DEBUG =====")
                print("seq_imgs shape:", seq_imgs.shape)       # (B, T, C, H, W)
                print("seq_speeds shape:", seq_speeds.shape)   # (B, T, 1)
                print("labels shape:", labels.shape)           # (B, 2)
                print("estados shape:", estados.shape)         # (B,)
                print("=======================\n")

            seq_imgs = seq_imgs.to(device).float()
            seq_speeds = seq_speeds.to(device).float()
            seq_deviations = seq_deviations.to(device).float()
            seq_deviations = seq_deviations / 100.0
            labels = labels.to(device).float()
            estados = estados.to(device).long()

            out = model(seq_imgs, seq_speeds, seq_deviations)

            if epoch == 0 and i == 0:
                print("\n===== MODEL OUTPUT DEBUG =====")
                print("out shape:", out.shape)         # (B, 2)
                print("pred[0]:", out[0].detach().cpu())
                print("gt[0]:", labels[0].detach().cpu())
                print("==============================\n")

            loss = weighted_mse(out, labels, estados, w_global)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.6f}")

        train_loss /= len(train_loader)

        model.eval()
        val_mse = 0.0

        with torch.no_grad():
            for seq_imgs, seq_speeds, seq_deviations, labels, _ in val_loader:
                seq_imgs = seq_imgs.to(device).float()
                seq_speeds = seq_speeds.to(device).float()
                seq_deviations = seq_deviations.to(device).float()
                seq_deviations = seq_deviations / 100.0
                labels = labels.to(device).float()

                out = model(seq_imgs, seq_speeds, seq_deviations)
                val_mse += criterion_mse(out, labels).item()

        val_mse /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_mse:.6f}")

        if val_mse < best_val - delta:
            best_val = val_mse
            best_model = deepcopy(model)
            print("Model Improved!")
        else:
            print("Not Improved")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_mse, epoch)
        writer.add_scalar("RMSE/Val", mse_to_rmse(val_mse), epoch)
        writer.add_scalar("PctError/Val", mse_to_pct(val_mse), epoch)

    model = best_model
    model.eval()

    test_mse = 0.0

    with torch.no_grad():
        for seq_imgs, seq_speeds, seq_deviations, labels, _ in test_loader:
            seq_imgs = seq_imgs.to(device).float()
            seq_speeds = seq_speeds.to(device).float()
            labels = labels.to(device).float()
            seq_deviations = seq_deviations.to(device).float()
            seq_deviations = seq_deviations / 100.0

            out = model(seq_imgs, seq_speeds, seq_deviations)
            test_mse += criterion_mse(out, labels).item()

    test_mse /= len(test_loader)

    writer.add_scalar("Loss/Test", test_mse, 0)
    writer.close()

    final_pth = os.path.join(exp_dir, "gru_model.pth")
    best_path = os.path.join(exp_dir, "best_model.pth")
    net_file_name = os.path.join(
        exp_dir,
        "gru_model_gpu.onnx" if torch.cuda.is_available() else "gru_model.onnx"
    )

    torch.save(model.state_dict(), final_pth)
    torch.save(best_model.state_dict(), best_path)

    dummy_imgs = torch.randn(1, seq_len, 3, 66, 200).to(device)
    dummy_speeds = torch.randn(1, seq_len, 1).to(device)
    dummy_devs = torch.randn(1, seq_len, 1).to(device)

    torch.onnx.export(
        model,
        (dummy_imgs, dummy_speeds, dummy_devs),
        net_file_name,
        verbose=False,
        export_params=True,
        opset_version=9,
        input_names=["images", "speeds"],
        output_names=["output"],
    )

    print("\n========== TRAINING FINISHED ==========")
    print(f"[OK] Best model : {best_path}")
    print(f"[OK] Final model: {final_pth}")
    print(f"[OK] ONNX       : {net_file_name}")
    print("=======================================\n")


if __name__ == "__main__":
    main()