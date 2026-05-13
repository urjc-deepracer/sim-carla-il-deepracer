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
    mse_to_rmse,
    mse_to_pct
)

# INPUT: [32, 10, 3, 66, 200]
# OUTPUT: [32,3]


#CNN → concat features → 1 GRU → FC
class CNN_GRU_FUSION(nn.Module):
    def __init__(self):
        super().__init__()

   
        # Images CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Get CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            self.cnn_out_size = self.cnn(dummy).shape[1]


        # Unique GRU (all data at once)
        self.gru = nn.GRU(
            input_size=self.cnn_out_size + 2 + 2,  # img + (speed,dev) + (steer,throttle)
            hidden_size=64,
            batch_first=True
        )

        # Final fusion
        self.fc = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  # steer, throttle
        )

    def forward(self, images, speeds, deviations):
        # images, speeds, devs already size 10 and aligned
        B, T, C, H, W = images.shape
        
        # CNN para las imágenes
        x = images.view(B*T, C, H, W)
        feats_img = self.cnn(x).view(B, T, -1)

        # Concat: [Img_t, Spd_t, Dev_t, Ctrl_t-1]
        # .pt arrives already shifted
        telemetry = torch.cat([speeds, deviations], dim=2)
        fused_seq = torch.cat([feats_img, telemetry], dim=2)

        gru_out, _ = self.gru(fused_seq)
        return self.fc(gru_out[:, -1, :])


def compute_r2(preds, targets):

    ss_res = torch.sum((targets - preds) ** 2, dim=0)
    ss_tot = torch.sum((targets - torch.mean(targets, dim=0)) ** 2, dim=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return torch.mean(r2).item()

def main():

    USE_DEVIATION = True
    num_epochs = 40
    batch_size = 32
    lr = 1e-3
    delta = 1e-4
    seq_len = 10

    device = torch.device("cuda") 
    print(f"Confirmado: Usando {torch.cuda.get_device_name(0)}")

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

    print("\n----- DATA DEBUG -----")
    sample = train_dataset[0]
    print("Seq imgs shape:", sample[0].shape)      # (T, C, H, W)
    print("Seq speeds shape:", sample[1].shape)   # (T, 1)
    print("Seq controls shape:", sample[3].shape)
    print("Label:", sample[4])
    print("Estado:", sample[5])
    print("======================\n")


    if USE_DEVIATION:
        model = CNN_GRU_FUSION().to(device)
    else:
        model = CNN_GRU_FUSION_NO_DEV().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()

    best_val = float("inf")
    best_model = deepcopy(model)

    print("\n*********** Training Started ************")

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0

        for i, (seq_imgs, seq_speeds, seq_deviations, seq_controls, labels, estados) in enumerate(train_loader):
            if epoch == 0 and i == 0:
                print("\n===== BATCH DEBUG =====")
                print("seq_imgs shape:", seq_imgs.shape)       # (B, T, C, H, W)
                print("seq_speeds shape:", seq_speeds.shape)   # (B, T, 1)
                print("labels shape:", labels.shape)           # (B, 2)
                print("estados shape:", estados.shape)         # (B,)
                print("=======================\n")

            seq_imgs = seq_imgs.to(device).float()
            seq_speeds = seq_speeds.to(device).float()
            labels = labels.to(device).float()
            seq_controls = seq_controls.to(device).float()
            estados = estados.to(device).long()

            if USE_DEVIATION:
                seq_devs = seq_deviations.to(device).float() / 100.0
                out = model(seq_imgs, seq_speeds, seq_devs, seq_controls)
            else:
                out = model(seq_imgs, seq_speeds, seq_controls)

            if epoch == 0 and i == 0:
                print("\n===== MODEL OUTPUT DEBUG =====")
                print("out shape:", out.shape)         # (B, 2)
                print("pred[0]:", out[0].detach().cpu())
                print("gt[0]:", labels[0].detach().cpu())
                print("==============================\n")

            loss = criterion_mse(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.6f}")

        train_loss /= len(train_loader)

        model.eval()
        val_mse = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for seq_imgs, seq_speeds, seq_deviations, seq_controls, labels, _ in val_loader:
                seq_imgs = seq_imgs.to(device).float()
                seq_speeds = seq_speeds.to(device).float()
                seq_controls = seq_controls.to(device).float()
                labels = labels.to(device).float()

                if USE_DEVIATION:
                    seq_devs = seq_deviations.to(device).float() / 100.0
                    out = model(seq_imgs, seq_speeds, seq_devs, seq_controls)
                else:
                    out = model(seq_imgs, seq_speeds, seq_controls)
                
                val_mse += criterion_mse(out, labels).item()
                all_preds.append(out.cpu())
                all_labels.append(labels.cpu())

        val_mse /= len(val_loader)
        val_preds_all = torch.cat(all_preds, dim=0)
        val_labels_all = torch.cat(all_labels, dim=0)
        val_r2 = compute_r2(val_preds_all, val_labels_all)
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
        writer.add_scalar("MSE_Percentage/Val", val_mse * 100.0, epoch)
        writer.add_scalar("R2_Score/Val", val_r2, epoch)

    model = best_model
    model.eval()

    test_mse = 0.0
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for seq_imgs, seq_speeds, seq_deviations, seq_controls, labels, _ in test_loader:
            seq_imgs = seq_imgs.to(device).float()
            seq_speeds = seq_speeds.to(device).float()
            labels = labels.to(device).float()

            seq_controls = seq_controls.to(device).float()

            if USE_DEVIATION:
                seq_devs = seq_deviations.to(device).float() / 100.0
                out = model(seq_imgs, seq_speeds, seq_devs, seq_controls)
            else:
                out = model(seq_imgs, seq_speeds, seq_controls)

            mse = nn.MSELoss()(out, labels)
            test_mse += criterion_mse(out, labels).item()

            all_test_preds.append(out.cpu())
            all_test_labels.append(labels.cpu())

    test_mse /= len(test_loader)
    test_preds_all = torch.cat(all_test_preds, dim=0)
    test_labels_all = torch.cat(all_test_labels, dim=0)
    
    test_r2 = compute_r2(test_preds_all, test_labels_all)
    test_rmse = test_mse ** 0.5
    test_pct = test_rmse * 100.0

    print("\n===== TEST RESULT =====")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {test_mse**0.5:.6f}")
    print(f"Test % error: {(test_mse**0.5)*100:.2f}%")

    writer.add_scalar("Loss/Test", test_mse, 0)
    writer.add_scalar("RMSE/Test", test_rmse, 0)
    writer.add_scalar("PctError/Test", test_pct, 0)
    writer.add_scalar("MSE_Percentage/Test", test_mse * 100.0, 0)
    writer.add_scalar("R2_Score/Test", test_r2, 0)
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
    dummy_controls = torch.randn(1, seq_len, 2).to(device)

    if USE_DEVIATION:
        dummy_devs = torch.randn(1, seq_len, 1).to(device)
        input_data = (dummy_imgs, dummy_speeds, dummy_devs, dummy_controls)
        input_names = ["images", "speeds", "deviations", "controls"]
    else:
        input_data = (dummy_imgs, dummy_speeds, dummy_controls)
        input_names = ["images", "speeds", "controls"]

    torch.onnx.export(
        model,
        input_data,
        net_file_name,
        input_names=input_names,
        output_names=["output"],
        opset_version=16 
    )

    print("\n========== TRAINING FINISHED ==========")
    print(f"[OK] Best model : {best_path}")
    print(f"[OK] Final model: {final_pth}")
    print(f"[OK] ONNX       : {net_file_name}")
    print("=======================================\n")


if __name__ == "__main__":
    main()