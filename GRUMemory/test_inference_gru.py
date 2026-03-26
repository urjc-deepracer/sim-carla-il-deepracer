#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import csv
from PIL import Image
from torchvision import transforms


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


COMMON_TRANSFORM = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


def load_sequence_from_folder(folder, seq_len=5):
    csv_path = os.path.join(folder, "dataset.csv")

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) < seq_len:
        raise ValueError("Not enough data")

    rows = rows[-seq_len:]

    images = []
    speeds = []
    deviations = []

    for row in rows:
        img_path = os.path.join(folder, row["mask_path"].lstrip("/"))

        img = Image.open(img_path).convert("RGB")
        img = COMMON_TRANSFORM(img)
        images.append(img)

        speed = float(row["speed"])
        speeds.append(speed)

        deviation = float(row["deviation"])
        deviations.append(deviation)

    seq_imgs = torch.stack(images).unsqueeze(0)
    seq_speeds = torch.tensor(speeds, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    seq_deviations = torch.tensor(deviations, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    seq_deviations = seq_deviations / 100.0

    return seq_imgs, seq_speeds, seq_deviations


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_GRU_FUSION().to(device)
    model.load_state_dict(
        torch.load("./experiments/exp_20260326_194305/best_model.pth", map_location=device)
    )
    model.eval()

    folder = "test_inference_dir"

    seq_imgs, seq_speeds, seq_deviations = load_sequence_from_folder(folder, seq_len=5)
    seq_imgs = seq_imgs.to(device).float()
    seq_speeds = seq_speeds.to(device).float()
    seq_deviations = seq_deviations.to(device).float()

    print("\n===== DEBUG INPUT =====")
    print("seq_imgs shape:", seq_imgs.shape)
    print("seq_speeds shape:", seq_speeds.shape)
    print("last speed:", seq_speeds[0, -1, 0].item())
    print("last deviation:", seq_deviations[0, -1, 0].item())

    with torch.no_grad():
        output = model(seq_imgs, seq_speeds, seq_deviations)

    steer = output[0, 0].item()
    throttle = output[0, 1].item()

    print("\n===== INFERENCE =====")
    print(f"Steer: {steer:.4f}")
    print(f"Throttle: {throttle:.4f}")


if __name__ == "__main__":
    main()