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

        self.gru_speed = nn.GRU(
            input_size=1,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_size + 32, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, images, speeds):
        B, T, C, H, W = images.shape

        x = images.view(B * T, C, H, W)
        feats_img = self.cnn(x)
        feats_img = feats_img.view(B, T, -1)
        feats_img = feats_img[:, -1, :]

        out_speed, _ = self.gru_speed(speeds)
        feats_speed = out_speed[:, -1, :]

        fused = torch.cat([feats_img, feats_speed], dim=1)

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
        raise ValueError("No hay suficientes datos")

    rows = rows[-seq_len:]

    images = []
    speeds = []

    for row in rows:
        img_path = os.path.join(folder, row["mask_path"].lstrip("/"))

        img = Image.open(img_path).convert("RGB")
        img = COMMON_TRANSFORM(img)
        images.append(img)

        speed = float(row["speed"])
        speeds.append(speed)

    seq_imgs = torch.stack(images).unsqueeze(0)
    seq_speeds = torch.tensor(speeds, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    return seq_imgs, seq_speeds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_GRU_FUSION().to(device)
    model.load_state_dict(
        torch.load("./experiments/exp_20260325_200517/best_model.pth", map_location=device)
    )
    model.eval()

    folder = "test_inference_dir"

    seq_imgs, seq_speeds = load_sequence_from_folder(folder, seq_len=5)
    seq_imgs = seq_imgs.to(device).float()
    seq_speeds = seq_speeds.to(device).float()

    print("\n===== DEBUG INPUT =====")
    print("seq_imgs shape:", seq_imgs.shape)
    print("seq_speeds shape:", seq_speeds.shape)
    print("last speed:", seq_speeds[0, -1, 0].item())

    with torch.no_grad():
        output = model(seq_imgs, seq_speeds)

    steer = output[0, 0].item()
    throttle = output[0, 1].item()

    print("\n===== INFERENCIA =====")
    print(f"Steer: {steer:.4f}")
    print(f"Throttle: {throttle:.4f}")


if __name__ == "__main__":
    main()