#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import csv
from PIL import Image
from torchvision import transforms


#CNN → concat features → 1 GRU → FC
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

        self.gru = nn.GRU(
            input_size=self.cnn_out_size + 2 + 2,  # img + (speed,dev) + (steer,throttle)
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  # steer, throttle
        )

    def forward(self, images, speeds, deviations, controls):

        T = min(
            images.shape[1],
            speeds.shape[1],
            deviations.shape[1],
            controls.shape[1]
        )

        images = images[:, :T]
        speeds = speeds[:, :T]
        deviations = deviations[:, :T]
        controls = controls[:, :T]

        controls = controls.detach()

        B, T, C, H, W = images.shape

        x = images.view(B*T, C, H, W)
        feats_img = self.cnn(x)
        feats_img = feats_img.view(B, T, -1)


        dyn = torch.cat([speeds, deviations], dim=2)      # (B,T,2)
        fused_seq = torch.cat([feats_img, dyn, controls], dim=2)  # (B,T,F)

        out, _ = self.gru(fused_seq)
        feat = out[:, -1, :]   # (B,64)

        return self.fc(feat)


def load_sequence_from_csv(csv_path, seq_len=10):
    rows = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) < seq_len:
        raise ValueError("Not enough rows in CSV")

    rows = rows[-seq_len:]
    base_folder = os.path.dirname(csv_path)

    images = []
    speeds = []
    deviations = []
    controls = []

    transform = transforms.Compose([
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    for row in rows:
        img_path = os.path.join(base_folder, row["mask_path"].lstrip("/"))

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        images.append(img)

        speeds.append([float(row["speed"])])
        deviations.append([float(row["deviation"]) / 100.0])
        controls.append([float(row["steer"]), float(row["throttle"])])

    seq_imgs = torch.stack(images).unsqueeze(0)        # (1,T,3,H,W)
    seq_speeds = torch.tensor(speeds).unsqueeze(0)     # (1,T,1)
    seq_deviations = torch.tensor(deviations).unsqueeze(0)

    seq_controls = torch.tensor(controls).unsqueeze(0)  # (1,T,2)

    return seq_imgs, seq_speeds, seq_deviations, seq_controls, rows

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_GRU_FUSION().to(device)

    model_path = "./experiments/exp_20260406_175407/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    csv_path = "./test_inference_dir/dataset.csv"

    seq_len = 10
    seq_imgs, seq_speeds, seq_deviations, seq_controls, rows = load_sequence_from_csv(
        csv_path,
        seq_len=seq_len
    )

    seq_imgs = seq_imgs.to(device).float()
    seq_speeds = seq_speeds.to(device).float()
    seq_deviations = seq_deviations.to(device).float()
    seq_controls = seq_controls.to(device).float()

    print("\n===== DEBUG INPUT =====")
    print("seq_imgs shape       :", seq_imgs.shape)
    print("seq_speeds shape     :", seq_speeds.shape)
    print("seq_deviations shape :", seq_deviations.shape)
    print("seq_controls shape   :", seq_controls.shape)
    print("last csv speed       :", float(rows[-1]["speed"]))
    print("last csv deviation   :", float(rows[-1]["deviation"]))
    print("last csv steer GT    :", float(rows[-1]["steer"]))
    print("last csv throttle GT :", float(rows[-1]["throttle"]))

    with torch.no_grad():
        output = model(seq_imgs, seq_speeds, seq_deviations, seq_controls)

    pred_steer = output[0, 0].item()
    pred_throttle = output[0, 1].item()

    print("\n===== INFERENCE =====")
    print(f"Pred steer    : {pred_steer:.6f}")
    print(f"Pred throttle : {pred_throttle:.6f}")

    print("\n===== GROUND TRUTH (última fila del CSV) =====")
    print(f"GT steer      : {float(rows[-1]['steer']):.6f}")
    print(f"GT throttle   : {float(rows[-1]['throttle']):.6f}")


if __name__ == "__main__":
    main()