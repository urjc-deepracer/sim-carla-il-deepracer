#!/usr/bin/env python3
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_default_transform():
    return transforms.Compose([
        transforms.Resize((66, 200)),                
        transforms.Normalize(
            mean=[0.5]*3,
            std=[0.5]*3
        )
    ])


class SequenceDataset(Dataset):

    def __init__(self, pt_path, transform=None):

        data = torch.load(pt_path)

        self.images = data["images"] 
        self.labels = data["labels"]
        self.speeds = data["speeds"]
        self.estados = data["estados"]
        self.deviations = data["deviations"]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        seq_imgs = self.images[idx]         # (T,3,66,200)
        seq_speeds = self.speeds[idx]       # (T,)
        seq_deviations = self.deviations[idx]

        if self.transform:
            seq_imgs = torch.stack([self.transform(img) for img in seq_imgs])

        seq_speeds = seq_speeds.unsqueeze(-1)  # (T,1)
        seq_deviations = seq_deviations.unsqueeze(-1)

        label = self.labels[idx]
        estado = self.estados[idx]

        return seq_imgs, seq_speeds, seq_deviations, label, estado



def get_dataloaders(
    train_path,
    val_path,
    test_path,
    batch_size=32,
    transform=None
):

    train_dataset = SequenceDataset(train_path, transform)
    val_dataset   = SequenceDataset(val_path, transform)
    test_dataset  = SequenceDataset(test_path, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def compute_class_weights(estados, device):

    estados_np = estados.numpy()

    counts = np.bincount(estados_np.astype(int), minlength=4)[1:]

    freq = counts / counts.sum()

    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32).to(device)


def weighted_mse(pred, target, estados, w_global):

    idx = (estados.long() - 1).clamp(0, 2)
    pesos = w_global[idx]

    mse = (pred - target) ** 2
    mse = mse * pesos.view(-1, 1)

    return mse.mean()


def mse_to_rmse(m):
    return float(m) ** 0.5


def mse_to_pct(m):
    return mse_to_rmse(m) * 100.0