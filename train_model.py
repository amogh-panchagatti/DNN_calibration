#!/usr/bin/env python3
"""
Training script for CalibNetDeep model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from pathlib import Path


class CalibNetDeep(nn.Module):
    def __init__(self, M, p_drop=0.3):
        super().__init__()
        self.M = M
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(256, 2*M)
        )
        self.softplus = nn.Softplus(beta=1.5)

    def forward(self, x):
        z = self.cnn(x)
        out = self.head(z)
        psi_raw, phi_raw = out[:, :self.M], out[:, self.M:]
        psi_hat = self.softplus(psi_raw)
        phi_hat = torch.tanh(phi_raw) * math.pi
        return torch.cat([psi_hat, phi_hat], dim=1)


def combined_loss(pred, true, M):
    """Combined loss for gains and phases."""
    pred_psi, pred_phi = pred[:, :M], pred[:, M:]
    true_psi, true_phi = true[:, :M], true[:, M:]

    # Gains MSE (skip reference ψ₁=1)
    loss_psi = torch.mean((pred_psi[:, 1:] - true_psi[:, 1:]) ** 2)

    # Phases circular MSE (skip reference φ₁=φ₂=0)
    diff = pred_phi[:, 2:] - true_phi[:, 2:]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    loss_phi = torch.mean(diff ** 2)

    return loss_psi + loss_phi


def train(train_path, save_path, M=6, epochs=100, batch_size=128, lr=1e-3, device="cpu"):
    print("="*70)
    print(f"Training CalibNetDeep | Device: {device}")
    print("="*70)

    # Load data
    data = np.load(train_path)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    print(f"Loaded {len(X)} samples from {train_path}")

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model = CalibNetDeep(M, p_drop=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(bX), by, M)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bX)

        avg_loss = total_loss / len(X)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f} (best = {best_loss:.6f})")

    print(f"\nTraining complete! Model saved to: {save_path}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        train_path="Datasets/train_full_snr.npz",
        save_path="calibnet_M6_improved.pt",
        M=6, epochs=100, batch_size=128, lr=1e-3, device=device
    )
