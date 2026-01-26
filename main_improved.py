#!/usr/bin/env python3
"""
IMPROVED training script for blind array-calibration with:
  1. Circular phase loss function (handles phase wrap-around)
  2. Deeper CalibNet architecture for better capacity
  3. Learning rate scheduling
  4. Training on FULL SNR range for better generalization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import math

# -------------------------------------------------
# 1) File locations & hyper-params
# -------------------------------------------------
BASE_DIR = Path("Datasets")
TRAIN_PATH = BASE_DIR / "train_full_snr.npz"      # NEW: Full SNR training set
TEST_PATH = BASE_DIR / "test_full_snr_bcrb.npz"   # NEW: Larger test set

BATCH = 64
LR = 1e-3
EPOCHS = 500
VAL_FRAC = 0.10
TEST_FRAC = 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_SEED = 0

print(f"[INFO] Using device: {DEVICE}")

# -------------------------------------------------
# 2) Peek at TRAIN_PATH to discover M
# -------------------------------------------------
assert TRAIN_PATH.exists(), f"TRAIN_PATH not found: {TRAIN_PATH}"
with np.load(TRAIN_PATH) as npz:
    n, ch, M, _ = npz["X"].shape
    assert ch == 2, "Expected 2-channel (real/imag) input"
print(f"[INFO] Using TRAIN_PATH = {TRAIN_PATH}")
print(f"[INFO] Detected dataset with N = {n} samples, array size M = {M}")

# -------------------------------------------------
# 3) Dataset wrapper
# -------------------------------------------------
class CovDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"]).float()  # (N, 2, M, M)
        self.y = torch.from_numpy(data["y"]).float()  # (N, 2M)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

torch.manual_seed(TORCH_SEED)
full_ds = CovDataset(TRAIN_PATH)

N = len(full_ds)
n_val = int(VAL_FRAC * N)
n_test = int(TEST_FRAC * N)
n_train = N - n_val - n_test
train_ds, val_ds, test_ds = random_split(
    full_ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(TORCH_SEED)
)

train_loader = DataLoader(train_ds, BATCH, shuffle=True)
val_loader = DataLoader(val_ds, BATCH, shuffle=False)
test_loader = DataLoader(test_ds, BATCH, shuffle=False)

print(f"[INFO] Train: {n_train}, Val: {n_val}, Test: {n_test}")

# -------------------------------------------------
# 4) IMPROVED: Deeper CalibNet with more capacity
# -------------------------------------------------
class CalibNetDeep(nn.Module):
    """
    Deeper CalibNet with increased capacity:
      - 4 conv layers (2→64→128→128→256)
      - 3-layer MLP head with dropout
      - Better suited for complex inverse problems
    """
    def __init__(self, M, p_drop=0.3):
        super().__init__()
        self.M = M

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B, 256, 1, 1)
            nn.Flatten()              # (B, 256)
        )

        self.head = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(256, 2*M)
        )

        self.softplus = nn.Softplus(beta=1.5)

    def forward(self, x):
        z = self.cnn(x)                                 # (B, 256)
        out = self.head(z)                              # (B, 2M)
        psi_raw, phi_raw = out[:, :self.M], out[:, self.M:]
        psi_hat = self.softplus(psi_raw)                # Ψ ∈ [0, ∞)
        phi_hat = torch.tanh(phi_raw) * math.pi         # Φ ∈ (-π, π)
        return torch.cat([psi_hat, phi_hat], dim=1)     # (B, 2M)

# -------------------------------------------------
# 5) CRITICAL FIX: Circular phase loss
# -------------------------------------------------
def circular_mse_loss(y_pred, y_true, M):
    """
    MSE loss with circular distance for phases.
    Handles phase wrap-around: angle difference is computed on circle.

    Args:
        y_pred: (B, 2M) predictions [gains, phases]
        y_true: (B, 2M) ground truth [gains, phases]
        M: Number of array elements

    Returns:
        Combined loss (gain MSE + circular phase MSE)
    """
    # Gains: regular MSE
    gain_pred = y_pred[:, :M]
    gain_true = y_true[:, :M]
    gain_loss = F.mse_loss(gain_pred, gain_true)

    # Phases: circular distance
    phase_pred = y_pred[:, M:]
    phase_true = y_true[:, M:]
    phase_diff = phase_pred - phase_true

    # Wrap difference to [-π, π] using atan2
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = (phase_diff ** 2).mean()

    return gain_loss + phase_loss

# -------------------------------------------------
# 6) Initialize model, optimizer, scheduler
# -------------------------------------------------
model = CalibNetDeep(M, p_drop=0.3).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Learning rate scheduler: reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=20, verbose=True
)

print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------------------------------
# 7) Training helper with circular loss
# -------------------------------------------------
def run_epoch(loader, training=False):
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if training:
                opt.zero_grad()
            y_hat = model(X)
            loss = circular_mse_loss(y_hat, y, M)  # Use circular loss!
            if training:
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

# -------------------------------------------------
# 8) Training loop with scheduler
# -------------------------------------------------
best_val = float("inf")
best_ckpt_path = Path(f"calibnet_M{M}_improved.pt")

print("\n" + "="*60)
print("Starting training with improved architecture and circular loss")
print("="*60 + "\n")

for epoch in range(1, EPOCHS+1):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    # Step scheduler based on validation loss
    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), best_ckpt_path)
        status = "✓ (saved)"
    else:
        status = ""

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS}  "
              f"train loss {train_loss:.6f}  "
              f"val loss {val_loss:.6f}  {status}")

# -------------------------------------------------
# 9) Final evaluation
# -------------------------------------------------
model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
test_loss = run_epoch(test_loader, training=False)
print(f"\n{'='*60}")
print(f"[RESULT] Best model internal test loss = {test_loss:.6f}")
print(f"[INFO]   Model weights saved to {best_ckpt_path.resolve()}")
print(f"{'='*60}\n")

# -------------------------------------------------
# 10) Evaluate on external test set with BCRB
# -------------------------------------------------
if TEST_PATH.exists():
    external_test = CovDataset(TEST_PATH)
    ext_loader = DataLoader(external_test, BATCH, shuffle=False)
    ext_loss = run_epoch(ext_loader, training=False)
    print(f"[RESULT] External test loss ({TEST_PATH.name}) = {ext_loss:.6f}")
    print(f"[INFO]   Ready for BCRB comparison - use evaluate_model_vs_bcrb()")
else:
    print(f"[WARN]   TEST_PATH not found: {TEST_PATH}")
    print(f"[INFO]   Run dataset_generation.py to create test set")
