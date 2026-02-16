#!/usr/bin/env python3
"""
Evaluation script for the fixed-parameter CRB experiment.
Uses datasets generated with dataset_generation_crb.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def circular_distance(pred, true):
    """Compute circular distance for phases in [-π, π]"""
    diff = pred - true
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return diff


def evaluate_model_vs_crb(model, npz_path, M, device="cpu", save_plots=True):
    """
    Evaluate model against fixed-parameter CRB.
    """
    npz_path = Path(npz_path)
    print(f"\n[INFO] Evaluating on: {npz_path}")

    with np.load(npz_path) as data:
        X = torch.from_numpy(data["X"]).float().to(device)
        y = torch.from_numpy(data["y"]).float().to(device)
        snr = data["snr_db"]
        crb_g = data["crb_gain"]    # Shape: (N, M-1)
        crb_p = data["crb_phase"]   # Shape: (N, M-2)

    print(f"[INFO] Dataset: {len(X)} samples")
    print(f"[INFO] Array size M = {M}")
    print(f"[INFO] CRB gain shape: {crb_g.shape}")
    print(f"[INFO] CRB phase shape: {crb_p.shape}")

    model.eval()
    with torch.no_grad():
        y_hat = model(X).cpu().numpy()
    y_true = y.cpu().numpy()

    # Split gains and phases
    gain_pred = y_hat[:, :M]
    gain_true = y_true[:, :M]
    phase_pred = y_hat[:, M:]
    phase_true = y_true[:, M:]

    # Only evaluate estimable parameters
    gain_pred_est = gain_pred[:, 1:]      # ψ₂...ψₘ
    gain_true_est = gain_true[:, 1:]
    phase_pred_est = phase_pred[:, 2:]    # φ₃...φₘ
    phase_true_est = phase_true[:, 2:]

    # Compute MSEs
    gain_err = (gain_pred_est - gain_true_est) ** 2
    mse_gain = gain_err.mean(axis=1)

    phase_diff = circular_distance(phase_pred_est, phase_true_est)
    phase_err = phase_diff ** 2
    mse_phase = phase_err.mean(axis=1)

    # Group by SNR
    by_snr_mse_g, by_snr_mse_p = defaultdict(list), defaultdict(list)
    by_snr_crb_g, by_snr_crb_p = defaultdict(list), defaultdict(list)

    for s, mg, mp, cg, cp in zip(snr, mse_gain, mse_phase,
                                  crb_g.mean(axis=1), crb_p.mean(axis=1)):
        by_snr_mse_g[int(s)].append(mg)
        by_snr_mse_p[int(s)].append(mp)
        by_snr_crb_g[int(s)].append(cg)
        by_snr_crb_p[int(s)].append(cp)

    snr_levels = sorted(by_snr_mse_g.keys())
    mean_mse_g = np.array([np.mean(by_snr_mse_g[s]) for s in snr_levels])
    mean_mse_p = np.array([np.mean(by_snr_mse_p[s]) for s in snr_levels])
    mean_crb_g = np.array([np.mean(by_snr_crb_g[s]) for s in snr_levels])
    mean_crb_p = np.array([np.mean(by_snr_crb_p[s]) for s in snr_levels])

    # Convert to dB
    def to_db(mse):
        return 10 * np.log10(np.maximum(mse, 1e-12))

    mse_g_db = to_db(mean_mse_g)
    mse_p_db = to_db(mean_mse_p)
    crb_g_db = to_db(mean_crb_g)
    crb_p_db = to_db(mean_crb_p)

    # Print statistics
    print(f"\n{'='*80}")
    print(f"GAIN ESTIMATION (ψ₂...ψₘ) - Fixed-Parameter CRB")
    print(f"{'='*80}")
    print(f"{'SNR (dB)':<10} {'CNN MSE':<15} {'CRB':<15} {'CNN [dB]':<12} {'CRB [dB]':<12} {'Gap':<10}")
    print(f"{'-'*80}")
    for s, cnn_g, crb_g_val, cnn_db, crb_db in zip(
        snr_levels, mean_mse_g, mean_crb_g, mse_g_db, crb_g_db):
        gap = cnn_g / crb_g_val if crb_g_val > 1e-10 else float('inf')
        print(f"{s:<10} {cnn_g:<15.6e} {crb_g_val:<15.6e} {cnn_db:<12.2f} {crb_db:<12.2f} {gap:<10.2f}x")

    print(f"\n{'='*80}")
    print(f"PHASE ESTIMATION (φ₃...φₘ) - Fixed-Parameter CRB")
    print(f"{'='*80}")
    print(f"{'SNR (dB)':<10} {'CNN MSE':<15} {'CRB':<15} {'CNN [dB]':<12} {'CRB [dB]':<12} {'Gap':<10}")
    print(f"{'-'*80}")
    for s, cnn_p, crb_p_val, cnn_db, crb_db in zip(
        snr_levels, mean_mse_p, mean_crb_p, mse_p_db, crb_p_db):
        gap = cnn_p / crb_p_val if crb_p_val > 1e-10 else float('inf')
        print(f"{s:<10} {cnn_p:<15.6e} {crb_p_val:<15.6e} {cnn_db:<12.2f} {crb_db:<12.2f} {gap:<10.2f}x")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Gains - linear scale
    ax = axes[0, 0]
    ax.semilogy(snr_levels, mean_crb_g, 'o--', color='orange', linewidth=2,
                markersize=8, label='CRB (fixed params)')
    ax.semilogy(snr_levels, mean_mse_g, 's-', color='blue', linewidth=2,
                markersize=8, label='CNN MSE')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Gain Estimation (ψ₂...ψₘ)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # Phases - linear scale
    ax = axes[0, 1]
    ax.semilogy(snr_levels, mean_crb_p, 'o--', color='orange', linewidth=2,
                markersize=8, label='CRB (fixed params)')
    ax.semilogy(snr_levels, mean_mse_p, 's-', color='blue', linewidth=2,
                markersize=8, label='CNN MSE')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Phase Estimation (φ₃...φₘ)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # Gains - dB scale
    ax = axes[1, 0]
    ax.plot(snr_levels, crb_g_db, 'o--', color='orange', linewidth=2,
            markersize=8, label='CRB (fixed params)')
    ax.plot(snr_levels, mse_g_db, 's-', color='blue', linewidth=2,
            markersize=8, label='CNN MSE')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE (dB)', fontsize=12)
    ax.set_title('Gain Estimation - dB Scale', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, ls='--', alpha=0.4)

    # Phases - dB scale
    ax = axes[1, 1]
    ax.plot(snr_levels, crb_p_db, 'o--', color='orange', linewidth=2,
            markersize=8, label='CRB (fixed params)')
    ax.plot(snr_levels, mse_p_db, 's-', color='blue', linewidth=2,
            markersize=8, label='CNN MSE')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE (dB)', fontsize=12)
    ax.set_title('Phase Estimation - dB Scale', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, ls='--', alpha=0.4)

    plt.suptitle('CNN vs CRB (Fixed Parameters)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_plots:
        plot_path = Path('evaluation_crb.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n[INFO] Plot saved to: {plot_path.resolve()}")

    plt.close()

    return {
        'snr_levels': snr_levels,
        'cnn_gain_mse': mean_mse_g,
        'cnn_phase_mse': mean_mse_p,
        'crb_gain': mean_crb_g,
        'crb_phase': mean_crb_p,
        'cnn_gain_db': mse_g_db,
        'cnn_phase_db': mse_p_db,
        'crb_gain_db': crb_g_db,
        'crb_phase_db': crb_p_db,
    }


if __name__ == "__main__":
    import sys
    import torch.nn as nn
    import math

    class CalibNetDeep(nn.Module):
        """Model definition."""
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

    M = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*80}")
    print(f"EVALUATION WITH FIXED-PARAMETER CRB")
    print(f"{'='*80}")
    print(f"Device: {device}")

    # Load model
    model_path = Path("calibnet_M6_improved.pt")
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    model = CalibNetDeep(M, p_drop=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"[INFO] Loaded model from: {model_path}")

    # Evaluate
    test_path = Path("Datasets/test_crb.npz")
    if not test_path.exists():
        print(f"[ERROR] Test dataset not found: {test_path}")
        print("[INFO]  Please run dataset_generation_crb.py first")
        sys.exit(1)

    results = evaluate_model_vs_crb(model, test_path, M, device=device)
    print("\n[SUCCESS] Evaluation complete!")
