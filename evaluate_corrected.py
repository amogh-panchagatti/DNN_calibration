#!/usr/bin/env python3
"""
Corrected evaluation script aligned with EUSIPCO 2025 paper notation.
Compares CNN performance against BCRB using proper parameter indexing.

Key fixes:
1. Only evaluate estimable parameters: ψ[2:M] and φ[3:M]
2. Plot in dB as in the paper
3. Match BCRB computation from the hybrid CRB formulation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import pandas as pd

def circular_distance(pred, true):
    """Compute circular distance for phases in [-π, π]"""
    diff = pred - true
    # Wrap to [-π, π]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return diff

def evaluate_model_vs_bcrb(model, npz_path, M, device="cuda", save_plots=True,
                           save_detailed_results=True, num_samples_to_show=10):
    """
    Evaluate model against BCRB with corrected parameter indexing.

    Following EUSIPCO 2025 paper notation:
    - Gains ψ = [ψ₁, ψ₂, ..., ψₘ] where ψ₁ = 1 (reference)
    - Phases φ = [φ₁, φ₂, ..., φₘ] where φ₁ = φ₂ = 0 (reference)
    - BCRB only computed for ψ[2:M] (M-1 gains) and φ[3:M] (M-2 phases)

    Args:
        model: Trained CalibNet model
        npz_path: Path to test dataset with BCRB labels
        M: Number of array elements
        device: 'cuda' or 'cpu'
        save_plots: Whether to save plots to disk
        save_detailed_results: Whether to save detailed CSV tables
        num_samples_to_show: Number of sample predictions to display
    """
    if not torch.cuda.is_available():
        device = "cpu"

    npz_path = Path(npz_path)
    print(f"\n[INFO] Evaluating on: {npz_path}")

    with np.load(npz_path) as data:
        X = torch.from_numpy(data["X"]).float().to(device)
        y = torch.from_numpy(data["y"]).float().to(device)
        snr = data["snr_db"]
        bcrb_g = data["bcrb_gain"]    # Shape: (N, M-1) for ψ[2:M]
        bcrb_p = data["bcrb_phase"]   # Shape: (N, M-2) for φ[3:M]

    print(f"[INFO] Dataset: {len(X)} samples")
    print(f"[INFO] Array size M = {M}")
    print(f"[INFO] BCRB gain shape: {bcrb_g.shape} (estimating ψ₂...ψₘ)")
    print(f"[INFO] BCRB phase shape: {bcrb_p.shape} (estimating φ₃...φₘ)")

    model.eval()
    with torch.no_grad():
        y_hat = model(X).cpu().numpy()
    y_true = y.cpu().numpy()

    # Split gains and phases
    gain_pred = y_hat[:, :M]      # (N, M)
    gain_true = y_true[:, :M]     # (N, M)
    phase_pred = y_hat[:, M:]     # (N, M)
    phase_true = y_true[:, M:]    # (N, M)

    # ================================================================
    # CRITICAL FIX: Only evaluate on ESTIMABLE parameters
    # ================================================================
    # Gains: ψ₂...ψₘ (skip ψ₁ which is fixed at 1)
    gain_pred_est = gain_pred[:, 1:]      # (N, M-1)
    gain_true_est = gain_true[:, 1:]      # (N, M-1)

    # Phases: φ₃...φₘ (skip φ₁, φ₂ which are fixed at 0)
    phase_pred_est = phase_pred[:, 2:]    # (N, M-2)
    phase_true_est = phase_true[:, 2:]    # (N, M-2)

    print(f"\n[INFO] Evaluating on estimable parameters only:")
    print(f"  - Gains: ψ₂...ψₘ, shape = {gain_pred_est.shape}")
    print(f"  - Phases: φ₃...φₘ, shape = {phase_pred_est.shape}")

    # Compute MSEs on estimable parameters
    # Gains: regular squared error
    gain_err = (gain_pred_est - gain_true_est) ** 2
    mse_gain = gain_err.mean(axis=1)  # Average over M-1 gains

    # Phases: circular squared error
    phase_diff = circular_distance(phase_pred_est, phase_true_est)
    phase_err = phase_diff ** 2
    mse_phase = phase_err.mean(axis=1)  # Average over M-2 phases

    # Group by SNR
    by_snr_mse_g, by_snr_mse_p = defaultdict(list), defaultdict(list)
    by_snr_bcrb_g, by_snr_bcrb_p = defaultdict(list), defaultdict(list)

    for s, mg, mp, bg, bp in zip(snr, mse_gain, mse_phase,
                                   bcrb_g.mean(axis=1), bcrb_p.mean(axis=1)):
        by_snr_mse_g[int(s)].append(mg)
        by_snr_mse_p[int(s)].append(mp)
        by_snr_bcrb_g[int(s)].append(bg)
        by_snr_bcrb_p[int(s)].append(bp)

    snr_levels = sorted(by_snr_mse_g.keys())
    mean_mse_g = np.array([np.mean(by_snr_mse_g[s]) for s in snr_levels])
    mean_mse_p = np.array([np.mean(by_snr_mse_p[s]) for s in snr_levels])
    mean_bcrb_g = np.array([np.mean(by_snr_bcrb_g[s]) for s in snr_levels])
    mean_bcrb_p = np.array([np.mean(by_snr_bcrb_p[s]) for s in snr_levels])

    # ================================================================
    # Convert to dB as in EUSIPCO paper
    # ================================================================
    def to_db(mse):
        """Convert MSE to dB: 10*log10(MSE)"""
        return 10 * np.log10(np.maximum(mse, 1e-12))  # Avoid log(0)

    mse_g_db = to_db(mean_mse_g)
    mse_p_db = to_db(mean_mse_p)
    bcrb_g_db = to_db(mean_bcrb_g)
    bcrb_p_db = to_db(mean_bcrb_p)

    # ================================================================
    # Print statistics
    # ================================================================
    print(f"\n{'='*80}")
    print(f"GAIN ESTIMATION (ψ₂...ψₘ)")
    print(f"{'='*80}")
    print(f"{'SNR (dB)':<10} {'CNN MSE':<15} {'BCRB':<15} {'CNN [dB]':<12} {'BCRB [dB]':<12} {'Gap':<10}")
    print(f"{'-'*80}")
    for s, cnn_g, bcrb_g_val, cnn_db, bcrb_db in zip(
        snr_levels, mean_mse_g, mean_bcrb_g, mse_g_db, bcrb_g_db):
        gap = cnn_g / bcrb_g_val if bcrb_g_val > 1e-10 else float('inf')
        print(f"{s:<10} {cnn_g:<15.6e} {bcrb_g_val:<15.6e} {cnn_db:<12.2f} {bcrb_db:<12.2f} {gap:<10.2f}x")
    print(f"{'='*80}\n")

    print(f"{'='*80}")
    print(f"PHASE ESTIMATION (φ₃...φₘ)")
    print(f"{'='*80}")
    print(f"{'SNR (dB)':<10} {'CNN MSE':<15} {'BCRB':<15} {'CNN [dB]':<12} {'BCRB [dB]':<12} {'Gap':<10}")
    print(f"{'-'*80}")
    for s, cnn_p, bcrb_p_val, cnn_db, bcrb_db in zip(
        snr_levels, mean_mse_p, mean_bcrb_p, mse_p_db, bcrb_p_db):
        gap = cnn_p / bcrb_p_val if bcrb_p_val > 1e-10 else float('inf')
        print(f"{s:<10} {cnn_p:<15.6e} {bcrb_p_val:<15.6e} {cnn_db:<12.2f} {bcrb_db:<12.2f} {gap:<10.2f}x")
    print(f"{'='*80}\n")

    # ================================================================
    # Create plots (both linear and dB scale)
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Row 1: Linear scale (log-log)
    # Gains - linear scale
    ax = axes[0, 0]
    ax.plot(snr_levels, mean_bcrb_g, 'o--', color='orange', linewidth=2,
            markersize=8, label='BCRB')
    ax.plot(snr_levels, mean_mse_g, 's-', color='blue', linewidth=2,
            markersize=8, label='CNN MSE')
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE (linear scale)', fontsize=12)
    ax.set_title('Gain Estimation (ψ₂...ψₘ) - Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # Phases - linear scale
    ax = axes[0, 1]
    ax.plot(snr_levels, mean_bcrb_p, 'o--', color='orange', linewidth=2,
            markersize=8, label='BCRB')
    ax.plot(snr_levels, mean_mse_p, 's-', color='blue', linewidth=2,
            markersize=8, label='CNN MSE')
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE (linear scale)', fontsize=12)
    ax.set_title('Phase Estimation (φ₃...φₘ) - Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # Row 2: dB scale (as in EUSIPCO paper)
    # Gains - dB scale
    ax = axes[1, 0]
    ax.plot(snr_levels, bcrb_g_db, 'o--', color='orange', linewidth=2,
            markersize=8, label='BCRB')
    ax.plot(snr_levels, mse_g_db, 's-', color='blue', linewidth=2,
            markersize=8, label='CNN MSE')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE (dB)', fontsize=12)
    ax.set_title('Gain Estimation (ψ₂...ψₘ) - dB Scale', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # Phases - dB scale
    ax = axes[1, 1]
    ax.plot(snr_levels, bcrb_p_db, 'o--', color='orange', linewidth=2,
            markersize=8, label='BCRB')
    ax.plot(snr_levels, mse_p_db, 's-', color='blue', linewidth=2,
            markersize=8, label='CNN MSE')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE (dB)', fontsize=12)
    ax.set_title('Phase Estimation (φ₃...φₘ) - dB Scale', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()

    if save_plots:
        plot_path = Path('evaluation_corrected.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Plots saved to: {plot_path.resolve()}")

    plt.show()

    return {
        'snr_levels': snr_levels,
        'cnn_gain_linear': mean_mse_g,
        'cnn_phase_linear': mean_mse_p,
        'bcrb_gain_linear': mean_bcrb_g,
        'bcrb_phase_linear': mean_bcrb_p,
        'cnn_gain_db': mse_g_db,
        'cnn_phase_db': mse_p_db,
        'bcrb_gain_db': bcrb_g_db,
        'bcrb_phase_db': bcrb_p_db,
    }


if __name__ == "__main__":
    import sys
    from main_improved import CalibNetDeep

    M = 6  # Array size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*80}")
    print(f"CORRECTED EVALUATION ALIGNED WITH EUSIPCO 2025 PAPER")
    print(f"{'='*80}")
    print(f"Model: CalibNetDeep with M={M} sensors")
    print(f"Device: {device}")
    print(f"\nKey corrections:")
    print(f"  1. Only evaluate estimable parameters: ψ[2:M], φ[3:M]")
    print(f"  2. Use circular distance for phases")
    print(f"  3. Plot in both linear and dB scale")
    print(f"{'='*80}\n")

    # Load model
    model_path = Path("calibnet_M6_improved.pt")
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("[INFO]  Please run main_improved.py first to train the model")
        sys.exit(1)

    model = CalibNetDeep(M, p_drop=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Loaded model from: {model_path}")

    # Evaluate
    test_path = Path("Datasets/test_full_snr_bcrb.npz")
    if not test_path.exists():
        print(f"[ERROR] Test dataset not found: {test_path}")
        print("[INFO]  Please run dataset_generation.py first")
        sys.exit(1)

    results = evaluate_model_vs_bcrb(model, test_path, M, device=device)
    print("\n[SUCCESS] Evaluation complete!")
    print(f"\n[INFO] Results summary:")
    print(f"  - Gain MSE range: {results['cnn_gain_db'].min():.2f} to {results['cnn_gain_db'].max():.2f} dB")
    print(f"  - Phase MSE range: {results['cnn_phase_db'].min():.2f} to {results['cnn_phase_db'].max():.2f} dB")
