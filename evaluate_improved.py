#!/usr/bin/env python3
"""
Improved evaluation script with circular phase distance metric.
Compares CNN performance against BCRB across SNR levels.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import math

def circular_distance(pred, true):
    """Compute circular distance for phases in [-π, π]"""
    diff = pred - true
    # Wrap to [-π, π]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return diff

def evaluate_model_vs_bcrb(model, npz_path, M, device="cuda", save_plots=True):
    """
    Evaluate model against BCRB using circular phase distance.

    Args:
        model: Trained CalibNet model
        npz_path: Path to test dataset with BCRB labels
        M: Number of array elements
        device: 'cuda' or 'cpu'
        save_plots: Whether to save plots to disk
    """
    if not torch.cuda.is_available():
        device = "cpu"

    npz_path = Path(npz_path)
    print(f"\n[INFO] Evaluating on: {npz_path}")

    with np.load(npz_path) as data:
        X = torch.from_numpy(data["X"]).float().to(device)
        y = torch.from_numpy(data["y"]).float().to(device)
        snr = data["snr_db"]
        bcrb_g = data["bcrb_gain"]
        bcrb_p = data["bcrb_phase"]

    model.eval()
    with torch.no_grad():
        y_hat = model(X).cpu().numpy()
    y_true = y.cpu().numpy()

    # Split gains and phases
    gain_pred = y_hat[:, :M]
    gain_true = y_true[:, :M]
    phase_pred = y_hat[:, M:]
    phase_true = y_true[:, M:]

    # Compute MSEs
    # Gains: regular squared error
    gain_err = (gain_pred - gain_true) ** 2
    mse_gain = gain_err.mean(axis=1)

    # Phases: circular squared error
    phase_diff = circular_distance(phase_pred, phase_true)
    phase_err = phase_diff ** 2
    mse_phase = phase_err.mean(axis=1)

    # Group by SNR
    by_snr_mse_g, by_snr_mse_p = defaultdict(list), defaultdict(list)
    by_snr_bcrb_g, by_snr_bcrb_p = defaultdict(list), defaultdict(list)

    for s, mg, mp, bg, bp in zip(snr, mse_gain, mse_phase, bcrb_g.mean(axis=1), bcrb_p.mean(axis=1)):
        by_snr_mse_g[int(s)].append(mg)
        by_snr_mse_p[int(s)].append(mp)
        by_snr_bcrb_g[int(s)].append(bg)
        by_snr_bcrb_p[int(s)].append(bp)

    snr_levels = sorted(by_snr_mse_g.keys())
    mean_mse_g = [np.mean(by_snr_mse_g[s]) for s in snr_levels]
    mean_mse_p = [np.mean(by_snr_mse_p[s]) for s in snr_levels]
    mean_bcrb_g = [np.mean(by_snr_bcrb_g[s]) for s in snr_levels]
    mean_bcrb_p = [np.mean(by_snr_bcrb_p[s]) for s in snr_levels]

    # Print statistics
    print(f"\n{'='*70}")
    print(f"{'SNR (dB)':<10} {'CNN Gain':<15} {'BCRB Gain':<15} {'Gap':<10}")
    print(f"{'-'*70}")
    for s, cnn_g, bcrb_g_val in zip(snr_levels, mean_mse_g, mean_bcrb_g):
        gap = cnn_g / bcrb_g_val if bcrb_g_val > 1e-10 else float('inf')
        print(f"{s:<10} {cnn_g:<15.6e} {bcrb_g_val:<15.6e} {gap:<10.2f}x")
    print(f"{'='*70}\n")

    print(f"{'SNR (dB)':<10} {'CNN Phase':<15} {'BCRB Phase':<15} {'Gap':<10}")
    print(f"{'-'*70}")
    for s, cnn_p, bcrb_p_val in zip(snr_levels, mean_mse_p, mean_bcrb_p):
        gap = cnn_p / bcrb_p_val if bcrb_p_val > 1e-10 else float('inf')
        print(f"{s:<10} {cnn_p:<15.6e} {bcrb_p_val:<15.6e} {gap:<10.2f}x")
    print(f"{'='*70}\n")

    # Plot gains
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(snr_levels, mean_bcrb_g, 'o-', color='orange', linewidth=2,
             markersize=8, label='BCRB (gains)')
    ax1.plot(snr_levels, mean_mse_g, 's-', color='blue', linewidth=2,
             markersize=8, label='CNN MSE (gains)')
    ax1.set_yscale('log')
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Average error / bound', fontsize=12)
    ax1.set_title('Gains: CNN MSE vs BCRB', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, which='both', ls='--', alpha=0.4)

    # Plot phases
    ax2.plot(snr_levels, mean_bcrb_p, 'o-', color='orange', linewidth=2,
             markersize=8, label='BCRB (phases)')
    ax2.plot(snr_levels, mean_mse_p, 's-', color='blue', linewidth=2,
             markersize=8, label='CNN MSE (phases)')
    ax2.set_yscale('log')
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Average error / bound', fontsize=12)
    ax2.set_title('Phases: CNN MSE vs BCRB', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()

    if save_plots:
        plot_path = Path('output_improved.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Plots saved to: {plot_path.resolve()}")

    plt.show()

    return {
        'snr_levels': snr_levels,
        'cnn_gain': mean_mse_g,
        'cnn_phase': mean_mse_p,
        'bcrb_gain': mean_bcrb_g,
        'bcrb_phase': mean_bcrb_p
    }


if __name__ == "__main__":
    # Example usage
    import sys
    from main_improved import CalibNetDeep

    M = 6  # Array size
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
