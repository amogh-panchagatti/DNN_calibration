#!/usr/bin/env python3
"""
Deep analysis of why the neural network doesn't reach the CRB.

Key factors investigated:
1. Bias vs Variance decomposition
2. Training/test distribution mismatch
3. Finite sample effects
4. Parameter-wise analysis
5. Error distribution characteristics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import torch.nn as nn
import math

# ============================================================================
# Model Definition (copied to avoid import issues)
# ============================================================================
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


def circular_distance(pred, true):
    """Compute circular distance for phases in [-π, π]"""
    diff = pred - true
    return np.arctan2(np.sin(diff), np.cos(diff))


def analyze_crb_gap(model, npz_path, M, device="cpu"):
    """
    Comprehensive analysis of why the neural network doesn't reach the CRB.
    """
    print("="*80)
    print("COMPREHENSIVE CRB GAP ANALYSIS")
    print("="*80)

    # Load data
    with np.load(npz_path) as data:
        X = torch.from_numpy(data["X"]).float().to(device)
        y = torch.from_numpy(data["y"]).float()
        snr = data["snr_db"]
        bcrb_g = data.get("bcrb_gain", data.get("crb_gain"))
        bcrb_p = data.get("bcrb_phase", data.get("crb_phase"))

    # Model predictions
    model.eval()
    with torch.no_grad():
        y_hat = model(X).cpu().numpy()
    y_true = y.numpy()

    # Split into gains and phases
    gain_pred = y_hat[:, :M]
    gain_true = y_true[:, :M]
    phase_pred = y_hat[:, M:]
    phase_true = y_true[:, M:]

    # Estimable parameters only
    gain_pred_est = gain_pred[:, 1:]  # ψ[2:M]
    gain_true_est = gain_true[:, 1:]
    phase_pred_est = phase_pred[:, 2:]  # φ[3:M]
    phase_true_est = phase_true[:, 2:]

    # Errors
    gain_err = gain_pred_est - gain_true_est
    phase_err = circular_distance(phase_pred_est, phase_true_est)

    # Group by SNR
    snr_levels = sorted(np.unique(snr))

    # ========================================================================
    # FACTOR 1: Bias-Variance Decomposition
    # ========================================================================
    print("\n" + "="*80)
    print("FACTOR 1: BIAS-VARIANCE DECOMPOSITION")
    print("="*80)
    print("\nCRB is a bound for UNBIASED estimators. Neural networks trained with")
    print("MSE loss learn MMSE (Minimum Mean Squared Error) estimators, which")
    print("can be BIASED, especially at low SNR.")
    print("\nMSE = Bias² + Variance")
    print("\n" + "-"*80)

    print(f"{'SNR':<8} | {'Gain Bias²':<12} | {'Gain Var':<12} | {'Gain MSE':<12} | {'BCRB':<12} | {'Bias/MSE':<10}")
    print("-"*80)

    bias_contribution_gain = []
    var_contribution_gain = []

    for s in snr_levels:
        mask = snr == s

        # Gain analysis
        g_err = gain_err[mask]
        g_bias = np.mean(g_err, axis=0)  # Per-parameter bias
        g_var = np.var(g_err, axis=0)    # Per-parameter variance

        mean_bias_sq = np.mean(g_bias**2)
        mean_var = np.mean(g_var)
        mean_mse = np.mean(g_err**2)
        mean_bcrb = np.mean(bcrb_g[mask])

        bias_ratio = mean_bias_sq / mean_mse if mean_mse > 0 else 0
        bias_contribution_gain.append(bias_ratio)
        var_contribution_gain.append(1 - bias_ratio)

        print(f"{s:<8} | {mean_bias_sq:<12.2e} | {mean_var:<12.2e} | {mean_mse:<12.2e} | {mean_bcrb:<12.2e} | {bias_ratio*100:<10.1f}%")

    print("\n" + "-"*80)
    print(f"{'SNR':<8} | {'Phase Bias²':<12} | {'Phase Var':<12} | {'Phase MSE':<12} | {'BCRB':<12} | {'Bias/MSE':<10}")
    print("-"*80)

    bias_contribution_phase = []

    for s in snr_levels:
        mask = snr == s

        # Phase analysis
        p_err = phase_err[mask]
        p_bias = np.mean(p_err, axis=0)
        p_var = np.var(p_err, axis=0)

        mean_bias_sq = np.mean(p_bias**2)
        mean_var = np.mean(p_var)
        mean_mse = np.mean(p_err**2)
        mean_bcrb = np.mean(bcrb_p[mask])

        bias_ratio = mean_bias_sq / mean_mse if mean_mse > 0 else 0
        bias_contribution_phase.append(bias_ratio)

        print(f"{s:<8} | {mean_bias_sq:<12.2e} | {mean_var:<12.2e} | {mean_mse:<12.2e} | {mean_bcrb:<12.2e} | {bias_ratio*100:<10.1f}%")

    # ========================================================================
    # FACTOR 2: MSE/CRB Ratio Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("FACTOR 2: MSE/CRB RATIO (Efficiency Analysis)")
    print("="*80)
    print("\nAn efficient estimator achieves MSE = CRB (ratio = 1.0)")
    print("Ratio > 1 means the estimator is suboptimal")
    print("\n" + "-"*80)

    print(f"{'SNR':<8} | {'Gain Ratio':<12} | {'Phase Ratio':<12} | {'Gain Gap [dB]':<14} | {'Phase Gap [dB]':<14}")
    print("-"*80)

    gain_ratios = []
    phase_ratios = []

    for s in snr_levels:
        mask = snr == s

        mse_g = np.mean(gain_err[mask]**2)
        mse_p = np.mean(phase_err[mask]**2)
        crb_g = np.mean(bcrb_g[mask])
        crb_p = np.mean(bcrb_p[mask])

        ratio_g = mse_g / crb_g if crb_g > 1e-15 else float('inf')
        ratio_p = mse_p / crb_p if crb_p > 1e-15 else float('inf')

        gap_g_db = 10 * np.log10(ratio_g) if ratio_g > 0 else 0
        gap_p_db = 10 * np.log10(ratio_p) if ratio_p > 0 else 0

        gain_ratios.append(ratio_g)
        phase_ratios.append(ratio_p)

        print(f"{s:<8} | {ratio_g:<12.2f}x | {ratio_p:<12.2f}x | {gap_g_db:<14.2f} | {gap_p_db:<14.2f}")

    # ========================================================================
    # FACTOR 3: Parameter-wise Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("FACTOR 3: PARAMETER-WISE ANALYSIS (at SNR=10 dB)")
    print("="*80)
    print("\nSome parameters may be harder to estimate than others.")
    print("\n" + "-"*80)

    snr_ref = 10 if 10 in snr_levels else snr_levels[len(snr_levels)//2]
    mask_ref = snr == snr_ref

    print(f"\nGain parameters at SNR={snr_ref} dB:")
    print(f"{'Param':<10} | {'MSE':<12} | {'CRB':<12} | {'Ratio':<10} | {'Bias':<12}")
    print("-"*60)

    for i in range(M-1):
        param_err = gain_err[mask_ref, i]
        param_mse = np.mean(param_err**2)
        param_crb = np.mean(bcrb_g[mask_ref, i])
        param_bias = np.mean(param_err)
        ratio = param_mse / param_crb if param_crb > 1e-15 else float('inf')
        print(f"ψ_{i+2:<7} | {param_mse:<12.2e} | {param_crb:<12.2e} | {ratio:<10.2f}x | {param_bias:<12.2e}")

    print(f"\nPhase parameters at SNR={snr_ref} dB:")
    print(f"{'Param':<10} | {'MSE':<12} | {'CRB':<12} | {'Ratio':<10} | {'Bias':<12}")
    print("-"*60)

    for i in range(M-2):
        param_err = phase_err[mask_ref, i]
        param_mse = np.mean(param_err**2)
        param_crb = np.mean(bcrb_p[mask_ref, i])
        param_bias = np.mean(param_err)
        ratio = param_mse / param_crb if param_crb > 1e-15 else float('inf')
        print(f"φ_{i+3:<7} | {param_mse:<12.2e} | {param_crb:<12.2e} | {ratio:<10.2f}x | {param_bias:<12.2e}")

    # ========================================================================
    # FACTOR 4: Error Distribution Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("FACTOR 4: ERROR DISTRIBUTION ANALYSIS")
    print("="*80)
    print("\nFor Gaussian errors, the MLE achieves the CRB asymptotically.")
    print("Non-Gaussian errors suggest model limitations or data issues.")

    # Check normality via kurtosis (manual computation to avoid scipy dependency)
    def compute_kurtosis(x):
        """Compute excess kurtosis (Gaussian = 0, pearson = 3)"""
        x = x - np.mean(x)
        m2 = np.mean(x**2)
        m4 = np.mean(x**4)
        return m4 / (m2**2) if m2 > 0 else 0  # Pearson kurtosis

    print(f"\nKurtosis of errors (Gaussian = 3.0):")
    print(f"{'SNR':<8} | {'Gain Kurtosis':<15} | {'Phase Kurtosis':<15}")
    print("-"*45)

    for s in snr_levels[::2]:  # Every other SNR for brevity
        mask = snr == s
        g_kurt = compute_kurtosis(gain_err[mask].flatten())
        p_kurt = compute_kurtosis(phase_err[mask].flatten())
        print(f"{s:<8} | {g_kurt:<15.2f} | {p_kurt:<15.2f}")

    # ========================================================================
    # FACTOR 5: Training vs Test Distribution
    # ========================================================================
    print("\n" + "="*80)
    print("FACTOR 5: TRAINING VS TEST DISTRIBUTION MISMATCH")
    print("="*80)

    print("""
The neural network is typically trained on:
  - RANDOM (ψ, φ) values from a prior distribution
  - Various SNR levels

But the CRB is computed at:
  - FIXED (ψ, φ) values
  - Specific SNR levels

This mismatch means:
1. The network learns an AVERAGE estimator across all (ψ, φ)
2. The CRB is specific to the TEST parameter values
3. The network cannot specialize to the specific test case

IMPLICATION: The network MSE is an average over parameter space,
while CRB is computed at specific fixed parameters. The network
trades off optimality at specific points for robustness across
the entire parameter space.
""")

    # ========================================================================
    # FACTOR 6: Fundamental Limits
    # ========================================================================
    print("="*80)
    print("FACTOR 6: FUNDAMENTAL THEORETICAL CONSIDERATIONS")
    print("="*80)

    print("""
THEORETICAL REASONS WHY NN MAY NOT REACH CRB:

1. CRB ASSUMPTIONS:
   - CRB assumes unbiased estimator
   - CRB assumes infinite samples (asymptotic)
   - CRB assumes exact model knowledge

2. NEURAL NETWORK CHARACTERISTICS:
   - Learns MMSE estimator (can be biased for better MSE at low SNR)
   - Works with finite training data
   - Approximates function through finite-capacity network
   - Generalizes across parameter space (not specialized)

3. SAMPLE COVARIANCE INPUT:
   - Network receives R̂ (sample covariance), not true R
   - R̂ has estimation error: Var(R̂) = R²/T + ...
   - This is a MISMATCHED bound problem

4. ACHIEVABILITY:
   - CRB is achievable by MLE only asymptotically (T → ∞)
   - For finite T, even MLE may not reach CRB
   - The paper's Fisher Scoring Algorithm is iterative MLE
   - NN is a different estimator class altogether
""")

    # ========================================================================
    # Summary and Recommendations
    # ========================================================================
    print("="*80)
    print("SUMMARY: WHY THE GAP EXISTS")
    print("="*80)

    avg_gain_ratio = np.mean(gain_ratios)
    avg_phase_ratio = np.mean(phase_ratios)
    avg_bias_gain = np.mean(bias_contribution_gain) * 100
    avg_bias_phase = np.mean(bias_contribution_phase) * 100

    print(f"""
KEY FINDINGS:

1. AVERAGE MSE/CRB RATIO:
   - Gain:  {avg_gain_ratio:.2f}x above CRB ({10*np.log10(avg_gain_ratio):.1f} dB gap)
   - Phase: {avg_phase_ratio:.2f}x above CRB ({10*np.log10(avg_phase_ratio):.1f} dB gap)

2. BIAS CONTRIBUTION TO MSE:
   - Gain:  {avg_bias_gain:.1f}% of MSE is due to bias
   - Phase: {avg_bias_phase:.1f}% of MSE is due to bias

3. MAIN CONTRIBUTORS TO THE GAP:
""")

    if avg_bias_gain > 10:
        print("   [!] Significant BIAS in gain estimation - network is not unbiased")
    else:
        print("   [✓] Low bias - network is approximately unbiased for gains")

    if avg_bias_phase > 10:
        print("   [!] Significant BIAS in phase estimation - network is not unbiased")
    else:
        print("   [✓] Low bias - network is approximately unbiased for phases")

    print("""
4. RECOMMENDATIONS TO CLOSE THE GAP:
   a) Train on fixed (ψ, φ) matching test conditions
   b) Increase model capacity / training data
   c) Use hybrid approach: NN initialization + Fisher Scoring refinement
   d) Implement per-SNR specialized networks
   e) Add regularization toward unbiased estimates
""")

    # ========================================================================
    # Create visualization
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: MSE/CRB ratio vs SNR
    ax = axes[0, 0]
    ax.plot(snr_levels, gain_ratios, 'o-', label='Gain', linewidth=2, markersize=8)
    ax.plot(snr_levels, phase_ratios, 's-', label='Phase', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Efficient (CRB)')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE / CRB Ratio', fontsize=12)
    ax.set_title('Efficiency Analysis: How far from CRB?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.set_yscale('log')

    # Plot 2: Bias contribution
    ax = axes[0, 1]
    ax.bar(np.array(snr_levels) - 1, [b*100 for b in bias_contribution_gain],
           width=2, label='Gain Bias²/MSE', alpha=0.7)
    ax.bar(np.array(snr_levels) + 1, [b*100 for b in bias_contribution_phase],
           width=2, label='Phase Bias²/MSE', alpha=0.7)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bias² contribution to MSE (%)', fontsize=12)
    ax.set_title('Bias-Variance Decomposition', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4, axis='y')

    # Plot 3: Error histogram at mid-SNR
    ax = axes[1, 0]
    mask_mid = snr == snr_ref
    ax.hist(gain_err[mask_mid].flatten(), bins=50, alpha=0.7, label='Gain errors', density=True)
    ax.hist(phase_err[mask_mid].flatten(), bins=50, alpha=0.7, label='Phase errors', density=True)
    ax.set_xlabel('Error', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Error Distribution at SNR={snr_ref} dB', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)

    # Plot 4: MSE and CRB vs SNR
    ax = axes[1, 1]
    mse_g_per_snr = [np.mean(gain_err[snr==s]**2) for s in snr_levels]
    mse_p_per_snr = [np.mean(phase_err[snr==s]**2) for s in snr_levels]
    crb_g_per_snr = [np.mean(bcrb_g[snr==s]) for s in snr_levels]
    crb_p_per_snr = [np.mean(bcrb_p[snr==s]) for s in snr_levels]

    ax.semilogy(snr_levels, mse_g_per_snr, 'o-', label='NN Gain MSE', linewidth=2)
    ax.semilogy(snr_levels, crb_g_per_snr, 'o--', label='Gain CRB', linewidth=2)
    ax.semilogy(snr_levels, mse_p_per_snr, 's-', label='NN Phase MSE', linewidth=2)
    ax.semilogy(snr_levels, crb_p_per_snr, 's--', label='Phase CRB', linewidth=2)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE / CRB', fontsize=12)
    ax.set_title('MSE vs CRB Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.4)

    plt.tight_layout()
    plt.savefig('crb_gap_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Analysis plots saved to: crb_gap_analysis.png")
    plt.show()

    return {
        'snr_levels': snr_levels,
        'gain_ratios': gain_ratios,
        'phase_ratios': phase_ratios,
        'bias_contribution_gain': bias_contribution_gain,
        'bias_contribution_phase': bias_contribution_phase
    }


if __name__ == "__main__":
    M = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model_path = Path("calibnet_M6_improved.pt")
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("[INFO] Please train a model first")
        exit(1)

    model = CalibNetDeep(M, p_drop=0.0).to(device)  # No dropout during eval
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[INFO] Loaded model from: {model_path}")

    # Find test data
    test_paths = [
        "Datasets/test_full_snr_bcrb.npz",
        "Datasets/test_crb.npz"
    ]

    test_path = None
    for p in test_paths:
        if Path(p).exists():
            test_path = p
            break

    if test_path is None:
        print("[ERROR] No test dataset found!")
        print("[INFO] Please run dataset_generation.py or dataset_generation_crb.py first")
        exit(1)

    print(f"[INFO] Using test data: {test_path}")

    results = analyze_crb_gap(model, test_path, M, device=device)
