#!/usr/bin/env python3
"""
Verify BCRB computation correctness.
Compares stored BCRB values in dataset against freshly computed values.
"""

import numpy as np
from pathlib import Path
from dataset_generation import bcrb_from_model, build_C_from_angles, steering_matrix


def verify_bcrb_dataset(npz_path, M=6, D=3, T=3000, kl=np.pi, b_scale=1.0,
                        num_samples=10, verbose=True):
    """
    Verify BCRB values stored in dataset by recomputing a subset.

    Args:
        npz_path: Path to test dataset with BCRB labels
        M: Number of array elements
        D: Number of sources
        T: Number of snapshots
        kl: Wavenumber * element spacing
        b_scale: Prior scale for truncated Laplace
        num_samples: Number of samples to verify
        verbose: Print detailed output
    """
    npz_path = Path(npz_path)
    print(f"\n{'='*60}")
    print(f"BCRB VERIFICATION")
    print(f"{'='*60}")
    print(f"Dataset: {npz_path}")

    with np.load(npz_path) as data:
        y = data["y"]
        snr_db = data["snr_db"]
        bcrb_gain_stored = data["bcrb_gain"]
        bcrb_phase_stored = data["bcrb_phase"]

    print(f"Total samples: {len(y)}")
    print(f"BCRB gain shape: {bcrb_gain_stored.shape} (M-1 = {M-1} gains)")
    print(f"BCRB phase shape: {bcrb_phase_stored.shape} (M-2 = {M-2} phases)")
    print(f"\nVerifying {num_samples} random samples...")

    # Random indices to verify
    rng = np.random.RandomState(42)
    indices = rng.choice(len(y), size=min(num_samples, len(y)), replace=False)

    errors_gain = []
    errors_phase = []

    for idx in indices:
        # Extract parameters from y
        Psi = y[idx, :M]
        Phi = y[idx, M:]
        snr = snr_db[idx]

        # Fixed theta (as used in test set generation)
        theta = np.deg2rad(np.array([8.0] * D))
        sigma_s_sq = np.ones(D)
        sigma_n_sq = 10.0 ** (-snr / 10.0)

        # Recompute BCRB
        bcrb_g_computed, bcrb_p_computed = bcrb_from_model(
            M, D, T, kl, Psi, Phi, theta, sigma_s_sq, sigma_n_sq, b_scale=b_scale
        )

        # Compare with stored values
        bcrb_g_stored = bcrb_gain_stored[idx]
        bcrb_p_stored = bcrb_phase_stored[idx]

        err_g = np.abs(bcrb_g_computed - bcrb_g_stored).max()
        err_p = np.abs(bcrb_p_computed - bcrb_p_stored).max()

        errors_gain.append(err_g)
        errors_phase.append(err_p)

        if verbose:
            print(f"\nSample {idx} (SNR={snr} dB):")
            print(f"  Gain BCRB stored:   {bcrb_g_stored}")
            print(f"  Gain BCRB computed: {bcrb_g_computed}")
            print(f"  Max error (gain):   {err_g:.2e}")
            print(f"  Phase BCRB stored:  {bcrb_p_stored}")
            print(f"  Phase BCRB computed:{bcrb_p_computed}")
            print(f"  Max error (phase):  {err_p:.2e}")

    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Samples verified: {len(indices)}")
    print(f"Max gain BCRB error:  {max(errors_gain):.2e}")
    print(f"Max phase BCRB error: {max(errors_phase):.2e}")

    threshold = 1e-10
    if max(errors_gain) < threshold and max(errors_phase) < threshold:
        print(f"\n[PASS] All BCRB values match within tolerance {threshold}")
        return True
    else:
        print(f"\n[WARN] Some BCRB values differ beyond tolerance {threshold}")
        return False


def analyze_bcrb_vs_snr(npz_path, M=6):
    """
    Analyze how BCRB varies with SNR.
    """
    npz_path = Path(npz_path)
    print(f"\n{'='*60}")
    print(f"BCRB vs SNR ANALYSIS")
    print(f"{'='*60}")

    with np.load(npz_path) as data:
        snr_db = data["snr_db"]
        bcrb_gain = data["bcrb_gain"]
        bcrb_phase = data["bcrb_phase"]

    snr_levels = sorted(np.unique(snr_db))

    print(f"\n{'SNR (dB)':<10} {'Mean BCRB Gain':<18} {'Mean BCRB Phase':<18}")
    print(f"{'-'*50}")

    for snr in snr_levels:
        mask = snr_db == snr
        mean_g = bcrb_gain[mask].mean()
        mean_p = bcrb_phase[mask].mean()
        print(f"{snr:<10} {mean_g:<18.6e} {mean_p:<18.6e}")

    # Check expected behavior: BCRB should decrease with increasing SNR
    mean_bcrb_g = [bcrb_gain[snr_db == s].mean() for s in snr_levels]
    mean_bcrb_p = [bcrb_phase[snr_db == s].mean() for s in snr_levels]

    print(f"\n{'='*60}")
    print(f"SANITY CHECKS")
    print(f"{'='*60}")

    # Check monotonicity
    gain_decreasing = all(mean_bcrb_g[i] >= mean_bcrb_g[i+1] for i in range(len(mean_bcrb_g)-1))
    phase_decreasing = all(mean_bcrb_p[i] >= mean_bcrb_p[i+1] for i in range(len(mean_bcrb_p)-1))

    print(f"Gain BCRB decreasing with SNR:  {'[PASS]' if gain_decreasing else '[FAIL]'}")
    print(f"Phase BCRB decreasing with SNR: {'[PASS]' if phase_decreasing else '[FAIL]'}")

    # Check positivity
    all_positive_g = (bcrb_gain > 0).all()
    all_positive_p = (bcrb_phase > 0).all()

    print(f"All gain BCRB positive:  {'[PASS]' if all_positive_g else '[FAIL]'}")
    print(f"All phase BCRB positive: {'[PASS]' if all_positive_p else '[FAIL]'}")

    return {
        'snr_levels': snr_levels,
        'mean_bcrb_gain': mean_bcrb_g,
        'mean_bcrb_phase': mean_bcrb_p
    }


if __name__ == "__main__":
    test_path = Path("Datasets/test_full_snr_bcrb.npz")

    if not test_path.exists():
        print(f"[ERROR] Test dataset not found: {test_path}")
        print("[INFO]  Please run dataset_generation.py first")
        exit(1)

    # Verify BCRB computation
    verify_bcrb_dataset(test_path, num_samples=5, verbose=True)

    # Analyze BCRB vs SNR
    analyze_bcrb_vs_snr(test_path)
