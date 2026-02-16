"""
Dataset generation with FIXED parameters for CRB computation.
This follows the EUSIPCO paper approach where CRB is computed at representative
parameter values, not averaged over random samples.

Key differences from dataset_generation.py:
1. CRB computed at FIXED typical (ψ, φ) values
2. Noise computed at reference SNR and rescaled per SNR bin
3. Files named with _crb suffix
"""

import numpy as np
from dataset_generation import (
    sample_trunc_laplace,
    steering_matrix,
    build_C_from_angles,
    generate_sample_cov,
    fischer_with_gradients_2
)

# =========================
# Fixed parameters for CRB (from EUSIPCO paper)
# =========================
def get_fixed_params(M):
    """
    Get fixed representative parameters for CRB computation.
    Based on EUSIPCO paper Section V.
    """
    # Base parameters (extended/truncated for different M)
    psi_base = np.array([1.0, 1.3, 1.1, 0.7, 2.2, 1.5, 0.9, 1.2, 1.4, 0.8])
    phi_base = np.deg2rad([0, 0, 5, 11, -8, 3, -5, 7, -3, 6])

    Psi_fixed = np.ones(M)
    Phi_fixed = np.zeros(M)

    # Fill with base values (repeat if needed)
    for i in range(M):
        Psi_fixed[i] = psi_base[i % len(psi_base)]
        Phi_fixed[i] = phi_base[i % len(phi_base)]

    # Enforce constraints
    Psi_fixed[0] = 1.0  # ψ₁ = 1 (reference)
    Phi_fixed[0] = 0.0  # φ₁ = 0
    Phi_fixed[1] = 0.0  # φ₂ = 0

    return Psi_fixed, Phi_fixed


def crb_from_fixed_params(M, D, T, kl, theta_deg, snr_db, b_scale=1.0):
    """
    Compute CRB at FIXED parameter values for a given SNR.

    This follows the EUSIPCO paper approach where CRB is evaluated
    at representative parameter values, not random ones.
    """
    Psi_fixed, Phi_fixed = get_fixed_params(M)
    theta = np.deg2rad(np.array([theta_deg] * D) if np.isscalar(theta_deg) else theta_deg)

    sigma_s_sq = np.ones(D)
    sigma_n_sq = 10.0 ** (-snr_db / 10.0)

    # Build C matrix
    C = build_C_from_angles(M, D, kl, theta, sigma_s_sq)

    # Compute FIM
    FIM, sl = fischer_with_gradients_2(C, Phi_fixed, Psi_fixed, sigma_n_sq, T_snapshots=T)
    K = FIM.shape[0]

    # Add prior on ψ block only (Bayesian)
    I_prior = np.zeros((K, K), dtype=np.float64)
    sl_psi, sl_phi = sl["slices"]
    I_prior[np.arange(sl_psi.start, sl_psi.stop),
            np.arange(sl_psi.start, sl_psi.stop)] = 1.0 / (b_scale**2)

    J_post = FIM + I_prior

    # Invert with regularization for numerical stability
    try:
        # Add small regularization for stability
        J_reg = J_post + 1e-10 * np.eye(K)
        C_post = np.linalg.inv(J_reg)
    except np.linalg.LinAlgError:
        C_post = np.linalg.pinv(J_post, rcond=1e-10)

    # Extract CRB for gains and phases
    crb_psi = np.diag(C_post[sl_psi, sl_psi]).astype(float)
    crb_phi = np.diag(C_post[sl_phi, sl_phi]).astype(float)

    # Ensure non-negative (numerical issues can cause tiny negative values)
    crb_psi = np.maximum(crb_psi, 1e-15)
    crb_phi = np.maximum(crb_phi, 1e-15)

    return crb_psi, crb_phi


def generate_sample_cov_rescaled(M, D, T, kl, Psi, Phi, theta,
                                  sigma_s_squared, base_noise_mat, noise_scale):
    """
    Generate sample covariance with pre-computed noise rescaled by SNR.

    Args:
        base_noise_mat: Pre-generated noise matrix at reference SNR
        noise_scale: Scale factor for this SNR level
    """
    Psi_mat = np.diag(Psi)
    Phi_mat = np.diag(np.exp(1j * Phi))
    A = steering_matrix(M, D, kl, theta)

    rMat = np.zeros((M, T), dtype=np.complex128)
    for t in range(T):
        s = np.sqrt(sigma_s_squared / 2) * (np.random.randn(D) + 1j * np.random.randn(D))
        rSignal = Psi_mat @ Phi_mat @ A @ s
        # Rescale pre-generated noise
        noise = base_noise_mat[:, t] * noise_scale
        rMat[:, t] = rSignal + noise

    return (rMat @ rMat.conj().T) / T


def single_sample_crb(M=6, D=3, T=100, kl=np.pi,
                      rng=np.random, theta_deg=None, snr_db=None,
                      base_noise_mat=None, ref_snr_db=0,
                      use_fixed_params=False):
    """
    Generate a single sample with noise rescaling.

    Args:
        base_noise_mat: Pre-generated noise at ref_snr_db
        ref_snr_db: Reference SNR for base noise
        use_fixed_params: If True, use fixed (ψ, φ) from get_fixed_params()
    """
    if use_fixed_params:
        # Use FIXED calibration parameters (for controlled CRB experiment)
        Psi, Phi = get_fixed_params(M)
    else:
        # Random calibration parameters (original behavior)
        Psi = sample_trunc_laplace(size=(M,), rng=rng, b=1.0, upper=2.0)
        Psi[0] = 1.0
        Phi = rng.uniform(-np.pi/2, np.pi/2, size=M)
        Phi[:2] = 0.0

    if theta_deg is None:
        theta_deg = rng.uniform(-80, 80, size=D)
    else:
        if np.isscalar(theta_deg):
            theta_deg = np.full(D, float(theta_deg))
        else:
            theta_deg = np.asarray(theta_deg, dtype=float)
    theta = np.deg2rad(theta_deg)

    sigma_s_sq = np.ones(D)
    sigma_n_sq = 10.0 ** (-snr_db / 10.0)

    # Compute noise scale relative to reference
    ref_sigma_n_sq = 10.0 ** (-ref_snr_db / 10.0)
    noise_scale = np.sqrt(sigma_n_sq / ref_sigma_n_sq)

    # Generate sample covariance
    if base_noise_mat is not None:
        R_hat = generate_sample_cov_rescaled(
            M, D, T, kl, Psi, Phi, theta, sigma_s_sq, base_noise_mat, noise_scale
        )
    else:
        R_hat = generate_sample_cov(M, D, T, kl, Psi, Phi, theta, sigma_s_sq, sigma_n_sq)

    X = np.stack([np.real(R_hat), np.imag(R_hat)], axis=0)
    y = np.concatenate([Psi, Phi])

    return X, y, int(snr_db), Psi, Phi, theta


def generate_train_dataset_crb(num_samples=40_000, M=6, D=3, T=100, kl=np.pi,
                                snr_db_levels=(0, 5, 10, 15, 20),
                                theta_fixed=None,
                                save_path="train_crb.npz",
                                rng_seed=123,
                                use_fixed_params=True):
    """
    Generate training dataset with FIXED calibration parameters.

    Args:
        use_fixed_params: If True (default), use fixed (ψ, φ) for all samples.
    """
    rng = np.random.RandomState(rng_seed)
    Xs, ys, snrs = [], [], []

    # Pre-generate base noise at reference SNR (0 dB)
    ref_snr_db = 0
    ref_sigma_n_sq = 10.0 ** (-ref_snr_db / 10.0)
    base_noise_mat = np.sqrt(ref_sigma_n_sq / 2) * (
        rng.randn(M, T) + 1j * rng.randn(M, T)
    )

    print(f"Generating {num_samples} training samples...")
    for i in range(num_samples):
        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i+1}/{num_samples}")

        snr_db = int(rng.choice(snr_db_levels))
        X, y, snr, *_ = single_sample_crb(
            M, D, T, kl, rng, theta_fixed, snr_db,
            base_noise_mat=base_noise_mat, ref_snr_db=ref_snr_db,
            use_fixed_params=use_fixed_params
        )
        Xs.append(X)
        ys.append(y)
        snrs.append(snr)

    X_arr = np.stack(Xs, axis=0)
    y_arr = np.stack(ys, axis=0)
    snr_arr = np.asarray(snrs, dtype=np.int16)

    if save_path:
        np.savez_compressed(save_path, X=X_arr, y=y_arr, snr_db=snr_arr)
        print(f"Saved training data to: {save_path}")

    return X_arr, y_arr, snr_arr


def generate_test_dataset_crb(num_per_snr=100, M=6, D=3, T=100, kl=np.pi,
                               snr_db_levels=tuple(range(-10, 31, 5)),
                               theta_fixed=8.0,
                               b_scale=1.0,
                               save_path="test_crb.npz",
                               rng_seed=321,
                               use_fixed_params=True):
    """
    Generate test dataset with FIXED calibration parameters and CRB.

    Key differences:
    1. CRB is computed once per SNR level at fixed parameters
    2. All samples use the same fixed (ψ, φ) values
    """
    rng = np.random.RandomState(rng_seed)

    # Pre-generate base noise
    ref_snr_db = 0
    ref_sigma_n_sq = 10.0 ** (-ref_snr_db / 10.0)
    base_noise_mat = np.sqrt(ref_sigma_n_sq / 2) * (
        rng.randn(M, T) + 1j * rng.randn(M, T)
    )

    # Compute CRB ONCE per SNR level at FIXED parameters
    print("Computing CRB at fixed parameters for each SNR level...")
    crb_per_snr = {}
    for snr_db in snr_db_levels:
        crb_psi, crb_phi = crb_from_fixed_params(
            M, D, T, kl, theta_fixed, snr_db, b_scale
        )
        crb_per_snr[snr_db] = (crb_psi, crb_phi)
        print(f"  SNR={snr_db:3d} dB: CRB_gain={crb_psi.mean():.2e}, CRB_phase={crb_phi.mean():.2e}")

    # Generate test samples
    print(f"\nGenerating {num_per_snr * len(snr_db_levels)} test samples...")
    Xs, ys, snrs = [], [], []
    crb_g_list, crb_p_list = [], []

    for snr_db in snr_db_levels:
        crb_psi, crb_phi = crb_per_snr[snr_db]

        for _ in range(num_per_snr):
            X, y, snr, *_ = single_sample_crb(
                M, D, T, kl, rng, theta_fixed, snr_db,
                base_noise_mat=base_noise_mat, ref_snr_db=ref_snr_db,
                use_fixed_params=use_fixed_params
            )
            Xs.append(X)
            ys.append(y)
            snrs.append(snr)
            # Same CRB for all samples at this SNR (fixed params)
            crb_g_list.append(crb_psi)
            crb_p_list.append(crb_phi)

    # Stack and shuffle
    X_arr = np.stack(Xs, axis=0)
    y_arr = np.stack(ys, axis=0)
    snr_arr = np.asarray(snrs, dtype=np.int16)
    crb_g = np.stack(crb_g_list, axis=0)
    crb_p = np.stack(crb_p_list, axis=0)

    idx = rng.permutation(X_arr.shape[0])
    X_arr = X_arr[idx]
    y_arr = y_arr[idx]
    snr_arr = snr_arr[idx]
    crb_g = crb_g[idx]
    crb_p = crb_p[idx]

    if save_path:
        np.savez_compressed(
            save_path,
            X=X_arr, y=y_arr, snr_db=snr_arr,
            crb_gain=crb_g, crb_phase=crb_p
        )
        print(f"Saved test data to: {save_path}")

    return X_arr, y_arr, snr_arr, crb_g, crb_p


# =========================
# Main execution
# =========================
if __name__ == "__main__":
    # Parameters matching the paper
    M = 6
    D = 3
    T = 2000
    kl = np.pi

    full_snr_levels = (-10, -5, 0, 5, 10, 15, 20, 25, 30)

    print("="*60)
    print("DATASET GENERATION WITH FIXED-PARAMETER CRB")
    print("="*60)
    print(f"M={M}, D={D}, T={T}")
    print(f"SNR levels: {full_snr_levels}")
    print()

    # Show fixed parameters used for CRB
    Psi_fixed, Phi_fixed = get_fixed_params(M)
    print("Fixed parameters for CRB computation:")
    print(f"  ψ_fixed = {Psi_fixed}")
    print(f"  φ_fixed = {np.rad2deg(Phi_fixed)} degrees")
    print()

    # Generate training data - RANDOM (ψ, φ) for learning
    print("-"*60)
    generate_train_dataset_crb(
        num_samples=10_000,
        M=M, D=D, T=T, kl=kl,
        snr_db_levels=full_snr_levels,
        theta_fixed=None,  # Random angles for training
        save_path="Datasets/train_crb.npz",
        rng_seed=123,
        use_fixed_params=False  # Random (ψ, φ) so model learns general estimator
    )

    # Generate test data with fixed-parameter CRB
    print("-"*60)
    generate_test_dataset_crb(
        num_per_snr=222,  # ~2000 total (222 * 9 SNR levels)
        M=M, D=D, T=T, kl=kl,
        snr_db_levels=full_snr_levels,
        theta_fixed=8.0,
        b_scale=1.0,
        save_path="Datasets/test_crb.npz",
        rng_seed=321
    )

    print()
    print("="*60)
    print("DONE")
    print("="*60)
