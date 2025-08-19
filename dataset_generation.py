import numpy as np

# -------------------------------------------------
# helper: truncated one-sided Laplace sampler (b = 1, clip at 2)
# -------------------------------------------------
def sample_trunc_laplace(size, rng=np.random, b=1.0, upper=2.0):
    u = rng.rand(*size)
    c = 1.0 - np.exp(-upper / b)          # normalising constant
    return -b * np.log(1.0 - u * c)       # inverse CDF


# -------------------------------------------------
# 1) covariance generator (unchanged)
# -------------------------------------------------
def generate_sample_cov(M, D, T, kl, Psi, Phi, theta,
                        sigma_s_squared, sigma_n_squared):
    Psi_mat = np.diag(Psi)
    Phi_mat = np.diag(np.exp(1j * Phi))
    m_idx   = np.arange(M)[:, None]
    A       = np.exp(1j * kl * m_idx * np.cos(theta))

    rMat = np.zeros((M, T), dtype=np.complex128)
    for t in range(T):
        s       = np.sqrt(sigma_s_squared / 2) * \
                  (np.random.randn(D) + 1j * np.random.randn(D))
        rSignal = Psi_mat @ Phi_mat @ A @ s
        noise   = np.sqrt(sigma_n_squared / 2) * \
                  (np.random.randn(M) + 1j * np.random.randn(M))
        rMat[:, t] = rSignal + noise

    return (rMat @ rMat.conj().T) / T


# -------------------------------------------------
# 2) one (X, y, snr, bcrb_gain, bcrb_phase) sample
# -------------------------------------------------
def single_sample_psi_phi(M=5, D=3, T=100, kl=np.pi,
                          theta_deg_range=(-80, 80),
                          snr_db_levels=(0, 5, 10, 15, 20)):
    rng = np.random

    # --- gains (Laplace, σ=1, clipped to 2) --------------------------
    Psi      = sample_trunc_laplace(size=(M,), rng=rng, b=1.0, upper=2.0)
    Psi[0]   = 1.0                           # reference gain

    # --- phases (uniform [-π/2, π/2]) -------------------------------
    Phi      = rng.uniform(-np.pi/2, np.pi/2, size=M)
    Phi[:2]  = 0.0                           # φ₁ = φ₂ = 0

    # --- scene & SNR -------------------------------------------------
    theta        = np.deg2rad(
                     rng.uniform(*theta_deg_range, size=D))
    sigma_s_sq   = np.ones(D)
    snr_db       = rng.choice(snr_db_levels)
    sigma_n_sq   = 10 ** (-snr_db / 10)

    # --- sample covariance ------------------------------------------
    R_hat = generate_sample_cov(M, D, T, kl, Psi, Phi, theta,
                                sigma_s_sq, sigma_n_sq)

    # --- plug-in hybrid CRB diagonals -------------------------------
    J_gain    = 2 * D / sigma_n_sq
    crb_gain  = 1.0 / (1.0 + J_gain)                 # shape (M−1,)
    crb_phase = sigma_n_sq / (2 * D * Psi[2:]**2)    # shape (M−2,)

    # --- package -----------------------------------------------------
    X = np.stack([np.real(R_hat), np.imag(R_hat)], axis=0)  # (2,M,M)
    y = np.concatenate([Psi, Phi])                          # (2M,)
    return X, y, snr_db, crb_gain, crb_phase


# -------------------------------------------------
# 3) full dataset builder
# -------------------------------------------------
def generate_dataset_psi_phi(num_samples=10_000, M=5, D=3, T=100,
                             save_path=None, **kwargs):
    Xs, ys, snrs, crb_gs, crb_ps = zip(
        *(single_sample_psi_phi(M=M, D=D, T=T, **kwargs)
          for _ in range(num_samples)))

    X_arr   = np.stack(Xs,   axis=0)               # (N, 2, M, M)
    y_arr   = np.stack(ys,   axis=0)               # (N, 2M)
    snr_arr = np.asarray(snrs,  dtype=np.int16)    # (N,)
    crb_g   = np.stack(crb_gs, axis=0)             # (N, M−1)
    crb_p   = np.stack(crb_ps, axis=0)             # (N, M−2)

    if save_path:
        np.savez_compressed(save_path,
                            X=X_arr,
                            y=y_arr,
                            snr=snr_arr,
                            bcrb_gain=crb_g,
                            bcrb_phase=crb_p)
    return X_arr, y_arr, snr_arr, crb_g, crb_p
    
generate_dataset_psi_phi(10000, M=6, save_path="psi_phi_snr_bcrb.npz")

