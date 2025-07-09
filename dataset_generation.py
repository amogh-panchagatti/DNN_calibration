import numpy as np

# -------------------------------------------------
# 1) covariance generator (unchanged)
# -------------------------------------------------
def generate_sample_cov(M, D, T, kl, Psi, Phi, theta,
                        sigma_s_squared, sigma_n_squared):
    Psi_mat = np.diag(Psi)
    Phi_mat = np.diag(np.exp(1j * Phi))
    m_idx = np.arange(M)[:, None]
    A = np.exp(1j * kl * m_idx * np.cos(theta))

    rMat = np.zeros((M, T), dtype=np.complex128)
    for t in range(T):
        s = np.sqrt(sigma_s_squared / 2) * (np.random.randn(D) + 1j * np.random.randn(D))
        rSignal = Psi_mat @ Phi_mat @ A @ s
        noise   = np.sqrt(sigma_n_squared / 2) * (np.random.randn(M) + 1j * np.random.randn(M))
        rMat[:, t] = rSignal + noise

    return (rMat @ rMat.conj().T) / T      # sample covariance R̂


# -------------------------------------------------
# 2) one (X, y) pair  –  y = [Ψ ‖ Φ]
# -------------------------------------------------
def single_sample_psi_phi(M=5, D=3, T=100, kl=np.pi,
                          psi_range=(0.7, 2.2),
                          phi_deg_range=(-10, 10),
                          theta_deg_range=(-80, 80),
                          snr_db_range=(0, 20)):
    # --- random gains & phases -------------------------------------------
    Psi        = np.random.uniform(*psi_range, size=M)
    Psi[0]     = 1.0                          # reference gain
    phi_deg    = np.random.uniform(*phi_deg_range, size=M)
    phi_deg[:2] = 0.0                         # Φ₁ = Φ₂ = 0
    Phi        = np.deg2rad(phi_deg)

    # --- scene parameters -------------------------------------------------
    theta      = np.deg2rad(np.random.uniform(*theta_deg_range, size=D))
    sigma_s_sq = np.ones(D)                   # unit source power
    snr_db     = np.random.uniform(*snr_db_range)
    sigma_n_sq = 10 ** (-snr_db / 10)

    # --- covariance -------------------------------------------------------
    R_hat = generate_sample_cov(M, D, T, kl, Psi, Phi, theta,
                                sigma_s_sq, sigma_n_sq)

    # labels: concatenate Ψ and Φ  → length 2M
    y = np.concatenate([Psi, Phi])

    # input: two-channel real/imag of R̂
    X = np.stack([np.real(R_hat), np.imag(R_hat)], axis=0)   # (2, M, M)
    return X, y


# -------------------------------------------------
# 3) full dataset builder  –  saves .npz if asked
# -------------------------------------------------
def generate_dataset_psi_phi(num_samples=10_000, M=5, D=3, T=100,
                             save_path=None, **kwargs):
    Xs, ys = zip(*(single_sample_psi_phi(M=M, D=D, T=T, **kwargs)
                   for _ in range(num_samples)))
    X_arr = np.stack(Xs, axis=0)   # (N, 2, M, M)
    y_arr = np.stack(ys,  axis=0)  # (N, 2M)
    if save_path:
        np.savez_compressed(save_path, X=X_arr, y=y_arr)
    return X_arr, y_arr
