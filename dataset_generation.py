import numpy as np

# =========================
# Samplers & steering
# =========================
def sample_trunc_laplace(size, rng=np.random, b=1.0, upper=2.0):
    """Truncated one-sided Laplace == truncated exponential(scale=b) on [0,upper]."""
    u = rng.rand(*size)
    c = 1.0 - np.exp(-upper / b)
    return -b * np.log(1.0 - u * c)

def steering_matrix(M, D, kl, theta):
    m = np.arange(M)[:, None]
    return np.exp(1j * kl * m * np.cos(theta))  # (M,D)

def build_C_from_angles(M, D, kl, theta, sigma_s_sq):
    A = steering_matrix(M, D, kl, theta)
    return A @ np.diag(sigma_s_sq) @ A.conj().T

def generate_sample_cov(M, D, T, kl, Psi, Phi, theta,
                        sigma_s_squared, sigma_n_squared):
    """Returns sample covariance (2-channel real/imag for X downstream)."""
    Psi_mat = np.diag(Psi)
    Phi_mat = np.diag(np.exp(1j * Phi))
    A       = steering_matrix(M, D, kl, theta)

    rMat = np.zeros((M, T), dtype=np.complex128)
    for t in range(T):
        s       = np.sqrt(sigma_s_squared / 2) * (np.random.randn(D) + 1j * np.random.randn(D))
        rSignal = Psi_mat @ Phi_mat @ A @ s
        noise   = np.sqrt(sigma_n_squared / 2) * (np.random.randn(M) + 1j * np.random.randn(M))
        rMat[:, t] = rSignal + noise
    return (rMat @ rMat.conj().T) / T

# =========================
# FIM & BCRB (your routine ported)
# =========================
def symmetric_toeplitz_mask(M, k1_based):
    """Ones on |i-j| = k-1. k1_based ∈ {1..M} (MATLAB style)."""
    k0 = k1_based - 1
    mask = np.zeros((M, M), dtype=np.float64)
    if k0 == 0:
        np.fill_diagonal(mask, 1.0)
    else:
        i = np.arange(M - k0)
        mask[i, i + k0] = 1.0
        mask[i + k0, i] = 1.0
    return mask

def fischer_with_gradients_2(C, Phi, Psi, sigma_n_squared, T_snapshots=1):
    """
    Python port of your MATLAB 'fischer_with_gradients_2'.
    Param order (total 4M-3):
      ψ(2..M) | φ(3..M) | ρ(1..M) | ι(2..M) | σ^2
    Returns real, symmetric FIM (scaled by T) and grad dict.
    """
    M = len(Psi)
    I_M = np.eye(M, dtype=np.complex128)

    Dpsi = np.diag(Psi)
    Eip  = np.diag(np.exp(1j * Phi))
    Ein  = np.diag(np.exp(-1j * Phi))

    C_phi_theta = Eip @ C @ Ein
    G_psi_theta = Dpsi @ C @ Dpsi

    R = Dpsi @ Eip @ C @ Ein @ Dpsi + sigma_n_squared * I_M
    inv_R = np.linalg.inv(R)

    # grads
    grad = {"Psi":[None]*(M-1), "Phi":[None]*(M-2), "Rho":[None]*M, "Io":[None]*(M-1), "Sigma":I_M}

    def Emat(m):
        e = np.zeros(M); e[m] = 1.0
        return np.diag(e)

    # dR/dψ(m), m=2..M
    for m1_ind in range(M-1):
        m1 = m1_ind + 1  # 0-based index for sensor m=2..M
        E = Emat(m1)
        dR = Dpsi @ C_phi_theta @ E
        dR = dR + dR.conj().T
        grad["Psi"][m1_ind] = dR

    # dR/dφ(m), m=3..M
    for m1_ind in range(M-2):
        m1 = m1_ind + 2
        E = Emat(m1)
        term1 = np.exp(1j * Phi[m1]) * (E @ G_psi_theta @ Ein)
        term2 = (np.diag(np.exp(1j*Phi)) @ G_psi_theta @ E) * np.exp(-1j * Phi[m1])
        dR = 1j * (term1 - term2)
        grad["Phi"][m1_ind] = dR

    # dR/dρ(k), k=1..M
    for k in range(1, M+1):
        mask = symmetric_toeplitz_mask(M, k)
        dR = np.zeros((M, M), dtype=np.complex128)
        i, j = np.nonzero(mask)
        dR[i, j] = Psi[i]*Psi[j]*np.exp(1j*(Phi[i]-Phi[j]))
        grad["Rho"][k-1] = dR

    # dR/dι(k), k=2..M
    for k_ind in range(M-1):
        k = k_ind + 2
        mask = symmetric_toeplitz_mask(M, k)
        dR = np.zeros((M, M), dtype=np.complex128)
        i, j = np.nonzero(mask)
        sgn = np.sign(j - i)  # +1 for i<j, -1 for i>j
        dR[i, j] = 1j * sgn * Psi[i]*Psi[j]*np.exp(1j*(Phi[i]-Phi[j]))
        grad["Io"][k_ind] = dR

    # index slices (0-based)
    sl_psi  = slice(0, M-1)
    sl_phi  = slice(M-1, (M-1)+(M-2))
    sl_rho  = slice(sl_phi.stop, sl_phi.stop + M)
    sl_iota = slice(sl_rho.stop, sl_rho.stop + (M-1))
    idx_sigma = sl_iota.stop  # last index

    K = 4*M - 3
    assert idx_sigma == K-1

    FIM = np.zeros((K, K), dtype=np.complex128)
    def tp(A, B): return np.trace(inv_R @ A @ inv_R @ B)

    # fill blocks
    # (Ψ,Ψ)
    for i in range(M-1):
        d1 = grad["Psi"][i]
        for j in range(i+1):
            d2 = grad["Psi"][j]
            val = tp(d1, d2)
            FIM[sl_psi.start+i, sl_psi.start+j] = val
            FIM[sl_psi.start+j, sl_psi.start+i] = val
    # (Ψ,Φ)
    for i in range(M-1):
        dpsi = grad["Psi"][i]
        for j in range(M-2):
            dphi = grad["Phi"][j]
            val = tp(dpsi, dphi)
            FIM[sl_psi.start+i, sl_phi.start+j] = val
            FIM[sl_phi.start+j, sl_psi.start+i] = val
    # (Φ,Φ)
    for i in range(M-2):
        d1 = grad["Phi"][i]
        for j in range(i+1):
            d2 = grad["Phi"][j]
            val = tp(d1, d2)
            FIM[sl_phi.start+i, sl_phi.start+j] = val
            FIM[sl_phi.start+j, sl_phi.start+i] = val
    # (ρ,ρ)
    for i in range(M):
        d1 = grad["Rho"][i]
        for j in range(i+1):
            d2 = grad["Rho"][j]
            val = tp(d1, d2)
            FIM[sl_rho.start+i, sl_rho.start+j] = val
            FIM[sl_rho.start+j, sl_rho.start+i] = val
    # (ι,ι)
    for i in range(M-1):
        d1 = grad["Io"][i]
        for j in range(i+1):
            d2 = grad["Io"][j]
            val = tp(d1, d2)
            FIM[sl_iota.start+i, sl_iota.start+j] = val
            FIM[sl_iota.start+j, sl_iota.start+i] = val
    # (Ψ,ρ)
    for i in range(M-1):
        dpsi = grad["Psi"][i]
        for j in range(M):
            drho = grad["Rho"][j]
            val = tp(dpsi, drho)
            FIM[sl_psi.start+i, sl_rho.start+j] = val
            FIM[sl_rho.start+j, sl_psi.start+i] = val
    # (Ψ,ι)
    for i in range(M-1):
        dpsi = grad["Psi"][i]
        for j in range(M-1):
            dio = grad["Io"][j]
            val = tp(dpsi, dio)
            FIM[sl_psi.start+i, sl_iota.start+j] = val
            FIM[sl_iota.start+j, sl_psi.start+i] = val
    # (Φ,ρ)
    for i in range(M-2):
        dphi = grad["Phi"][i]
        for j in range(M):
            drho = grad["Rho"][j]
            val = tp(dphi, drho)
            FIM[sl_phi.start+i, sl_rho.start+j] = val
            FIM[sl_rho.start+j, sl_phi.start+i] = val
    # (Φ,ι)
    for i in range(M-2):
        dphi = grad["Phi"][i]
        for j in range(M-1):
            dio = grad["Io"][j]
            val = tp(dphi, dio)
            FIM[sl_phi.start+i, sl_iota.start+j] = val
            FIM[sl_iota.start+j, sl_phi.start+i] = val
    # σ cross
    dS = grad["Sigma"]
    for i in range(M-1):
        val = tp(grad["Psi"][i], dS)
        FIM[sl_psi.start+i, idx_sigma] = val; FIM[idx_sigma, sl_psi.start+i] = val
    for i in range(M-2):
        val = tp(grad["Phi"][i], dS)
        FIM[sl_phi.start+i, idx_sigma] = val; FIM[idx_sigma, sl_phi.start+i] = val
    for i in range(M):
        val = tp(grad["Rho"][i], dS)
        FIM[sl_rho.start+i, idx_sigma] = val; FIM[idx_sigma, sl_rho.start+i] = val
    for i in range(M-1):
        val = tp(grad["Io"][i], dS)
        FIM[sl_iota.start+i, idx_sigma] = val; FIM[idx_sigma, sl_iota.start+i] = val
    FIM[idx_sigma, idx_sigma] = tp(dS, dS)

    # T snapshots + real symmetrize
    FIM = T_snapshots * np.real((FIM + FIM.T) / 2.0)
    return FIM, {"slices": (sl_psi, sl_phi)}

def bcrb_from_model(M, D, T, kl, Psi, Phi, theta, sigma_s_sq, sigma_n_sq,
                    b_scale=1.0):
    """
    Build C, run your FIM, add Bayesian prior on ψ(2..M) (truncated exp scale=b),
    invert, and return BCRB diagonals for ψ(2..M) and φ(3..M).
    """
    C = build_C_from_angles(M, D, kl, theta, sigma_s_sq)

    # slight jitter for numerical stability
    # (handled inside inverse via FIM; C is well-conditioned in most cases)

    FIM, sl = fischer_with_gradients_2(C, Phi, Psi, sigma_n_sq, T_snapshots=T)
    K = FIM.shape[0]
    I_prior = np.zeros((K, K), dtype=np.float64)

    # prior on ψ block only
    sl_psi, sl_phi = sl["slices"]
    I_prior[np.arange(sl_psi.start, sl_psi.stop), np.arange(sl_psi.start, sl_psi.stop)] = 1.0 / (b_scale**2)

    J_post = FIM + I_prior
    try:
        C_post = np.linalg.inv(J_post)
    except np.linalg.LinAlgError:
        C_post = np.linalg.pinv(J_post, rcond=1e-12)

    bcrb_psi = np.diag(C_post[sl_psi, sl_psi]).astype(float)
    bcrb_phi = np.diag(C_post[sl_phi, sl_phi]).astype(float)
    return bcrb_psi, bcrb_phi

# =========================
# Dataset builders
# =========================
def single_sample(M=6, D=3, T=100, kl=np.pi,
                  rng=np.random, theta_deg=None,
                  snr_db=None):
    """Returns (X=(2,M,M), y=(2M,), snr_db, Psi, Phi, theta)."""
    Psi = sample_trunc_laplace(size=(M,), rng=rng, b=1.0, upper=2.0); Psi[0] = 1.0
    Phi = rng.uniform(-np.pi/2, np.pi/2, size=M); Phi[:2] = 0.0

    if theta_deg is None:
        theta_deg = rng.uniform(-80, 80, size=D)
    else:
        if np.isscalar(theta_deg):
            theta_deg = np.full(D, float(theta_deg))
        else:
            theta_deg = np.asarray(theta_deg, dtype=float); assert theta_deg.shape == (D,)
    theta = np.deg2rad(theta_deg)

    sigma_s_sq = np.ones(D)
    sigma_n_sq = 10.0 ** (-snr_db / 10.0)

    # sample covariance for network input
    R_hat = generate_sample_cov(M, D, T, kl, Psi, Phi, theta, sigma_s_sq, sigma_n_sq)
    X = np.stack([np.real(R_hat), np.imag(R_hat)], axis=0)
    y = np.concatenate([Psi, Phi])
    return X, y, int(snr_db), Psi, Phi, theta, sigma_s_sq, sigma_n_sq

def generate_train_dataset(num_samples=40_000, M=6, D=3, T=100, kl=np.pi,
                           snr_db_levels=(0,5,10,15,20),
                           theta_fixed=None,
                           save_path="train_random_angles.npz",
                           rng_seed=123):
    """Train set: SNR ∈ {0,5,10,15,20}, no BCRB stored (compute saved)."""
    rng = np.random.RandomState(rng_seed)
    Xs, ys, snrs = [], [], []
    for _ in range(num_samples):
        snr_db = int(rng.choice(snr_db_levels))
        X, y, snr, *_ = single_sample(M, D, T, kl, rng, theta_fixed, snr_db)
        Xs.append(X); ys.append(y); snrs.append(snr)
    X_arr = np.stack(Xs, axis=0)
    y_arr = np.stack(ys, axis=0)
    snr_arr = np.asarray(snrs, dtype=np.int16)
    if save_path:
        np.savez_compressed(save_path, X=X_arr, y=y_arr, snr_db=snr_arr)
    return X_arr, y_arr, snr_arr

def generate_test_dataset_balanced_bcrb(num_per_snr=2000, M=6, D=3, T=100, kl=np.pi,
                                        snr_db_levels=tuple(range(-20, 21, 5)),
                                        theta_fixed=10.0,
                                        b_scale=1.0,
                                        save_path="test_balanced_bcrb.npz",
                                        rng_seed=321):
    """
    Test set: balanced SNR bins in {-20,-15,...,20}, and store BCRB(ψ,φ).
    Angles fixed by default (theta_fixed=10°) for consistent evaluation.
    """
    rng = np.random.RandomState(rng_seed)
    Xs, ys, snrs, bcrb_g_list, bcrb_p_list = [], [], [], [], []

    for snr_db in snr_db_levels:
        for _ in range(num_per_snr):
            X, y, snr, Psi, Phi, theta, sigma_s_sq, sigma_n_sq = single_sample(
                M, D, T, kl, rng, theta_fixed, snr_db
            )
            # compute BCRB for this specific configuration and SNR
            bcrb_g, bcrb_p = bcrb_from_model(
                M, D, T, kl, Psi, Phi, theta, sigma_s_sq, sigma_n_sq, b_scale=b_scale
            )
            Xs.append(X); ys.append(y); snrs.append(snr)
            bcrb_g_list.append(bcrb_g); bcrb_p_list.append(bcrb_p)

    # stack & shuffle
    X_arr   = np.stack(Xs, axis=0)
    y_arr   = np.stack(ys, axis=0)
    snr_arr = np.asarray(snrs, dtype=np.int16)
    bcrb_g  = np.stack(bcrb_g_list, axis=0)  # (N, M-1)
    bcrb_p  = np.stack(bcrb_p_list, axis=0)  # (N, M-2)

    idx = rng.permutation(X_arr.shape[0])
    X_arr, y_arr, snr_arr, bcrb_g, bcrb_p = X_arr[idx], y_arr[idx], snr_arr[idx], bcrb_g[idx], bcrb_p[idx]

    if save_path:
        np.savez_compressed(save_path,
                            X=X_arr, y=y_arr, snr_db=snr_arr,
                            bcrb_gain=bcrb_g, bcrb_phase=bcrb_p)
    return X_arr, y_arr, snr_arr, bcrb_g, bcrb_p

# =========================
# Example calls
# =========================
# TRAIN: SNR in {0,5,10,15,20}, random angles by default
high_train_levels = (25, 30)
high_test_levels  = (-10,-5,0,5,10,15, 20, 25, 30)

# Train: high SNR only (keep angles random to avoid memorizing a fixed geometry)
generate_train_dataset(
    num_samples=40_000, M=6, D=3, T=3000, kl=np.pi,
    snr_db_levels=high_train_levels,
    theta_fixed=10.0,                 # random angles
    save_path="Datasets/train_high_snr.npz",
    rng_seed=123
)

# Test: high SNR only (keep fixed angle for controlled eval, or set None to stress generalization)
generate_test_dataset_balanced_bcrb(
    num_per_snr=1, M=6, D=3, T=3000, kl=np.pi,
    snr_db_levels=high_test_levels,   # restrict to high SNR
    theta_fixed=8.0,                 # fixed angle protocol as before
    b_scale=1.0,
    save_path="Datasets/test_high_snr_bcrb.npz",
    rng_seed=321
)

