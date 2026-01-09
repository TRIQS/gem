import numpy as np
from scipy.optimize import root
from scipy.optimize import least_squares

# -------------------------
# Build H
# -------------------------
def build_H(Lambda, Lambda_c, D, R):
    H11 = Lambda
    H12 = D @ R.T
    H21 = R.conj() @ D.T.conj()
    H22 = -Lambda_c
    return np.block([[H11, H12],
                     [H21, H22]])

# -------------------------------------------------
# Scalar Fermi-like function (stable)
# -------------------------------------------------
def fermi(eps, beta):
    # avoids overflow using where since else is zero
    return np.where(
        beta * eps > 0,
        np.exp(-beta * eps) / (1 + np.exp(-beta * eps)),
        1 / (1 + np.exp(beta * eps))
    )

def fermi_prime(eps, beta):
    f = fermi(eps, beta)
    return -beta * f * (1 - f)

# -------------------------------------------------
# F(H) via diagonalization
# -------------------------------------------------
def F_of_H(H, beta):
    eps, U = np.linalg.eigh(H)
    f = fermi(eps, beta)
    return (U * f) @ U.conj().T

# -------------------------------------------------
# Frechet derivative dF(H)[dH]
# -------------------------------------------------
def dF_spectral(H, dH, beta, tol=1e-12):
    eps, U = np.linalg.eigh(H)

    f = fermi(eps, beta)
    fp = fermi_prime(eps, beta)

    # matrix G_ij
    eps_i = eps[:, None]
    eps_j = eps[None, :]

    G = np.zeros((len(eps), len(eps)))
    mask = np.abs(eps_i - eps_j) > tol

    G[mask] = (f[:, None] - f[None, :])[mask] / (eps_i - eps_j)[mask]
    np.fill_diagonal(G, fp)

    # rotate dH into eigenbasis
    dH_tilde = U.conj().T @ dH @ U

    # Hadamard product
    dF_tilde = G * dH_tilde

    return U @ dF_tilde @ U.conj().T

def dH_dLambda(dLambda, n):
    Z = np.zeros((n, n), dtype=complex)
    return np.block([[dLambda, Z],
                     [Z, Z]])

def dH_dR(dR, D):
    """
    dR is the perturbation to R.
    If R_target is perturbed by dR, then H12 changes by D @ dR.T.
    Because H is Hermitian, H21 must change by (D @ dR.T).conj().T
    """
    n, p = D.shape
    Z_nn = np.zeros((n, n), dtype=complex)
    
    term12 = D @ dR.T
    term21 = term12.conj().T 
    
    return np.block([
        [Z_nn,   term12],
        [term21, Z_nn  ]
    ])




# ============================================================
# Packing / unpacking parameters
# ============================================================
def pack_params(Lambda, R):
    n = Lambda.shape[0]
    p = R.shape[1]

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    Lam_re = Lambda.real[iu]
    Lam_im = Lambda.imag[iu_strict]

    return np.concatenate([
        Lam_re,
        Lam_im,
        R.real.reshape(-1),
        R.imag.reshape(-1)
    ])


def unpack_params(x, n, p):
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    n_re = len(iu[0])
    n_im = len(iu_strict[0])

    Lam_re = x[:n_re]
    Lam_im = x[n_re:n_re+n_im]

    offset = n_re + n_im

    R_re = x[offset:offset+n*p].reshape(n, p)
    offset += n*p
    R_im = x[offset:offset+n*p].reshape(n, p)

    # rebuild Hermitian Lambda
    Lambda = np.zeros((n, n), dtype=complex)
    Lambda.real[iu] = Lam_re
    Lambda.real[(iu[1], iu[0])] = Lam_re
    Lambda.imag[iu_strict] = Lam_im
    Lambda.imag[np.tril_indices(n, k=-1)] = -Lam_im

    R = R_re + 1j * R_im
    return Lambda, R


# ============================================================
# Residual
# ============================================================
def residual_old(x, beta, Lambda_c, D, F11_target, F12_target):
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta)

    F11 = F[:n, :n]
    F12 = F[:n, n:]

    return np.concatenate([
        (F11 - F11_target).real.ravel(),
        (F11 - F11_target).imag.ravel(),
        (F12 - F12_target).real.ravel(),
        (F12 - F12_target).imag.ravel()
    ])

def residual_good(x, beta, Lambda_c, D, F11_target, F12_target):
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta)

    F11 = F[:n, :n]
    F12 = F[:n, n:]

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    
    return np.concatenate([
        # F11 (Hermitian)
        (F11.real[iu] - F11_target.real[iu]),
        (F11.imag[iu_strict] - F11_target.imag[iu_strict]),

        # F12 (full)
        #(F12.real - F12_target.real).ravel(),
        #(F12.imag - F12_target.imag).ravel()
        # only p columns are independent
        (F12.real[:, :p] - F12_target.real[:, :p]).ravel(),
        (F12.imag[:, :p] - F12_target.imag[:, :p]).ravel()
    ])
def residual(x, beta, Lambda_c, D, F11_target, F12_target):
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta)

    F11 = F[:n, :n]
    F12 = F[:n, n:]

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # Standard Physical Residual
    res = np.concatenate([
        (F11.real[iu] - F11_target.real[iu]),
        (F11.imag[iu_strict] - F11_target.imag[iu_strict]),
        (F12.real[:, :p] - F12_target.real[:, :p]).ravel(),
        (F12.imag[:, :p] - F12_target.imag[:, :p]).ravel()
    ])

    return res


# ============================================================
# Jacobian via Frechet derivative
# ============================================================
def jacobian_full(x, beta, Lambda_c, D, F11_target, F12_target):
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    Jcols = []

    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = 1.0

        dLambda, dR = unpack_params(dx, n, p)
        dH = dH_dLambda(dLambda, n) + dH_dR(dR, D)
        dF = dF_spectral(H, dH, beta)

        dF11 = dF[:n, :n]
        dF12 = dF[:n, n:]

        Jcols.append(np.concatenate([
            dF11.real.ravel(),
            dF11.imag.ravel(),
            dF12.real.ravel(),
            dF12.imag.ravel()
        ]))

    return np.column_stack(Jcols)

# ============================================================
# Jacobian via Frechet derivative (Corrected)
# ============================================================
def jacobian(x, beta, Lambda_c, D, F11_target, F12_target):
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    Jcols = []
    
    # Indices for Hermitian reduction
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = 1.0

        dLambda, dR = unpack_params(dx, n, p)
        dH = dH_dLambda(dLambda, n) + dH_dR(dR, D)
        dF = dF_spectral(H, dH, beta)

        dF11 = dF[:n, :n]
        dF12 = dF[:n, n:]

        # Must match the concatenation logic in residual()
        Jcols.append(np.concatenate([
            # F11 (Hermitian reduction)
            dF11.real[iu],
            dF11.imag[iu_strict],

            # F12 (Reduced to first p columns)
            dF12.real[:, :p].ravel(),
            dF12.imag[:, :p].ravel()
        ]))

    return np.column_stack(Jcols)


# ============================================================
# Root solve
# ============================================================
def solve_F_old(beta, Lambda_c, D, Lambda0, R0, F11_target, F12_target):
    x0 = pack_params(Lambda0, R0)

    sol = root(
        residual,
        x0,
        jac=jacobian,
        args=(beta, Lambda_c, D, F11_target, F12_target),
        method="hybr"
    )

    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

def solve_F_dF(beta, Lambda_c, D, Lambda0, R0, F11_target, F12_target):
    x0 = pack_params(Lambda0, R0)

    sol = root(
        residual,
        x0,
        jac=jacobian,
        args=(beta, Lambda_c, D, F11_target, F12_target),
        method="lm", # Change from 'hybr' to 'lm'
        options={'ftol': 1e-10, 'xtol': 1e-10}
    )

    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol


def solve_F_3(beta, Lambda_c, D, Lambda0, R0, F11_target, F12_target):
    x0 = pack_params(Lambda0, R0)

    sol = root(
        residual,
        x0,
        jac=jacobian, #_vectorized,
        args=(beta, Lambda_c, D, F11_target, F12_target),
        method="trf", # More robust trust-region method
        options={'xtol': 1e-12, 'gtol': 1e-12}
    )
    
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

def solve_F(beta, Lambda_c, D, Lambda0, R0, F11_target, F12_target):
    x0 = pack_params(Lambda0, R0)

    # least_squares is generally more robust for this physics-based fitting
    sol = least_squares(
        residual,
        x0,
        jac=jacobian, # Use the jacobian function we fixed earlier
        args=(beta, Lambda_c, D, F11_target, F12_target),
        method="trf",  # Now this will work!
        xtol=1e-12,
        ftol=1e-12,
        verbose=2      # This will show you the convergence progress
    )

    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol


def solve_F_only(beta, Lambda_c, D, Lambda0, R0, F11_target, F12_target):
    x0 = pack_params(Lambda0, R0)

    # We use least_squares because it supports 'trf' and '2-point' (numerical) jacobians
    sol = least_squares(
        residual,
        x0,
        jac='2-point', # This tells Scipy to compute the gradient numerically
        args=(beta, Lambda_c, D, F11_target, F12_target),
        method="trf",
        max_nfev=200,  # Limits total function calls to 200
        xtol=1e-12,
        ftol=1e-12,
        verbose=2      # Useful to see if the cost function is actually decreasing
    )
    
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

