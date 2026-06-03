#######################################################
# Finite temperature density matrix fitting routines
# for thermal ghost Gutzwiller
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
#######################################################

import numpy as np
from scipy.optimize import root, least_squares, brentq, minimize


# -------------------------
# Build H
# -------------------------
def build_H(Lambda, Lambda_c, D, R):
    H11 = Lambda
    H12 = R @ D.T
    H21 = H12.T.conj()
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
def dF_spectral(H, dH, beta, tol=1e-10):
    eps, U = np.linalg.eigh(H)
    f = fermi(eps, beta)
    fp = fermi_prime(eps, beta)

    # matrix G_ij
    eps_i = eps[:, None]
    eps_j = eps[None, :]

    G = 0.5*(fp.reshape((1,-1)) + fp.reshape((-1,1)) )
    mask = np.abs(eps_i - eps_j) > tol
    # when not degenerate
    G[mask] = (f[:, None] - f[None, :])[mask] / (eps_i - eps_j)[mask]

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
    If R is perturbed by dR, then H12 changes by dR @ D.T
    Because H is Hermitian, H21 must change by (dR @ D.T).conj().T
    """
    n, p = D.shape
    Z_nn = np.zeros((n, n), dtype=complex)

    term12 = dR @ D.T
    term21 = term12.conj().T

    return np.block([
        [Z_nn,   term12],
        [term21, Z_nn  ]
    ])

def dH_dLambdac(dLambdac, n):
    Z = np.zeros((n, n), dtype=complex)
    return np.block([[Z, Z],
                     [Z, -dLambdac]])

def dH_dD(dD, R):
    """
    dD is the perturbation to D.
    If D is perturbed by dD, then H12 changes by R @ dD.T
    Because H is Hermitian, H21 must change by (R @ dD.T).conj().T
    """
    n, p = R.shape
    Z_nn = np.zeros((n, n), dtype=complex)

    term12 = R @ dD.T
    term21 = term12.conj().T

    return np.block([
        [Z_nn,   term12],
        [term21, Z_nn  ]
    ])


# ============================================================
# Packing / unpacking parameters
# It works for both Lambda+R and Lambda_c+D
# ============================================================
def pack_params(Lambda, R):
    n = R.shape[0]
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
    Lambda.real[iu] = Lam_re.real
    Lambda.real[(iu[1], iu[0])] = Lam_re.real
    Lambda.imag[iu_strict] = Lam_im.real
    Lambda.imag[iu_strict[1],iu_strict[0]] = -Lam_im.real
    R = R_re + 1j * R_im
    return Lambda, R


# ============================================================
# Residual
# ============================================================
def residual_LR(x, beta, Lambda_c, D, F22_target, RTF12_target):
    '''
    Residual for Lambda+R fit.
    '''
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T

    F22 = F[n:, n:]
    F12 = F[:n, n:]
    RTF12 = R.T@F12

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # Standard Physical Residual
    res = np.concatenate([
        (F22.real[iu] - F22_target.real[iu]),
        (F22.imag[iu_strict] - F22_target.imag[iu_strict]),
        (RTF12.real - RTF12_target.real).ravel(),
        (RTF12.imag - RTF12_target.imag).ravel()
    ])
    return res

def residual_LcD(x, beta, Lambda, R, F11_target, F12D_target):
    '''
    Residual for Lambda_c+D fit
    '''
    n, p = R.shape
    Lambda_c, D = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T

    F11 = F[:n, :n]
    F12 = F[:n, n:]
    F12D = F12@D

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # Standard Physical Residual
    res = np.concatenate([
        (F11.real[iu] - F11_target.real[iu]),
        (F11.imag[iu_strict] - F11_target.imag[iu_strict]),
        (F12D.real - F12D_target.real).ravel(),
        (F12D.imag - F12D_target.imag).ravel()
    ])
    return res


# ============================================================
# Jacobian via Frechet derivative
# ============================================================
def jacobian_LR(x, beta, Lambda_c, D, F22_target, RTF12_target):
    '''
    Jacobian for Lambda+R fit
    '''
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T

    F22 = F[n:, n:]
    F12 = F[:n, n:]

    Jcols = []

    # Indices for Hermitian reduction
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # dLambda and dR
    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = 1.0

        dLambda, dR = unpack_params(dx, n, p)
        dH = dH_dLambda(dLambda, n) + dH_dR(dR, D)
        dF = dF_spectral(H, dH, beta).T

        dF22 = dF[n:, n:]
        dF12 = dF[:n, n:]
        dRTF12 = R.T@dF12 + dR.T@F12

        # Must match the concatenation logic in residual()
        Jcols.append(np.concatenate([
            # F22 (Hermitian reduction)
            dF22.real[iu],
            dF22.imag[iu_strict],
            # RTF12 (full)
            dRTF12.real.ravel(),
            dRTF12.imag.ravel()
        ]))

    return np.column_stack(Jcols)

def jacobian_LcD(x, beta, Lambda, R, F11_target, F12D_target):
    '''
    Jacobian for Lambda_c+D
    '''
    n, p = R.shape
    Lambda_c, D = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T

    F11 = F[:n, :n]
    F12 = F[:n, n:]

    Jcols = []

    # Indices for Hermitian reduction
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = 1.0

        dLambda_c, dD = unpack_params(dx, n, p)
        dH = dH_dLambdac(dLambda_c, n) + dH_dD(dD, R)
        dF = dF_spectral(H, dH, beta).T

        dF11 = dF[:n, :n]
        dF12 = dF[:n, n:]
        dF12D = dF12@D + F12@dD

        # Must match the concatenation logic in residual()
        Jcols.append(np.concatenate([
            # F22 (Hermitian reduction)
            dF11.real[iu],
            dF11.imag[iu_strict],
            # F12D (full)
            dF12D.real.ravel(),
            dF12D.imag.ravel()
        ]))

    return np.column_stack(Jcols)


# ---------- Movement-penalized residuals / jacobians for Lambda+R -----------
def residual_LR_movement(x, beta, Lambda_c, D, F22_target, RTF12_target, x0, alpha=1e-5):
    """Original residuals + movement-penalty residuals sqrt(2*alpha)*(x-x0)."""
    r = residual_LR(x, beta, Lambda_c, D, F22_target, RTF12_target)
    # movement residuals
    if alpha <= 0.0:
        return r
    factor = np.sqrt(2.0 * alpha)
    r_move = factor * (x - x0)
    return np.concatenate([r, r_move])

def jacobian_LR_movement(x, beta, Lambda_c, D, F22_target, RTF12_target, x0, alpha=1e-5):
    """Stack original Jacobian with movement-penalty rows: factor * I."""
    J = jacobian_LR(x, beta, Lambda_c, D, F22_target, RTF12_target)  # shape (m, N)
    if alpha <= 0.0:
        return J
    N = x.size
    factor = np.sqrt(2.0 * alpha)
    # create extra rows: factor * identity (N rows x N cols)
    Jpen = factor * np.eye(N, dtype=float)
    # stack (original m x N) with (N x N) => (m+N, N)
    return np.vstack([J, Jpen])

def solve_F_dF_LR_with_movement(beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target,
                                alpha=1e-5, use_analytic_jac=True, max_nfev=200, verbose=0):
    """
    Least-squares solve with quadratic movement penalty around starting x0.
    alpha: penalty strength for sum_j (x_j-x0_j)^2 in objective.
    If use_analytic_jac True, provides jacobian via jacobian_LR_movement to least_squares.
    """
    x0 = pack_params(Lambda0, R0)
    if use_analytic_jac:
        sol = least_squares(
            residual_LR_movement,
            x0,
            jac=jacobian_LR_movement,
            args=(beta, Lambda_c, D, F22_target, RTF12_target, x0, alpha),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            verbose=verbose
        )
    else:
        sol = least_squares(
            residual_LR_movement,
            x0,
            jac='2-point',
            args=(beta, Lambda_c, D, F22_target, RTF12_target, x0, alpha),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            verbose=verbose
        )
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

def update_self_energy_thermal_penalty(Lambda0, R0, Lambda_c, D, F22_target, RTF12_target,
                             beta=200, alpha=1e-5, method="dF", verbose=0):
    """
    Replacement of new_self_energy that includes movement penalty alpha * ||x-x0||^2.
    method is passed to choose between analytic/numeric jacobian inside the solver.
    verbose: passed to least_squares (0=silent, 1=final, 2=per-iteration).
    """
    sol, Lambda_sol, R_sol = solve_F_dF_LR_with_movement(
        beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target,
        alpha=alpha, use_analytic_jac=(method=="dF"), verbose=verbose
    )
    return Lambda_sol, R_sol


# ---------- Movement-penalized residuals / jacobians for Lambda_c + D -----------
def residual_LcD_movement(x, beta, Lambda, R, F11_target, F12D_target, x0, alpha=1e-5):
    r = residual_LcD(x, beta, Lambda, R, F11_target, F12D_target)
    if alpha <= 0.0:
        return r
    factor = np.sqrt(2.0 * alpha)
    r_move = factor * (x - x0)
    return np.concatenate([r, r_move])

def jacobian_LcD_movement(x, beta, Lambda, R, F11_target, F12D_target, x0, alpha=1e-5):
    J = jacobian_LcD(x, beta, Lambda, R, F11_target, F12D_target)  # (m, N)
    if alpha <= 0.0:
        return J
    N = x.size
    factor = np.sqrt(2.0 * alpha)
    Jpen = factor * np.eye(N, dtype=float)
    return np.vstack([J, Jpen])

def solve_F_dF_LcD_with_movement(beta, Lambda, R, Lambda_c0, D0, F11_target, F12D_target,
                                 alpha=1e-5, use_analytic_jac=True, max_nfev=200, verbose=0):
    x0 = pack_params(Lambda_c0, D0)
    if use_analytic_jac:
        sol = least_squares(
            residual_LcD_movement,
            x0,
            jac=jacobian_LcD_movement,
            args=(beta, Lambda, R, F11_target, F12D_target, x0, alpha),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            verbose=verbose
        )
    else:
        sol = least_squares(
            residual_LcD_movement,
            x0,
            jac='2-point',
            args=(beta, Lambda, R, F11_target, F12D_target, x0, alpha),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            verbose=verbose
        )
    Lambda_c_sol, D_sol = unpack_params(sol.x, D0.shape[0], D0.shape[1])
    return sol, Lambda_c_sol, D_sol

def update_hybridization_thermal_penalty(Lambda_c0, D0, Lambda, R, F11_target, F12D_target,
                               beta=200, alpha=1e-5, method="dF", verbose=0):
    '''
    Function to update the hybridization function parameters D and Lambda_c.
    verbose: passed to least_squares (0=silent, 1=final, 2=per-iteration).
    '''
    sol, Lambda_c_sol, D_sol = solve_F_dF_LcD_with_movement(
        beta, Lambda, R, Lambda_c0, D0, F11_target, F12D_target,
        alpha=alpha, use_analytic_jac=(method=="dF"), verbose=verbose
    )
    return Lambda_c_sol, D_sol

