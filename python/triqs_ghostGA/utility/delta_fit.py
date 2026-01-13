import numpy as np
from scipy.optimize import root, least_squares, brentq

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
# Get chemical potential
# -------------------------------------------------
def get_mu(eps, beta):
    # avoids overflow using where since else is zero
    def occupation(mu,eps,beta):
        return np.sum( fermi( eps-mu,beta))-len(eps)/2
    mu_min=np.min(eps)-10/beta
    mu_max=np.max(eps)+10/beta

    return brentq( occupation, mu_min, mu_max, args=(eps,beta) )



# -------------------------------------------------
# F(H) via diagonalization
# -------------------------------------------------
def F_of_H(H, beta):
    eps, U = np.linalg.eigh(H)
    mu   = get_mu(eps, beta)
    eps -= mu
    f = fermi(eps, beta)
    return (U * f) @ U.conj().T

# -------------------------------------------------
# Frechet derivative dF(H)[dH]
# -------------------------------------------------
def dF_spectral(H, dH, beta, tol=1e-14):
    eps, U = np.linalg.eigh(H)
    mu   = get_mu(eps, beta)
    eps -= mu

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
    Lambda.real[iu] = Lam_re
    Lambda.real[(iu[1], iu[0])] = Lam_re
    Lambda.imag[iu_strict] = Lam_im
    Lambda.imag[iu_strict[1],iu_strict[0]] = -Lam_im
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
    F = F_of_H(H, beta)

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
    F = F_of_H(H, beta)

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
    F = F_of_H(H, beta)

    F22 = F[n:, n:]
    F12 = F[:n, n:]

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

        dF22 = dF[n:, n:]
        dF12 = dF[:n, n:]
        dRTF12 = R.T@dF12 +dR.T@F12
        
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
    F = F_of_H(H, beta)

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
        dF = dF_spectral(H, dH, beta)

        dF11 = dF[:n, :n]
        dF12 = dF[:n, n:]
        dF12D = dF12@D +F12@dD
        
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


# ============================================================
# Root solve for Lambda and R (aka Self Energy)
# ============================================================
# This one works without derivatives and with least square instead of root
def solve_F_only_LR(beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target):
    '''
    Solve the root finding problem for a given set of Lambda_c,D,beta,F22=<badg b> and RTF12=R.T@<fdag b>
    using numerical derivatives
    '''
    x0 = pack_params(Lambda0, R0)
    # We use least_squares because it supports 'trf' and '2-point' (numerical) jacobians
    sol = least_squares(
        residual_LR,
        x0,
        jac='2-point', # This tells Scipy to compute the gradient numerically
        args=(beta, Lambda_c, D, F22_target, RTF12_target),
        method="trf",
        max_nfev=100,  # Limits total function calls to 200
        xtol=1e-9,
        ftol=1e-9,
        verbose=2      # Useful to see if the cost function is actually decreasing
    )
    
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol


# This one works
def solve_F_dF_LR(beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target):
    '''
    Solve the root finding problem for Lambda and R
    given set of Lambda_c,D,beta,F22=<badg b> and RTF12=R.T@<fdag b>
    using analytical derivatives.
    Least_squares() proved to be faster than root()
    '''
    x0 = pack_params(Lambda0, R0)
    if(True):
        sol = least_squares(
            residual_LR,
            x0,
            jac=jacobian_LR,
            args=(beta, Lambda_c, D, F22_target, RTF12_target),
            method="trf", # Change from 'hybr' to 'lm'
            max_nfev=100,
            ftol=1e-9,
            xtol=1e-9,
            verbose=2
        )
    elif(False):
        sol = root(
            residual_LR,
            x0,
            jac=jacobian_LR,
            args=(beta, Lambda_c, D, F22_target, RTF12_target),
            method="lm", # Change from 'hybr' to 'lm'
            options={'ftol': 1e-9, 'xtol': 1e-9}
        )
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

def new_self_energy( Lambda0,R0, Lambda_c,D, F22_target,RTF12_target, beta=200, method="dF"):
    print("New self-energy fitting Lambda and R")
    if(method=="dF"):
        res, new_Lambda, new_R = solve_F_dF_LR(beta, Lambda_c,D,Lambda0,R0,F22_target,RTF12_target)
    elif(method=="F"):
        res, new_Lambda, new_R = solve_F_only_LR(beta, Lambda_c,D,Lambda0,R0,F22_target,RTF12_target)
    else:
        raise ValueError(f"Tried new_self_energy with method={method} - only \"F\" and \"dF\" methods are available")
    return new_Lambda, new_R

# ============================================================
# Root solve for Lambda_c and D (aka Hybridization)
# ============================================================
# This one works without derivatives and with least square instead of root
def solve_F_only_LcD(beta, Lambda, R, Lambda_c0, D0, F11_target, F12D_target):
    x0 = pack_params(Lambda_c0, D0)
    # We use least_squares because it supports 'trf' and '2-point' (numerical) jacobians
    sol = least_squares(
        residual_LcD,
        x0,
        jac='2-point', # This tells Scipy to compute the gradient numerically
        args=(beta, Lambda, R, F11_target, F12D_target),
        method="trf",
        max_nfev=100,  # Limits total function calls to 200
        xtol=1e-9,
        ftol=1e-9,
        verbose=2      # Useful to see if the cost function is actually decreasing
    )
    Lambda_c_sol, D_sol = unpack_params(sol.x, D0.shape[0], D0.shape[1])
    return sol, Lambda_c_sol, D_sol


# This one works
def solve_F_dF_LcD(beta, Lambda, R, Lambda_c0, D0, F11_target, F12D_target):
    '''
    Solve the root finding problem for Lambda_c and D
    given set of Lambda,R,beta,F11=<fadg f> and RTF12=<fdag b>@D
    using analytical derivatives.
    Least_squares() proved to be faster than root()
    '''
    x0 = pack_params(Lambda_c0, D0)
    if(True):
        sol = least_squares(
            residual_LcD,
            x0,
            jac=jacobian_LcD,
            args=(beta, Lambda, R, F11_target, F12D_target),
            method="trf", # Change from 'hybr' to 'lm'
            max_nfev=100,
            ftol=1e-9,
            xtol=1e-9,
            verbose=2
        )
    elif(False):
        sol = root(
            residual_LcD,
            x0,
            jac=jacobian_LcD,
            args=(beta, Lambda, R, F11_target, F12D_target),
            method="lm", # Change from 'hybr' to 'lm'
            options={'ftol': 1e-9, 'xtol': 1e-9}
        )
    Lambda_c_sol, D_sol = unpack_params(sol.x, D0.shape[0], D0.shape[1])
    return sol, Lambda_c_sol, D_sol


def new_hybridization( Lambda_c0,D0, Lambda,R, F11_target, F12D_target, beta=200, method="dF"):
    print("New hybridization fitting Lambda_c and D")
    if(method=="dF"):
        res, new_Lambda_c, new_D = solve_F_dF_LcD(beta, Lambda,R,Lambda_c0,D0,F11_target,F12D_target)
    elif(method=="F"):
        res, new_Lambda_c, new_D = solve_F_only_LcD(beta, Lambda,R,Lambda_c0,D0,F11_target,F12D_target)
    else:
        raise ValueError(f"Tried new_hybridization with method={method} - only \"F\" and \"dF\" methods are available")
    
    return new_Lambda_c, new_D
    
