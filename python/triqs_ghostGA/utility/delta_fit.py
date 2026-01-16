import numpy as np
from scipy.optimize import minimize, root, least_squares, brentq

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
    #return np.where(
    #    beta * eps > 0,
    #    np.exp(-beta * eps) / (1 + np.exp(-beta * eps)),
    #    1 / (1 + np.exp(beta * eps))
    #)
    f = []
    for e in eps:
        if e*beta<500:
                f.append(1./(1+np.exp(e*beta)))
        elif e*beta< -500:
            f.append(1)
        elif e*beta> 500:
            f.append(0)
    return np.array(f)

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
    If R_target is perturbed by dR, then H12 changes by dR @ D.T
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
def residual(x, beta, Lambda_c, D, F22_target, RTF12_target):
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
    #print("in residual:",np.sum(np.abs(res)))
    return res

def residual_minimize(x, beta, Lambda_c, D, F22_target, RTF12_target):
    r = residual(x, beta, Lambda_c, D, F22_target, RTF12_target)
    return 0.5 * np.dot(r, r)

def residual_11(x, beta, Lambda_c, D, F11_target, RTF12_target):
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta)

    F11 = F[:n, :n]
    F12 = F[:n, n:]
    
    RTF12 = R.T@F12

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # Standard Physical Residual
    res = np.concatenate([
        (F11.real[iu] - F11_target.real[iu]),
        (F11.imag[iu_strict] - F11_target.imag[iu_strict]),
        (RTF12.real - RTF12_target.real).ravel(),
        (RTF12.imag - RTF12_target.imag).ravel()
    ])
    #print("in residual:",np.sum(np.abs(res)))
    return res


# ============================================================
# Jacobian via Frechet derivative
# ============================================================
def jacobian(x, beta, Lambda_c, D, F22_target, RTF12_target):
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

    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = 1.0

        dLambda, dR = unpack_params(dx, n, p)
        dH = dH_dLambda(dLambda, n) + dH_dR(dR, D)
        dF = dF_spectral(H, dH, beta)

        dF22 = dF[n:, n:]
        dF12 = dF[:n, n:]
        #  d(R.T@F12) with respect to R is
        #  R.T @ dF12 + d(R.T) @ F12
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

def jacobian_minimize(x, beta, Lambda_c, D, F22_target, RTF12_target):
    r = residual(x, beta, Lambda_c, D, F22_target, RTF12_target)
    J = jacobian(x, beta, Lambda_c, D, F22_target, RTF12_target)
    return J.T @ r

def jacobian_11(x, beta, Lambda_c, D, F11_target, RTF12_target):
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

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

        dLambda, dR = unpack_params(dx, n, p)
        dH = dH_dLambda(dLambda, n) + dH_dR(dR, D)
        dF = dF_spectral(H, dH, beta)

        dF11 = dF[:n, :n]
        dF12 = dF[:n, n:]
        #  d(R.T@F12) with respect to R is
        #  R.T @ dF12 + d(R.T) @ F12
        dRTF12 = R.T@dF12 +dR.T@F12
        

        # Must match the concatenation logic in residual()
        Jcols.append(np.concatenate([
            # F11 (Hermitian reduction)
            dF11.real[iu],
            dF11.imag[iu_strict],
            # RTF12 (full)
            dRTF12.real.ravel(),
            dRTF12.imag.ravel()
        ]))

    return np.column_stack(Jcols)


# ============================================================
# Root solve
# ============================================================
# This one works without derivatives and with least square instead of root
def solve_F_only(beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target):
    x0 = pack_params(Lambda0, R0)

    # We use least_squares because it supports 'trf' and '2-point' (numerical) jacobians
    sol = least_squares(
        residual,
        x0,
        jac='2-point', # This tells Scipy to compute the gradient numerically
        args=(beta, Lambda_c, D, F22_target, RTF12_target),
        method="trf",
        max_nfev=200,  # Limits total function calls to 200
        xtol=1e-14,
        ftol=1e-14,
        verbose=2      # Useful to see if the cost function is actually decreasing
    )
    
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol


# This one works
def solve_F_dF(beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target):
    x0 = pack_params(Lambda0, R0)
    res = residual(x0, beta, Lambda_c, D, F22_target, RTF12_target)
    if(False):
        sol = least_squares(
            residual,
            x0,
            jac=jacobian,
            args=(beta, Lambda_c, D, F22_target, RTF12_target),
            method="trf", # Change from 'hybr' to 'lm'
            ftol=1e-14,
            xtol=1e-14,
            verbose=2
        )
    elif(True):
        sol = root(
            residual,
            x0,
            jac=jacobian,
            args=(beta, Lambda_c, D, F22_target, RTF12_target),
            method="lm", # Change from 'hybr' to 'lm'
            options={'ftol': 1e-14, 'xtol': 1e-14}
        )
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

def solve_F_dF_minimize(beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target):
    x0 = pack_params(Lambda0, R0)
    res = residual(x0, beta, Lambda_c, D, F22_target, RTF12_target)
    sol = minimize(
        residual_minimize,
        x0,
        #jac=jacobian_minimize,
        args=(beta, Lambda_c, D, F22_target, RTF12_target),
        method="BFGS",
        tol=1e-5,
        options={'disp':False, 'eps': 1e-10, 'maxiter': len(x0)*10000}
        )
    print('sols.fun=',sol.fun)
    print('sols.message=',sol.message)
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

# This one works
def solve_F_dF_11(beta, Lambda_c, D, Lambda0, R0, F11_target, RTF12_target):
    x0 = pack_params(Lambda0, R0)
    sol = root(
        residual_11,
        x0,
        jac=jacobian_11,
        args=(beta, Lambda_c, D, F11_target, RTF12_target),
        method="lm", # Change from 'hybr' to 'lm'
        options={'ftol': 1e-14, 'xtol': 1e-14}
    )
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol
