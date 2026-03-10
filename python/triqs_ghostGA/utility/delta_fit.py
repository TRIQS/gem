import numpy as np
from scipy.optimize import minimize, root, least_squares


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
    f=[]
    for xx in eps*beta:
        # This one is used to stablize selective Mott, but would lead to suprious OSMT if temperature is too high.
        #f.append(1./(1+np.exp(500*xx))) 
        # This one is important to get the correct phase diagram (especially for criyical t2/t1), but not stable in OSMP.
        if abs(xx)<500:
            f.append(1./(1+np.exp(xx)))
        elif xx< -500:
            f.append(1)
        elif xx> 500:
            f.append(0)
    return np.array(f)

# -------------------------------------------------
# F(H) via diagonalization
# -------------------------------------------------
def F_of_H(H, beta):
    eps, U = np.linalg.eigh(H)
    f = fermi(eps, beta)
    return (U * f) @ U.conj().T

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
        (RTF12.imag - RTF12_target.imag).ravel(),
    ])
    return res.real

def loewner_matrix(eps, beta):
    f = fermi(eps, beta)
    df = -beta * f * (1 - f)

    diff = eps[:, None] - eps[None, :]
    L = np.zeros_like(diff)

    mask = np.abs(diff) > 1e-12
    L[mask] = (f[:, None] - f[None, :])[mask] / diff[mask]
    np.fill_diagonal(L, df)

    return L

def residual_LcD(x, beta, Lambda, R, Delta_target, right_target):
    '''
    Residual for Lambda+R fit.
    '''
    n, p = R.shape
    Lambda_c, D = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T

    F11 = F[:n, :n]
    F12 = F[:n, n:]
    F12D = F12 @ D

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # Standard Physical Residual
    res = np.concatenate([
        (F11.real[iu] - Delta_target.real[iu]),
        (F11.imag[iu_strict] - Delta_target.imag[iu_strict]),
        (F12D.real - right_target.T.real).ravel(),
        (F12D.imag - right_target.T.imag).ravel(),
    ])
    return res.real

def dF_dH(H, dH, beta, eig_cache=None):
    if eig_cache is None:
        eps, U = np.linalg.eigh(H)
    else:
        eps, U = eig_cache
    L = loewner_matrix(eps, beta)

    UH_dH_U = U.conj().T @ dH @ U
    dF = U @ (L * UH_dH_U) @ U.conj().T
    return dF

def dH_dLambda(dLambda, n, p):
    Z = np.zeros((n, n), dtype=complex)
    return np.block([
        [dLambda, Z],
        [Z.T.conj(), np.zeros((n, n), dtype=complex)]
    ])

def dH_dLambda_c(dLambda_c, n, p):
    dH = np.zeros((2*n, 2*n), dtype=complex)
    dH[n:, n:] = -dLambda_c
    return dH

def dH_dR(dR, D, n, p):
    Znn = np.zeros((n, n), dtype=complex)
    return np.block([
        [Znn, dR @ D.T],
        [(dR @ D.T).T.conj(), Znn]
    ])

def dH_dD(dD, R, n, p):
    dH = np.zeros((2*n, 2*n), dtype=complex)
    dB = R @ dD.T
    dH[:n, n:] = dB
    dH[n:, :n] = dB.conj().T
    return dH

def jacobian_LR(x, beta, Lambda_c, D, F22_target, RTF12_target):
    n, p = D.shape

    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    eps, U = np.linalg.eigh(H)
    eig_cache = (eps, U)

    F = (U * fermi(eps, beta)) @ U.conj().T
    F = F.T

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # Number of rows in residual
    res_size = len(iu[0]) + len(iu_strict[0]) + 2 * n * p
    J = np.zeros((res_size, x.size))

    # Row offsets (fixed)
    row_F22_re = 0
    row_F22_im = len(iu[0])
    row_RTF12   = row_F22_im + len(iu_strict[0])

    col = 0

    # ----------------------------
    # Lambda real parameters (i <= j)
    # ----------------------------
    for i, j in zip(iu[0], iu[1]):

        dLam = np.zeros((n, n), dtype=complex)
        dLam[i, j] = 1.0
        dLam[j, i] = 1.0

        dH = dH_dLambda(dLam, n, p)
        dF = dF_dH(H, dH, beta, eig_cache).T

        # Fill J
        J[row_F22_re:row_F22_re + len(iu[0]), col] = dF[n:, n:].real[iu]
        J[row_F22_im:row_F22_im + len(iu_strict[0]), col] = dF[n:, n:].imag[iu_strict]

        dRTF12 = R.T @ dF[:n, n:]
        J[row_RTF12:row_RTF12 + n*p, col] = dRTF12.real.ravel()
        J[row_RTF12 + n*p:row_RTF12 + 2*n*p, col] = dRTF12.imag.ravel()

        col += 1

    # ----------------------------
    # Lambda imaginary parameters (i < j)
    # ----------------------------
    for i, j in zip(iu_strict[0], iu_strict[1]):

        dLam = np.zeros((n, n), dtype=complex)
        dLam[i, j] = 1.0j
        dLam[j, i] = -1.0j

        dH = dH_dLambda(dLam, n, p)
        dF = dF_dH(H, dH, beta, eig_cache).T

        # Fill J
        J[row_F22_re:row_F22_re + len(iu[0]), col] = dF[n:, n:].real[iu]
        J[row_F22_im:row_F22_im + len(iu_strict[0]), col] = dF[n:, n:].imag[iu_strict]

        dRTF12 = R.T @ dF[:n, n:]
        J[row_RTF12:row_RTF12 + n*p, col] = dRTF12.real.ravel()
        J[row_RTF12 + n*p:row_RTF12 + 2*n*p, col] = dRTF12.imag.ravel()

        col += 1

    # ----------------------------
    # R parameters (real + imag)
    # ----------------------------
    for part in ["real", "imag"]:
        for a in range(n):
            for b in range(p):
                dR = np.zeros((n, p), dtype=complex)
                dR[a, b] = 1.0 if part == "real" else 1.0j
                dH = dH_dR(dR, D, n, p)
                dF = dF_dH(H, dH, beta, eig_cache).T

                # Fill J
                J[row_F22_re:row_F22_re + len(iu[0]), col] = dF[n:, n:].real[iu]
                J[row_F22_im:row_F22_im + len(iu_strict[0]), col] = dF[n:, n:].imag[iu_strict]

                dRTF12 = R.T @ dF[:n, n:] + dR.T @ F[:n, n:]
                J[row_RTF12:row_RTF12 + n*p, col] = dRTF12.real.ravel()
                J[row_RTF12 + n*p:row_RTF12 + 2*n*p, col] = dRTF12.imag.ravel()

                col += 1

    return J

def jacobian_LcD(x, beta, Lambda, R, Delta_target, right_target):
    """
    Analytic Jacobian of residual_LcD using Loewner derivative.
    """

    n, p = R.shape

    # unpack parameters
    Lambda_c, D = unpack_params(x, n, p)

    # build Hamiltonian
    H = build_H(Lambda, Lambda_c, D, R)

    # eigendecomposition (cached)
    eps, U = np.linalg.eigh(H)
    eig_cache = (eps, U)

    # F(H)
    F = (U * fermi(eps, beta)) @ U.conj().T
    F = F.T

    F11 = F[:n, :n]
    F12 = F[:n, n:]

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # residual size
    res_size = (
        len(iu[0]) +           # Re(F11)
        len(iu_strict[0]) +    # Im(F11)
        2 * n * p              # Re/Im(F12 @ D)
    )

    J = np.zeros((res_size, x.size))
    col = 0

    # ============================================================
    # Lambda_c parameters
    # ============================================================

    # --- real symmetric part (i <= j) ---
    for i, j in zip(iu[0], iu[1]):

        dLc = np.zeros((n, n), dtype=complex)
        dLc[i, j] = 1.0
        dLc[j, i] = 1.0

        dH = dH_dLambda_c(dLc, n, p)
        dF = dF_dH(H, dH, beta, eig_cache).T

        dF11 = dF[:n, :n]
        dF12 = dF[:n, n:]

        row = 0
        J[row:row+len(iu[0]), col] = dF11.real[iu]
        row += len(iu[0])

        J[row:row+len(iu_strict[0]), col] = dF11.imag[iu_strict]
        row += len(iu_strict[0])

        dF12D = dF12 @ D
        J[row:row+n*p, col] = dF12D.real.ravel()
        row += n*p

        J[row:row+n*p, col] = dF12D.imag.ravel()

        col += 1

    # --- imaginary antisymmetric part (i < j) ---
    for i, j in zip(iu_strict[0], iu_strict[1]):

        dLc = np.zeros((n, n), dtype=complex)
        dLc[i, j] = 1.0j
        dLc[j, i] = -1.0j

        dH = dH_dLambda_c(dLc, n, p)
        dF = dF_dH(H, dH, beta, eig_cache).T

        dF11 = dF[:n, :n]
        dF12 = dF[:n, n:]

        row = 0
        J[row:row+len(iu[0]), col] = dF11.real[iu]
        row += len(iu[0])

        J[row:row+len(iu_strict[0]), col] = dF11.imag[iu_strict]
        row += len(iu_strict[0])

        dF12D = dF12 @ D
        J[row:row+n*p, col] = dF12D.real.ravel()
        row += n*p

        J[row:row+n*p, col] = dF12D.imag.ravel()

        col += 1

    # ============================================================
    # D parameters
    # ============================================================

    for part in ["real", "imag"]:
        for a in range(n):
            for b in range(p):

                dD = np.zeros((n, p), dtype=complex)
                dD[a, b] = 1.0 if part == "real" else 1.0j

                dH = dH_dD(dD, R, n, p)
                dF = dF_dH(H, dH, beta, eig_cache).T

                dF11 = dF[:n, :n]
                dF12 = dF[:n, n:]

                row = 0
                J[row:row+len(iu[0]), col] = dF11.real[iu]
                row += len(iu[0])

                J[row:row+len(iu_strict[0]), col] = dF11.imag[iu_strict]
                row += len(iu_strict[0])

                # product rule
                dF12D = dF12 @ D + F12 @ dD
                J[row:row+n*p, col] = dF12D.real.ravel()
                row += n*p

                J[row:row+n*p, col] = dF12D.imag.ravel()

                col += 1

    return J

def minimize_LR(x, beta, Lambda_c, D, F22_target, RTF12_target):
    r = residual_LR(x, beta, Lambda_c, D, F22_target, RTF12_target)
    return 0.5 * np.dot(r, r)

def grad_LR(x, beta, Lambda_c, D, F22_target, RTF12_target):
    r = residual_LR(x, beta, Lambda_c, D, F22_target, RTF12_target)
    J = jacobian_LR(x, beta, Lambda_c, D, F22_target, RTF12_target)
    return J.T @ r

def minimize_LcD(x, beta, Lambda, R, Delta_target, right_target):
    r = residual_LcD(x, beta, Lambda, R, Delta_target, right_target)
    return 0.5 * np.dot(r, r)

def grad_LcD(x, beta, Lambda, R, Delta_target, right_target):
    r = residual_LcD(x, beta, Lambda, R, Delta_target, right_target)
    J = jacobian_LcD(x, beta, Lambda, R, Delta_target, right_target)
    return J.T @ r

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
        verbose=1      # Useful to see if the cost function is actually decreasing
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
    #print('x0=',x0)
    #print('beta=',beta)
    #print('Lambda_c=')
    #print(Lambda_c)
    #print('D=')
    #print(D)
    #print('F22_target=')
    #print(F22_target)
    #print('RTF12_target=')
    #print(RTF12_target)
    #print(residual_LR(x0,beta, Lambda_c, D, F22_target, RTF12_target))
    #quit()
    if(False):
        sol = least_squares(
            residual_LR,
            x0,
            jac=jacobian_LR,
            #jac="2-point",
            #x_scale='jac',
            args=(beta, Lambda_c, D, F22_target, RTF12_target),
            method="trf", # Change from 'hybr' to 'lm'
            x_scale="jac", # should help
            max_nfev=100,
            ftol=1e-9,
            xtol=1e-9,
            verbose=2
        )
    elif(True):
        sol = minimize(
                  fun=lambda x: minimize_LR(x, beta, Lambda_c, D, F22_target, RTF12_target),
                  x0=x0,
                  jac=lambda x: grad_LR(x, beta, Lambda_c, D, F22_target, RTF12_target),
                  method="BFGS",#"BFGS",
                  options={"maxiter": 20000,
                           #"maxcor": 20,
                           #"ftol": 1e-12,
                           "eps": 1e-12,
                           "gtol": 1e-12,
                  }
              )
        print('sol.message=', sol.message)
        print('sol.fun=', sol.fun)
    elif(False):
        sol = root(
            residual_LR,
            x0,
            jac=jacobian_LR,
            args=(beta, Lambda_c, D, F22_target, RTF12_target),
            method="lm", # Change from 'hybr' to 'lm'
            options={'eps': 1e-9, 'factor': 0.5, 'xtol': 1e-9}
        )
        print('sol.success=', sol.success)
        print('sol.fun=', sol.fun)
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

def solve_F_only_LcD(beta, Lambda, R, Lambda_c0, D0, Delta_target, right_target):
    '''
    Solve the root finding problem for a given set of Lambda_c,D,beta,F22=<badg b> and RTF12=R.T@<fdag b>
    using numerical derivatives
    '''
    x0 = pack_params(Lambda_c0, D0)
    # We use least_squares because it supports 'trf' and '2-point' (numerical) jacobians
    sol = least_squares(
        residual_LcD,
        x0,
        jac='2-point', # This tells Scipy to compute the gradient numerically
        args=(beta, Lambda, R, Delta_target, right_target),
        method="trf",
        max_nfev=100,  # Limits total function calls to 200
        xtol=1e-9,
        ftol=1e-9,
        verbose=2      # Useful to see if the cost function is actually decreasing
    )

    Lambda_c_sol, D_sol = unpack_params(sol.x, Lambda_c0.shape[0], D0.shape[1])
    return sol, Lambda_c_sol, D_sol

def solve_F_dF_LcD(beta, Lambda, R, Lambda_c0, D0, Delta_target, right_target):
    '''
    Solve the root finding problem for Lambda and R
    given set of Lambda_c,D,beta,F22=<badg b> and RTF12=R.T@<fdag b>
    using analytical derivatives.
    Least_squares() proved to be faster than root()
    '''
    x0 = pack_params(Lambda_c0, D0)
    if(False):
        sol = least_squares(
            residual_LcD,
            x0,
            jac=jacobian_LcD,
            #jac='2-point',
            args=(beta, Lambda, R, Delta_target, right_target),
            method="trf", # Change from 'hybr' to 'lm'
            x_scale="jac", # should help
            max_nfev=100,
            ftol=1e-9,
            xtol=1e-9,
            verbose=2
        )
    elif(True):
        sol = minimize(
              fun=lambda x: minimize_LcD(x, beta, Lambda, R, Delta_target, right_target),
              x0=x0,
              jac=lambda x: grad_LcD(x, beta, Lambda, R, Delta_target, right_target),
              method="BFGS", # BFGS, SLSQP, L-BFGS-B
              options={"maxiter": 5000,
                       #"maxcor": 20,
                       #"ftol": 1e-12,
                       "eps": 1e-12,
                       "gtol": 1e-12,
              }

              )
        print('sol.message=', sol.message)
        print('sol.fun=', sol.fun)
    elif(False):
        sol = root(
            residual_LcD,
            x0,
            jac=jacobian_LcD,
            args=(beta, Lambda, R, Delta_target, right_target),
            method="lm", # Change from 'hybr' to 'lm'
            options={'ftol': 1e-9, 'xtol': 1e-9}
        )
    Lambda_c_sol, D_sol = unpack_params(sol.x, Lambda_c0.shape[0], D0.shape[1])
    return sol, Lambda_c_sol, D_sol

