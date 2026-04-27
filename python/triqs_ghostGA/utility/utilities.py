###########################################
#      utilities for quantum embedding methods
###########################################

import numpy as np
from numpy import sqrt, heaviside as hside, pi, arcsin
from numpy.linalg import inv, eigh
import h5py
import cmath
from cmath import sqrt
import itertools as it
from numba import jit  #, prange
import numba
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy.special import factorial as fact


# FUNCTIONS TRULY EXPOSED:


def calc_Lambda_c(R, Lambda, Delta_p, D, H_list):
    r"""Compute Lambda_c matrix from R, D, Lambda and Delta. It is given by

    .. math::
        \Lambda_c = \sum_s l^c_s h_s

    where :math:`h_s` are a set of matrices that forms a basis for the space of Hermitian matrices
    with dimension :math:`\nu`. The elements :math:`l^c_s` are obtained by calculating

    .. math::
        l^c_s = -l_s - \sum_{\alpha} \text{Tr} \left[ \frac{\partial}{\partial d^P_s} \left[ \Delta ( 1 - \Delta ) ]^{1/2} ( D R^T ) \right]

    where :math:`l_s` are the elements of the decomposition of :math:`\Lambda = \sum_s l_s h_s`
    and :math:`d^P_s` of :math:`\Delta_p = \sum_s d_s h_s^T`.

    Here, extracting the :math:`l_s`s from :math:`\Lambda` is done using the inverse_realHcombination function.
    The derivative with respect to :math:`d^P_s` is done using the dF function.
    :math:`\Lambda_c` is reconstructed using the realHcombinaison function.

    """
    l = inverse_realHcombination(Lambda, H_list)
    lc = np.copy(l) * 0.0
    DR = np.dot(D, np.transpose(R))
    for k in range(len(H_list)):
        AA = Delta_p
        HH = H_list[k].T
        derivative = dF(AA, HH, denRm1, ddenRm1)
        tt = np.trace(np.dot(DR ,derivative))
        lc[k] = -l[k] - (tt + np.conjugate(tt)).real
    Lambda_c = realHcombination(lc, H_list)
    return Lambda_c



def calc_Lambda(R, Lambda_c, Delta_p, D, H_list):
    """ Compute Lambda matrix from R, D, Lambda_c and Delta. It comes from the same equation as for Lambda_c. Lambda is expressed as

    .. math::
        \Lambda = \sum_s l_s h_s

    where :math:`h_s` are a set of matrices that forms a basis for the space of Hermitian matrices
    with dimension :math:`\nu`. They are given in H_list. The elements :math:`l_s` are obtained by calculating

    .. math::
        l_s = -l^c_s - \sum_{cb\alpha} \frac{\partial}{\partial d^P_s} \left[ \Delta ( 1 - \Delta ) ]^{1/2}_{cb} ( D R^T)_{bc}

    where :math:`l^c_s` are the elements of the decomposition of :math:`\Lambda_c = \sum_s l^c_s h_s`
    and :math:`d^P_s` of :math:`\Delta_p = \sum_s d_s h_s^T`.

    Here, extracting the :math:`l^c_s`s from :math:`\Lambda_c` is done using the inverse_realHcombination function.
    The derivative with respect to :math:`d^P_s` is done using the dF function.
    :math:`\Lambda` is reconstructed using the realHcombinaison function.

    """
    lc = inverse_realHcombination(Lambda_c, H_list)
    l = np.copy(lc) * 0.0
    DR = np.dot(D, np.transpose(R))
    for k in range(len(H_list)):
        AA = Delta_p
        HH = H_list[k].T
        derivative = dF(AA,HH, denRm1, ddenRm1)
        tt = np.trace(np.dot(DR, derivative))
        l[k] = -lc[k] - (tt + np.conjugate(tt)).real
    Lambda = realHcombination(l, H_list)
    return Lambda





#
def funcMat(H, f, tol=1e-8, pr=False):
    """
    Apply a scalar function f to a Hermitian matrix H via diagonalization.
    """
    H = np.asarray(H)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    if np.max(np.abs(H - H.conj().T)) > tol:
        raise ValueError("H must be Hermitian")

    w, U = eigh(H)
    if pr:
        print(f(w))
    return (U @ np.diag(f(w)) ) @ U.conj().T


def Hermitian_list(N: int, *, dtype=np.complex128):
    """
    Return an orthonormal (Hilbert–Schmidt) basis of N×N Hermitian matrices.

    Basis elements:
      - Diagonal: E_ii
      - Off-diagonal symmetric: (E_ij + E_ji)/sqrt(2)
      - Off-diagonal antisymmetric: i(E_ij - E_ji)/sqrt(2)

    Also returns a list of plain transposes (not conjugate transposes), matching
    the original code's tH_list behavior.

    Parameters
    ----------
    N : int
        Matrix dimension.
    dtype : numpy dtype
        Output dtype (default complex128).

    Returns
    -------
    H_list : list[np.ndarray]
        Hermitian basis matrices, length N^2.
    tH_list : list[np.ndarray]
        Plain transpose of each basis matrix, length N^2.
    """
    H_list, tH_list = [], []
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    # Diagonal basis
    for i in range(N):
        H = np.zeros((N, N), dtype=dtype)
        H[i, i] = 1.0
        H_list.append(H)
        tH_list.append(H)  # diagonal => transpose is itself

    # Off-diagonals
    for i in range(N):
        for j in range(i + 1, N):
            # symmetric real part
            H = np.zeros((N, N), dtype=dtype)
            H[i, j] = 1.0
            H[j, i] = 1.0
            H *= inv_sqrt2
            H_list.append(H)
            tH_list.append(H)  # symmetric => transpose equals itself

            # antisymmetric imaginary part
            H = np.zeros((N, N), dtype=dtype)
            H[i, j] = 1.0j
            H[j, i] = -1.0j
            H *= inv_sqrt2
            H_list.append(H)
            tH_list.append(H.T)  # matches original: transpose (not conjugate)

    assert len(H_list) == N * N
    return H_list, tH_list




# FUNCTIONS TO WORK ON

#problem with deg and used in Lambda/Lambda_c linear eq, need to use the same for 
def dF(A, H, function, d_function):
    evals, evecs = eigh(A)
    Hbar = np.dot(np.conj(evecs).T, np.dot( H, evecs) ) # transform H to A's basis
    #create Loewner matrix in A's basis
    loewm = np.zeros(evecs.shape,dtype=A.dtype)#dtype=np.complex128)
    for i in range(loewm.shape[0]):
        for j in range(loewm.shape[1]):
            if i==j:
                loewm[i,i] = d_function(evals[i]) # derivative(function, evals[i], dx=1e-12)
            if i!=j:
                if evals[i] != evals[j]:
                    loewm[i,j]= ( function(evals[i]) - function(evals[j]) )/(evals[i]-evals[j])
                else:
                    loewm[i,j] = d_function(evals[i]) # derivative(function, evals[i], dx=1e-12)

    # Perform the Schur product in A's basis then transform back to original basis.
    deriv = np.dot(evecs, np.dot( loewm*Hbar, np.conj(evecs).T ) )
    return deriv

#Also on delta fit

@jit(nopython=True)
def calc_Fermi(x):
    """
    calculate the fermi function for a vector x. Sometimes smearing the fermi function can
    lead to better convergence (but also lead to ficticious result if beta is too small).
    """
    f=[]
    for xx in x:
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

@jit(nopython=True)
def calc_nf(H,T):
    evals, evecs = eigh(H/T)
    func = calc_Fermi(evals)
    dm = np.zeros(H.shape,dtype=np.complex128)
    #dm[range(H.shape[0]),range(H.shape[0])] = func
    for i in range(H.shape[0]):
        dm[i,i] = func[i]
    return np.dot(evecs ,np.dot(dm, evecs.conj().T))




#define on the fly when needed
def denR(x):
    # return (x*((1.0+0.j)-x))**(-0.5)
    return (x*((1.0+0.j)-x)+1e-12)**(-0.5)

def denR_real(x):
    #return 1./sqrt(x*((1.0)-x))
    # return (x*((1.0)-x))**(-0.5)
    return (x*((1.0)-x)+1e-12)**(-0.5)

def denRm1(x):
    # return (x*((1.0+0.j)-x))**(0.5)
    return (x*((1.0+0.j)-x)+1e-12)**(0.5)

def denRm1_real(x):
    #return sqrt(x*((1.0)-x))#(x*((1.0)-x))**(0.5)
    return sqrt(x*((1.0)-x)+1e-12)#(x*((1.0)-x))**(0.5)

def ddenRm1(x):
    # return ((0.5-x)/(x*((1.0+0.j)-x))**0.5)
    return ((0.5-x)/(x*((1.0+0.j)-x)+1e-12)**0.5)

def ddenRm1_real(x):
    # return ((0.5-x)/sqrt(x*((1.0)-x)))#(x*((1.0)-x))**0.5)
    return ((0.5-x)/sqrt(x*((1.0)-x)+1e-12))#(x*((1.0)-x))**0.5)





#TO BE BETTER UNDERSTOOD:

def realHcombination(x, H_list):
    """H = sum_i x[i] * H_list[i]  (x real)."""
    M = len(x)
    N = H_list[0].shape[0]
    assert H_list[0].shape[0] == H_list[0].shape[1]
    assert len(x) == len(H_list)
    H = np.zeros((N, N))
    for i in range(M):
        H = H + x[i] * H_list[i]
    return H


def inverse_realHcombination(H, H_list):
    """x[i] = Re[Tr(H_list[i] @ H)] for Hermitian basis (returns real vector)."""
    M = len(H_list)
    assert H.shape[0] == H.shape[1]
    x_list = [np.trace(np.dot(H_list[i], H)).real for i in range(M)]
    return np.array(x_list)






#NOT USEFUL WHEN WE HAVE MODEST
def cut_small(mat, tol=1.0e-10):
    """ Utility function that sets all elements smaller than "tol"
    to zero in an numpy ndarray. """

    mat_return = np.zeros(mat.shape,dtype=mat.dtype)
    mat_return += mat.real * ( np.abs(mat.real) > tol )
    if mat.dtype == np.complex128:
        mat_return += 1.0j * mat.imag * ( np.abs(mat.imag) > tol )
    return mat_return




## INTERACTIONS

# The interaction matrix in desired basis.
# U^{spherical}_{m1 m4 m2 m3} =
# \sum_{k=0}^{2l} F_k angular_matrix_element(l, k, m1, m2, m3, m4)
# H = \frac{1}{2} \sum_{ijkl,\sigma \sigma'} U_{ikjl}
# a_{i \sigma}^\dagger a_{j \sigma'}^\dagger a_{l \sigma'} a_{k \sigma}.
def U_matrix_slater(l, radial_integrals=None, U_int=None, J_hund=None):
    r"""
    Calculate the full four-index U matrix being given either
    radial_integrals or U_int and J_hund.
    The convetion for the U matrix is that used to construct
    the Hamiltonians, namely:
    .. math:: H = \frac{1}{2} \sum_{ijkl,\sigma \sigma'} U_{ikjl}
            a_{i \sigma}^\dagger a_{j \sigma'}^\dagger
            a_{l \sigma'} a_{k \sigma}.
    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated
        (l=2 for d shell, l=3 for f shell).
    radial_integrals : list, optional
                       Slater integrals [F0,F2,F4,..].
                       Must be provided if U_int and J_hund are not given.
                       Preferentially used to compute the U_matrix
                       if provided alongside U_int and J_hund.
    U_int : scalar, optional
            Value of the screened Hubbard interaction.
            Must be provided if radial_integrals are not given.
    J_hund : scalar, optional
             Value of the Hund's coupling.
             Must be provided if radial_integrals are not given.
    Returns
    -------
    U_matrix : float numpy array
               The four-index interaction matrix in the chosen basis.
    """

    # Check all necessary information is present and consistent
    if radial_integrals is None and (U_int is None and J_hund is None):
        raise ValueError("U_matrix: provide either the radial_integrals" +
                " or U_int and J_hund.")
    if radial_integrals is None and (U_int is not None and J_hund is not None):
        radial_integrals = U_J_to_radial_integrals(l, U_int, J_hund)
    if radial_integrals is not None and \
            (U_int is not None and J_hund is not None):
        if len(radial_integrals) - 1 != l:
            raise ValueError("U_matrix: inconsistency in l" +
                    " and number of radial_integrals provided.")
        if not np.allclose(radial_integrals,
                U_J_to_radial_integrals(l, U_int, J_hund)):
            print(" Warning: U_matrix: radial_integrals provided\n" +
            " do not match U_int and J_hund.\n" +
            " Using radial_integrals to calculate U_matrix.")

    # Full interaction matrix
    # Basis of spherical harmonics Y_{-2}, Y_{-1}, Y_{0}, Y_{1}, Y_{2}
    # U^{spherical}_{m1 m4 m2 m3} = \sum_{k=0}^{2l} F_k
    # angular_matrix_element(l, k, m1, m2, m3, m4)
    U_matrix = np.zeros((2 * l + 1, 2 * l + 1, 2 * l + 1, 2 * l + 1), dtype=np.float64)

    m_range = range(-l, l + 1)
    for n, F in enumerate(radial_integrals):
        k = 2 * n
        for m1, m2, m3, m4 in it.product(m_range, m_range, m_range, m_range):
            U_matrix[m1 + l, m3 + l, m2 + l, m4 + l] += \
                    F * angular_matrix_element(l, k, m1, m2, m3, m4)
    return U_matrix


def U_matrix_kanamori(n_orb, U_int, J_hund):
    r"""
    Calculate the Kanamori U and Uprime matrices.
    Parameters
    ----------
    n_orb : integer
            Number of orbitals in basis.
    U_int : scalar
            Value of the screened Hubbard interaction.
    J_hund : scalar
             Value of the Hund's coupling.
    Returns
    -------
    U_matrix : float numpy array
               The four-index interaction matrix in the chosen basis.
    """

    # TODO: Use the native TRIQS function.

    U_matrix = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.float64)
    m_range = range(n_orb)
    for m, mp in it.product(m_range, m_range):
        if m == mp:
            U_matrix[m, m, mp, mp] = U_int
        else:
            U_matrix[m, m, mp, mp] = U_int - 2.0 * J_hund
            U_matrix[m, mp, mp, m] = J_hund
            U_matrix[m, mp, m, mp] = J_hund
    norb = U_matrix.shape[0]
    norb2 = norb * 2
    Ufull_matrix = np.zeros((norb2, norb2, norb2, norb2), dtype=np.complex128)
    Ufull_matrix[::2, ::2, ::2, ::2] = U_matrix  # up, up
    Ufull_matrix[1::2, 1::2, 1::2, 1::2] = U_matrix  # dn, dn
    Ufull_matrix[::2, ::2, 1::2, 1::2] = U_matrix  # up, dn
    Ufull_matrix[1::2, 1::2, ::2, ::2] = U_matrix  # dn, up
    return Ufull_matrix#, u_avg, j_avg


# Convert U,J -> radial integrals F_k
def U_J_to_radial_integrals(l, U_int, J_hund):
    r"""
    Determine the radial integrals F_k from U_int and J_hund.
    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated
        (l=2 for d shell, l=3 for f shell).
    U_int : scalar
            Value of the screened Hubbard interaction.
    J_hund : scalar
             Value of the Hund's coupling.
    Returns
    -------
    radial_integrals : list
                       Slater integrals [F0,F2,F4,..].
    """

    F = np.zeros((l + 1), dtype=np.float64)
    F[0] = U_int
    if l == 0:
        pass
    elif l == 1:
        F[1] = J_hund * 5.0
    elif l == 2:
        F[1] = J_hund * 14.0 / (1.0 + 0.625)
        F[2] = 0.625 * F[1]
    elif l == 3:
        F[1] = 6435.0 * J_hund / (286.0 + 195.0 * 0.668 + 250.0 * 0.494)
        F[2] = 0.668 * F[1]
        F[3] = 0.494 * F[1]
    else:
        raise ValueError(
            " U_J_to_radial_integrals: implemented only for l=0,1,2,3")
    return F


# Convert radial integrals F_k -> U,J
def radial_integrals_to_U_J(l, F):
    r"""
    Determine U_int and J_hund from the radial integrals.
    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated
        (l=2 for d shell, l=3 for f shell).
    F : list
        Slater integrals [F0,F2,F4,..].
    Returns
    -------
    U_int : scalar
            Value of the screened Hubbard interaction.
    J_hund : scalar
             Value of the Hund's coupling.
    """

    U_int = F[0]
    if l == 0:
        J_Hund = 0.
    elif l == 1:
        J_hund = F[1] / 5.0
    elif l == 2:
        J_hund = F[1] * (1.0 + 0.625) / 14.0
    elif l == 3:
        J_hund = F[1] * (286.0 + 195.0 * 0.668 + 250.0 * 0.494) / 6435.0
    else: raise ValueError("radial_integrals_to_U_J:" +
            " implemented only for l=2,3")

    return U_int, J_hund


# Angular matrix elements of particle-particle interaction
# (2l+1)^2 ((l 0) (k 0) (l 0))^2 \sum_{q=-k}^{k} (-1)^{m1+m2+q}
# ((l -m1) (k q) (l m3)) ((l -m2) (k -q) (l m4))
def angular_matrix_element(l, k, m1, m2, m3, m4):
    r"""
    Calculate the angular matrix element
    .. math::
       (2l+1)^2
       \begin{pmatrix}
            l & k & l \\
            0 & 0 & 0
       \end{pmatrix}^2
       \sum_{q=-k}^k (-1)^{m_1+m_2+q}
       \begin{pmatrix}
            l & k & l \\
         -m_1 & q & m_3
       \end{pmatrix}
       \begin{pmatrix}
            l & k  & l \\
         -m_2 & -q & m_4
       \end{pmatrix}.
    Parameters
    ----------
    l : integer
    k : integer
    m1 : integer
    m2 : integer
    m3 : integer
    m4 : integer
    Returns
    -------
    ang_mat_ele : scalar
                  Angular matrix element.
    """
    ang_mat_ele = 0
    for q in range(-k, k + 1):
        ang_mat_ele += three_j_symbol((l, -m1), (k, q), (l, m3)) * \
                three_j_symbol((l, -m2), (k, -q), (l, m4)) * \
                (-1.0 if (m1 + q + m2) % 2 else 1.0)
    ang_mat_ele *= (2 * l + 1) ** 2 * (three_j_symbol((l, 0), (k, 0), (l, 0)) ** 2)
    return ang_mat_ele


# Wigner 3-j symbols
# ((j1 m1) (j2 m2) (j3 m3))
def three_j_symbol(jm1, jm2, jm3):
    r"""
    Calculate the three-j symbol
    .. math::
       \begin{pmatrix}
        l_1 & l_2 & l_3\\
        m_1 & m_2 & m_3
       \end{pmatrix}.
    Parameters
    ----------
    jm1 : tuple of integers
          (j_1 m_1)
    jm2 : tuple of integers
          (j_2 m_2)
    jm3 : tuple of integers
          (j_3 m_3)
    Returns
    -------
    three_j_sym : scalar
                  Three-j symbol.
    """
    j1, m1 = jm1
    j2, m2 = jm2
    j3, m3 = jm3

    if (m1 + m2 + m3 != 0 or
        m1 < -j1 or m1 > j1 or
        m2 < -j2 or m2 > j2 or
        m3 < -j3 or m3 > j3 or
        j3 > j1 + j2 or
        j3 < abs(j1 - j2)):
        return .0

    three_j_sym = -1.0 if (j1 - j2 - m3) % 2 else 1.0
    three_j_sym *= sqrt(fact(j1 + j2 - j3) * fact(j1 - j2 + j3) * \
            fact(-j1 + j2 + j3) / fact(j1 + j2 + j3 + 1))
    three_j_sym *= sqrt(fact(j1 - m1) * fact(j1 + m1) * fact(j2 - m2) * \
            fact(j2 + m2) * fact(j3 - m3) * fact(j3 + m3))

    t_min = max(j2 - j3 - m1, j1 - j3 + m2, 0)
    t_max = min(j1 - m1, j2 + m2, j1 + j2 - j3)

    t_sum = 0
    for t in range(t_min, t_max + 1):
        t_sum += (-1.0 if t % 2 else 1.0) / (fact(t) * fact(j3 - j2 + m1 + t) * \
                fact(j3 - j1 - m2 + t) * fact(j1 + j2 - j3 - t) * fact(j1 - m1 - t) * fact(j2 + m2 - t))

    three_j_sym *= t_sum
    return three_j_sym


def get_average_uj(v2e):
    m_range = range(v2e.shape[0])
    u_avg = 0; j_avg = 0
    isum_u = 0; isum_j = 0
    for i, j in it.product(m_range, m_range):
        u_avg += v2e[i, i, j, j]
        isum_u += 1
        if i != j:
            j_avg += v2e[i, i, j, j] - v2e[i, j, j, i]
            isum_j += 1
    u_avg /= isum_u
    if isum_j > 0:
        j_avg = u_avg - j_avg / isum_j
    return u_avg, j_avg


def get_v2e_list(lhub, l_list, imap_list, utrans_list, u_list=None,
        j_list=None, f_list=None):
    mode_list = ['manual', 'slater-condon', 'kanamori', 'slater-condon']
    if lhub > 0:
        v2e_list = []
        u_avg_list = []
        j_avg_list = []

        for i, imap in enumerate(imap_list):
            if i > imap:
                v2e_list.append(v2e_list[imap])
                u_avg_list.append(u_avg_list[imap])
                j_avg_list.append(j_avg_list[imap])
                continue
            assert len(l_list[i]) == 1, " more than one l with lhub>0!"
            l_imp = l_list[i][0]
            utrans = utrans_list[i]

            if lhub in [1, 2]:
                v2e, u_avg, j_avg = U_matrix(mode_list[lhub], l_imp,
                        U_int=u_list[i],
                        J_hund=j_list[i], T=utrans)
            else:
                v2e, u_avg, j_avg = U_matrix(mode_list[lhub], l_imp,
                        radial_integrals=f_list[i][:l_imp + 1], T=utrans)

            v2e_list.append(v2e)
            u_avg_list.append(u_avg)
            j_avg_list.append(j_avg)
    else:
        v2e_list = None
        u_avg_list = None
        j_avg_list = None

    return v2e_list, u_avg_list, j_avg_list

