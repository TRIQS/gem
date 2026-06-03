###########################################
#      utilities for quantum embedding methods
###########################################

import numpy as np
from numpy.linalg import  eigh
from cmath import sqrt
import itertools as it
from numba import jit 
#from scipy.optimize import bisect
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

def denRm1(x):
    # return (x*((1.0+0.j)-x))**(0.5)
    return (x*((1.0+0.j)-x)+1e-12)**(0.5)

def ddenRm1(x):
    # return ((0.5-x)/(x*((1.0+0.j)-x))**0.5)
    return ((0.5-x)/(x*((1.0+0.j)-x)+1e-12)**0.5)

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
            U_matrix[m, m, mp, mp] = U_int # intra-orbital
        else:
            U_matrix[m, m, mp, mp] = U_int - 2.0 * J_hund # inter-orbital opposite-spin
            U_matrix[m, mp, mp, m] = J_hund # spin-flip
            U_matrix[m, mp, m, mp] = J_hund # pair-hopping
    norb = U_matrix.shape[0]
    norb2 = norb * 2
    Ufull_matrix = np.zeros((norb2, norb2, norb2, norb2), dtype=np.complex128)
    Ufull_matrix[::2, ::2, ::2, ::2] = U_matrix  # up, up
    Ufull_matrix[1::2, 1::2, 1::2, 1::2] = U_matrix  # dn, dn
    Ufull_matrix[::2, ::2, 1::2, 1::2] = U_matrix  # up, dn
    Ufull_matrix[1::2, 1::2, ::2, ::2] = U_matrix  # dn, up
    return Ufull_matrix#, u_avg, j_avg

# Convert radial integrals F_k -> U,J
# ((l -m1) (k q) (l m3)) ((l -m2) (k -q) (l m4))
