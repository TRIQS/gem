###########################################
#      utilities for grisb
#      Author: Tsung-Han Lee
#      email: henhans74716@gmail.com
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
# from scipy.linalg import eigh
from scipy.special import factorial as fact
# from scipy.misc import derivative

def get_1D_e_list(nmesh=500, t=0.5):
    """
    get 1D DOS energy list
    Input:
        nmesh: number of e points
        d: half-bandwidth
    Output:
        e_list: list of e points
    """
    k_list = np.linspace(-pi,pi,nmesh)
    e_list = 2.*t*np.cos(k_list)
    return e_list

def get_semicircle_e_list(nmesh=500, d=1.0, plot=False):
    """
    get semicircular DOS energy list
    Input:
        nmesh: number of e points
        d: half-bandwidth
    Output:
        e_list: list of e points
    """
    # dos
    dos = lambda e: 2./(pi*d**2) * sqrt(d**2-e**2)
    # cumulent dos
    # cdos = lambda e: (( e/d**2*sqrt(d**2-e**2) + np.arctan(e/sqrt(d**2-e**2)) )
    #                   / (pi) + 0.5)
    cdos = lambda e: ( e/d**2*sqrt(d**2-e**2) + arcsin(e/sqrt(d**2)) ) / (pi) + 0.5

    if plot is True:
        e_list = np.linspace(-d,d,100)
        plt.plot(e_list,dos(e_list))
        plt.plot(e_list,cdos(e_list))
        plt.plot(e_list,np.linspace(0,1,100))
        plt.show()
        quit()


    cdos_list = np.linspace(0,1,nmesh+1)
    e_list = [bisect(lambda x: cdos(x)-a, -d ,d) for a in cdos_list]
    e_list = np.asarray(e_list)
    e_list = (e_list[1:] + e_list[0:-1])/2

    if plot is True:
        plt.plot(e_list,cdos(e_list),'o')
        plt.plot(e_list,cdos_list,'-')
        plt.show()
        quit()

    return e_list

def get_flat_e_list(nmesh=500, d=1.0):
    """
    get flat DOS energy list
    Input:
        nmesh: number of e points
        d: half-bandwidth
    Output:
        e_list: list of e points
    """
    # dos
    dos = lambda e: 1./(2*d)*hside(e+d,0.5)*hside(-e+d,0.5)
    # cumulent dos
    cdos = lambda e: (1./(2*d)*hside(d+e,0.5)*
                      ((-d+e+2*d*hside(d,0.5))*hside(-d-e,0.5)*hside(d-e,0.5)
                       +hside(d,0.5)*(2*d+(-d+e)*hside(d-e,0.5))*hside(d+e,0.5)))

    #'''
    #e_list = np.linspace(-d,d,100)
    #plt.plot(e_list,dos(e_list))
    #plt.plot(e_list,cdos(e_list))
    #plt.plot(e_list,np.linspace(0,1,100))
    #plt.show()
    #quit()
    #'''

    cdos_list = np.linspace(0,1,nmesh+1)
    e_list = [bisect(lambda x: cdos(x)-a, -d ,d) for a in cdos_list]
    e_list = np.asarray(e_list)
    e_list = (e_list[1:] + e_list[0:-1])/2

    '''
    plt.plot(e_list,cdos(e_list),'o')
    plt.plot(e_list,cdos_list,'-')
    plt.show()
    quit()
    '''
    return e_list

def funcMat(H, function, pr=False):
    tiny = 1e-8 # use to regularize eigen problem for singular matrix
    H = H #+ np.eye(H.shape[0])*tiny
    N = H.shape[0]
    #print np.max(H.conj().T-H)
    assert(H.shape[0] == H.shape[1])
    #assert(np.allclose(H.conj().T,H,rtol=1e-08, atol=1e-08))

    if np.max(abs(H.conj().T-H)) > 1e-8:
        print('max(abs(H.conj().T-H))=',np.max(abs(H.conj().T-H)))
        raise
    #
    eigenvalues,U = eigh(H)
    Udagger = U.conj().T

    if pr:
        print(eigenvalues)
    #
    functioneigenvalues = function(eigenvalues)

    #print functioneigenvalues.dtype
    if pr:
        print(functioneigenvalues)
    #if H.dtype == float:#np.complex:
    #    #print functioneigenvalues.imag
    #    assert max(abs(functioneigenvalues.imag)) < 1e-12
    #    functioneigenvalues = functioneigenvalues.real
    functionE = np.zeros(U.shape,dtype=H.dtype)#dtype=np.complex128)
    functionE[np.arange(N), np.arange(N)] = functioneigenvalues
    functionH =  np.dot(np.dot(U,functionE),Udagger)
    #print functionE
    #
    return functionH

def calc_expH(H):
    '''
    exp of matrix with cutoff
    '''
    #print H
    evals, evecs = eigh(H)
    #func = np.exp(evals)
    func = []
    for e in evals:
        if e > 500.:
            func.append(np.exp(500))
        elif e < -500:
            func.append(np.exp(-500))
        else:
            func.append(np.exp(e))
    dm = np.zeros(H.shape)
    dm[list(range(H.shape[0])),list(range(H.shape[0]))] = func
    return np.dot(evecs ,np.dot(dm, evecs.conj().T))

def calc_logH(H):
    '''
    log of matrix with cutoff
    '''
    evals, evecs = eigh(H)
    #print evals
    #func = np.log(evals+0.0000001)
    func = [cmath.log(e+0.0000001) for e in evals]
    dm = np.zeros(H.shape,dtype=np.complex128)
    dm[list(range(H.shape[0])),list(range(H.shape[0]))] = func
    return np.dot(evecs ,np.dot(dm, evecs.conj().T))


def make_trivial_matrix_basis(N, symmetric=False):
    h_list = []
    for i,j in it.product(list(range(N)), list(range(N))):
        h = np.zeros((N, N))
        if symmetric:
            if i<=j:
                h[i,j] = 1
                h[j,i] = 1
            else:
                continue
        else:
            h[i,j] = 1
        h_list.append(h/sqrt(np.trace(np.dot(h.conj().T,h))))
    return h_list


def make_trivial_matrix_basis_sc(N):
    h_list = []
    for i,j in it.product(list(range(N)), list(range(N))):
        h = np.zeros((N, N))
        nl = bin(i).count('1')
        nr = bin(j).count('1')
        if (nl - nr)%2 == 0:
            h[i,j] = 1
        else:
            continue
        h_list.append(h/sqrt(np.trace(np.dot(h.conj().T,h))))
    return h_list


def map_to_herm_matrix(X, sp_basis_herm):
    return np.sum([x*h for x,h in zip(X, sp_basis_herm)], axis=0)

def map_to_matrix(X,sp_basis):
    return np.sum([x*h for x,h in zip(X, sp_basis)], axis=0)

def map_to_herm_vector(X, sp_basis_herm):
    return [np.trace(np.dot(np.matrix(h).getH(), X))/np.trace(np.dot(np.matrix(h).getH(), h)) for h in sp_basis_herm]

def map_to_vector(X, sp_basis):
    return [np.trace(np.dot(np.matrix(h).getH(), X))/np.trace(np.dot(np.matrix(h).getH(), h)) for h in sp_basis]

def Hermitian_list(N):
    H_list=[]
    tH_list=[]
    #
    Z=np.zeros((N,N),dtype=np.complex128)
    for i in range(N):
        H=Z*1.0
        H[i,i]=1.0
        H_list.append(H*1.0)
        tH_list.append(H*1.0)
    #
    for i in range(N):
        for j in range(i+1,N):
            H=Z*1.0
            H[i,j]=1.0
            H[j,i]=1.0
            H_list.append(H/sqrt(2.0))
            tH_list.append(H/sqrt(2.0))
            H=Z*1.0
            H[i,j]=1.0j
            H[j,i]=-1.0j
            H_list.append(H/sqrt(2.0))
            tH_list.append(np.transpose(H)/sqrt(2.0))
    #
    assert(len(H_list)==N**2)
    return H_list, tH_list

# Create canonical basis Symmetry NxN matrices (and its transpose)
def Symmetry_list(N):
    H_list=[]
    tH_list=[]
    #
    Z=np.zeros((N,N))
    for i in range(N):
        H=Z*1.0
        H[i,i]=1.0
        H_list.append(H*1.0)
        tH_list.append(H*1.0)
    #
    for i in range(N):
        for j in range(i+1,N):
            H=Z*1.0
            H[i,j]=1.0
            H[j,i]=1.0
            H_list.append(H/sqrt(2.0))
            tH_list.append(H/sqrt(2.0))
    #
    #print H_list
    assert(len(H_list)==N*(N+1)/2)
    return H_list, tH_list

# Create canonical basis Symmetry NxN matrices (and its transpose)
def Anti_symmetry_list(N):
    AS_list=[]
    tAS_list=[]
    #
    Z=np.zeros((N,N))
    #for i in range(N):
    #    AS=Z*1.0
    #    AS[i,i]=1.0
    #    AS_list.append(AS*1.0)
    #    tAS_list.append(AS*1.0)
    #
    for i in range(N):
        for j in range(i+1,N):
            AS=Z*1.0
            AS[i,j]=1.0
            AS[j,i]=-1.0
            AS_list.append(AS/sqrt(2.0))
            tAS_list.append(AS/sqrt(2.0))
    #
    #print AS_list
    assert(len(AS_list)==(N*(N+1)/2-N))
    return AS_list, tAS_list

def realcombination(x,H_list):
    M=len(x)
    N=H_list[0].shape[0]
    assert(H_list[0].shape[0]==H_list[0].shape[1])
    assert(len(x)==len(H_list))
    #
    H=np.zeros((N,N))
    for i in range(M):
        H = H + x[i] * H_list[i]
    #
    return H

# Given REAL array x and list of matrices H_list, construct
# linear combination:  H = \sum_n x_n [H_list]_n

def inverse_realcombination(H,H_list):
    #H_list,tH_list=Hermitian_list(N)
    M=len(H_list)
    assert(H.shape[0]==H.shape[1])
    #assert(M==N**2)
    #
    x_list=[]
    for i in range(M):
        x_list.append(np.trace(np.dot(H_list[i].conj().T,H)).real)
    x=np.array(x_list)
    #
    return x


def realHcombination(x,H_list):

    M=len(x)
    N=H_list[0].shape[0]
    assert(H_list[0].shape[0]==H_list[0].shape[1])
    assert(len(x)==len(H_list))
    #
    H=np.zeros((N,N))
    for i in range(M):
        H = H + x[i] * H_list[i]
    #
    return H

# Given REAL array x and list of matrices H_list, construct
# linear combination:  H = \sum_n x_n [H_list]_n

def inverse_realHcombination(H,H_list):
    #H_list,tH_list=Hermitian_list(N)
    M=len(H_list)
    assert(H.shape[0]==H.shape[1])
    #assert(M==N**2)
    #
    x_list=[]
    for i in range(M):
        x_list.append(np.trace(np.dot(H_list[i],H)).real)
    x=np.array(x_list)
    #
    return x

# Given Hermitian matrix H extract components x_n with respect
# to a set H_list of Hermitian matrices

def complexHcombination(v,H_list):
    twiceM=len(v)
    M=twiceM//2
    N=H_list[0].shape[0]
    assert(twiceM % 2 == 0) # Checking that dimension of v is even
    assert(H_list[0].shape[0]==H_list[0].shape[1])
    assert(len(v)==2*len(H_list))
    #
    x=v[0:M]
    y=v[M:twiceM]
    H=np.zeros((N,N))
    for i in range(M):
        H = H + x[i] * H_list[i]
    for i in range(M):
        H = H + 1j*y[i] * H_list[i]
    #
    return H


def complexAScombination(v,AS_list):
    twiceM=len(v)
    M=twiceM//2
    N=AS_list[0].shape[0]
    assert(twiceM % 2 == 0) # Checking that dimension of v is even
    assert(AS_list[0].shape[0]==AS_list[0].shape[1])
    assert(len(v)==2*len(AS_list))
    #
    x=v[0:M]
    y=v[M:twiceM]
    H=np.zeros((N,N))
    for i in range(M):
        H = H + x[i] * AS_list[i]
    for i in range(M):
        H = H + 1j*y[i] * AS_list[i]
    #
    return H

# Given REAL array x and list of matrices H_list, construct
# linear combination:  H = \sum_n x_n [H_list]_n
def realScombination(x,H_list):
    M=len(x)
    N=H_list[0].shape[0]
    assert(H_list[0].shape[0]==H_list[0].shape[1])
    assert(len(x)==len(H_list))
    #
    H=np.zeros((N,N))
    for i in range(M):
        H = H + x[i] * H_list[i]
    #
    return H

def realAScombination(v,AS_list):
    M=len(v)
    N=AS_list[0].shape[0]
    assert(AS_list[0].shape[0]==AS_list[0].shape[1])
    assert(len(v)==len(AS_list))
    #
    x=v[0:M]
    H=np.zeros((N,N))
    for i in range(M):
        H = H + x[i] * AS_list[i]
    #
    return H

def inverse_complexHcombination(H,H_list):
    #H_list,tH_list=Hermitian_list(N)
    M=len(H_list)
    assert(H.shape[0]==H.shape[1])
    #assert(M==N**2)
    #
    xr_list=[]
    xi_list=[]
    for i in range(M):
        xr_list.append( np.real( np.trace(np.dot(H_list[i],H) ) ) )
        xi_list.append( np.imag( np.trace(np.dot(H_list[i],H) ) ) )
    xr=np.array(xr_list)
    xi=np.array(xi_list)
    #
    return np.hstack((xr,xi))

# Given Hermitian matrix H extract components x_n with respect
# to a set H_list of Hermitian matrices
def inverse_realScombination(H,H_list):
    #H_list,tH_list=Symmetry_list(N)
    M=len(H_list)
    assert(H.shape[0]==H.shape[1])
    #assert(M==N*(N+1)/2)
    #
    x_list=[]
    for i in range(M):
        x_list.append(np.trace(np.dot(H_list[i],H)))
    x=np.array(x_list)
    #
    return x


def inverse_complexAScombination(AS,AS_list):
    #AS_list,tAS_list=Anti_symmetry_list(N)
    M=len(AS_list)
    assert(AS.shape[0]==AS.shape[1])
    #assert(M==N**2)
    #
    xr_list=[]
    xi_list=[]
    for i in range(M):
        xr_list.append( np.real( np.trace(np.dot(AS_list[i].conj().T,AS) ) ) )
        xi_list.append( np.imag( np.trace(np.dot(AS_list[i].conj().T,AS) ) ) )
        #xr_list.append( np.real( np.trace(np.dot(-AS_list[i].T,AS) ) ) )
        #xi_list.append( np.imag( np.trace(np.dot(-AS_list[i].T,AS) ) ) )
    xr=np.array(xr_list)
    xi=np.array(xi_list)
    #
    return np.hstack((xr,xi))

def inverse_realAScombination(AS,AS_list):
    #AS_list,tAS_list=Anti_symmetry_list(N)
    M=len(AS_list)
    assert(AS.shape[0]==AS.shape[1])
    #assert(M==N**2)
    #
    xr_list=[]
    for i in range(M):
        xr_list.append( np.real( np.trace(np.dot(AS_list[i].conj().T,AS) ) ) )
    xr=np.array(xr_list)
    #
    return xr

def build_qpH(ek, R, Lambda):
    '''
    Build quasiparticle Hamiltonian
    '''
    return np.dot(R, np.dot(ek, R.conj().T ) ) + Lambda

#def build_qpHbdg(ek, R, Lambda):
#    '''
#    Build quasiparticle Bogoliubov de Gennes Hamiltonian
#    '''
#    no = ek.shape[0]
#    Lambdabdg = np.copy(qpHbdg)
#    qpHbdg[:no,:no] = ek
#    qpHbdg[no:,no:] = -ek.conj()
#    return qpHbdg

def set_R_BdG(R, Q):
    no = R.shape[0]
    R_BdG = np.zeros((2*no,2*no),dtype=R.dtype)
    R_BdG[:no,:no] = R
    R_BdG[no:,no:] = -R.conj()
    R_BdG[:no,no:] = -Q.conj()
    R_BdG[no:,:no] = Q
    return R_BdG

def set_Lambda_BdG(Lambda, Lambdap):
    no = Lambda.shape[0]
    Lambda_BdG = np.zeros((2*no,2*no),dtype=Lambda.dtype)
    Lambda_BdG[:no,:no] = Lambda
    Lambda_BdG[no:,no:] = -Lambda.conj()
    Lambda_BdG[:no,no:] = Lambdap
    Lambda_BdG[no:,:no] = -Lambdap.conj()
    return Lambda_BdG

def updn_to_spinful_mat(Mup,Mdn):
    M = np.zeros((2*Mup.shape[0],2*Mup.shape[1]),dtype=Mup.dtype)
    M[::2,::2] = Mup
    M[1::2,1::2] = Mdn
    return M

def duplicate_in_spin_space(A):
    """
    Take  matrix acting in single-particle spinless space and
    duplicate it to act in spin space, i.e construct :math:`\tilde{A}` such that:

    .. math::
               \tilde{A}_{2i, 2j} = \tilde{A}_{2i+1,2j+1} = A_{ij}

    Parameters
    ----------
    A : NxN matrix

    Returns
    -------
    At : 2Nx2N matrix

    See also
    --------
    spin_symmetrize
    """
    if "Gf" in str(type(A)):
        At = type(A)(mesh = A.mesh, shape = (A.target_shape[0]*2,A.target_shape[1]*2))
        for iw in range(len(At.mesh)):
            for i,j in it.product(list(range(A.target_shape[0])), list(range(A.target_shape[1]))):
                At.data[iw, 2*i,2*j] = A.data[iw,i,j]
                At.data[iw, 2*i+1,2*j+1] = A.data[iw,i,j]
    else:
        At = np.zeros((A.shape[0]*2,A.shape[1]*2), dtype=A.dtype)
        for i,j in it.product(list(range(A.shape[0])), list(range(A.shape[1]))):
            At[2*i,2*j]=A[i,j]
            At[2*i+1,2*j+1]=A[i,j]
    return At


def enlarge_in_spin_space(A):
    """
    Take matrix acting in single-particle spinless space and
    enlarge it to act in spin space up and down separately.

    Parameters
    ----------
    A : NxN matrix

    Returns
    -------
    At : 2 2Nx2N matrices corresponds to spin up and sown

    """
    if "Gf" in str(type(A)):
#     At = type(A)(mesh = A.mesh, shape = (A.target_shape[0]*2,A.target_shape[1]*2))
#     for iw in range(len(At.mesh)):
#      for i,j in it.product(range(A.target_shape[0]), range(A.target_shape[1])):
#        At.data[iw, 2*i,2*j] = A.data[iw,i,j]
#        At.data[iw, 2*i+1,2*j+1] = A.data[iw,i,j]
        raise Exception("Gf is not yet implemented enlarge in spin space!")
    else:
        Atup = np.zeros((A.shape[0]*2,A.shape[1]*2), dtype=A.dtype)
        Atdn = np.zeros((A.shape[0]*2,A.shape[1]*2), dtype=A.dtype)
        for i,j in it.product(list(range(A.shape[0])), list(range(A.shape[1]))):
            Atup[2*i,2*j]=A[i,j]
            Atdn[2*i+1,2*j+1]=A[i,j]
    return Atup, Atdn


def coefficients_identity(N):
    d=2*N**2
    v=np.zeros(d)
    v[0:N]=np.ones(N)
    #
    return v

def get_blocks(H, s):
    N=H.shape[0]
    assert(H.shape[0]==H.shape[1])
    assert(s<=N)
    #
    S=H[0:s,0:s]
    B=H[s:N,s:N]
    V=H[0:s,s:N]
    Vdagger=H[s:N,0:s]
    #
    return S,B,V,Vdagger

def set_blocks(S,B,V):
    s=S.shape[0]
    b=B.shape[0]
    N=s+b
    #
    H=np.zeros((N,N),dtype=S.dtype)#dtype=np.complex128)
    H[0:s,0:s]=S
    H[s:N,s:N]=B
    H[0:s,s:N]=V
    H[s:N,0:s]=np.transpose(np.conjugate(V))
    #
    return H

def set_blocks_not_Hermitian(S,B,V1,V2):
    s=S.shape[0]
    b=B.shape[0]
    N=s+b
    #
    H=np.zeros((N,N),dtype=S.dtype)#dtype=np.complex128)
    H[0:s,0:s]=S
    H[s:N,s:N]=B
    H[0:s,s:N]=V1
    H[s:N,0:s]=V2
    #
    return H

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

def dF_real(A, H, function, d_function):
    #tiny = 1e-8 # use to regularize eigen problem for singular matrix
    A = A #+ np.eye(H.shape[0])*tiny

    evals, evecs = eigh(A)
    #print A
    #print evals
    #print evecs
    Hbar = np.dot(np.conj(evecs).T, np.dot( H, evecs) ) # transform H to A's basis
    #create Loewner matrix in A's basis
    loewm = np.zeros(evecs.shape)
    for i in range(loewm.shape[0]):
        for j in range(loewm.shape[1]):
            if i==j:
                loewm[i,i] = d_function(evals[i])
                #loewm[i,i] = derivative(function, evals[i], dx=1e-12)
            if i!=j:
                if evals[i] != evals[j]:
                    loewm[i,j]= ( ( function(evals[i]) - function(evals[j]) )/(evals[i]-evals[j]) )
                else:
                    loewm[i,j] = d_function(evals[i])
                    #loewm[i,j] = derivative(function, evals[i], dx=1e-12)

    # Perform the Schur product in A's basis then transform back to original basis.
    deriv = np.dot(evecs, np.dot( loewm*Hbar, np.conj(evecs).T ) )
    return deriv

def calc_Hqp(T,Lambda,R):
    assert(Lambda.shape==R.shape)
    #
    N=T.shape[0]
    s=Lambda.shape[0]
    b=N-s
    #
    S,B,V,Vdagger=get_blocks(T,s)
    Vstar = np.dot(R,V)
    Hqp=set_blocks(Lambda,B,Vstar)
    #
    return Hqp

#def calc_nf(H,T):
#    evals, evecs = eigh(H/T)
#    func = calc_Fermi(evals)
#    dm = np.zeros(H.shape)
#    dm[list(range(H.shape[0])),list(range(H.shape[0]))] = func
#    return np.dot(evecs ,np.dot(dm, evecs.conj().T))
#
#def calc_nf0(H):
#    evals, evecs = eigh(H)
#    func = calc_Fermi0(evals)
#    dm = np.zeros(H.shape)
#    dm[list(range(H.shape[0])),list(range(H.shape[0]))] = func
#    return np.dot(evecs ,np.dot(dm, evecs.conj().T))

@jit(nopython=True)
def calc_nf(H,T):
    evals, evecs = eigh(H/T)
    func = calc_Fermi(evals)
    dm = np.zeros(H.shape,dtype=np.complex128)
    #dm[range(H.shape[0]),range(H.shape[0])] = func
    for i in range(H.shape[0]):
        dm[i,i] = func[i]
    return np.dot(evecs ,np.dot(dm, evecs.conj().T))

#@jit(nopython=True)
def calc_nf0(H):
    evals, evecs = eigh(H)
    func = calc_Fermi0(evals)
    dm = np.zeros(H.shape)
    dm[range(H.shape[0]),range(H.shape[0])] = func
    return np.dot(evecs ,np.dot(dm, evecs.conj().T))

def calc_C(H):
    assert(H.shape[0] == H.shape[1])
    #
    #print "H in calc_C"
    #print H
    C = funcMat(H, calc_Fermi0)
    #
    return C

def calc_C_T(H,T):
    assert(H.shape[0] == H.shape[1])
    #
    #print "H in calc_C"
    #print H
    C = funcMat(H/T, calc_Fermi)
    #
    return C

def calc_C_hole(H):
    assert(H.shape[0] == H.shape[1])
    #
    #print "H in calc_C"
    #print H
    C = funcMat(H, calc_hole_Fermi0)
    #
    return C

def denR(x,eps=0.0):
    # return (x*((1.0+0.j)-x))**(-0.5)
    #return (x*((1.0+0.j)-x)+1e-12)**(-0.5)
    return ( (x+eps)*((1.0+0.j)-x+eps))**(-0.5)

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

#def calc_Fermi(x):
#    """
#    calculate the fermi function for a vector x. Sometimes smearing the fermi function can
#    lead to better convergence (but also lead to ficticious result if beta is too small).
#    """
#    f=[]
#    for xx in x:
#        # This one is used to stablize selective Mott, but would lead to suprious OSMT if temperature is too high.
#        #f.append(1./(1+np.exp(500*xx)))
#        # This one is important to get the correct phase diagram (especially for criyical t2/t1), but not stable in OSMP.
#        if abs(xx)<500:
#            f.append(1./(1+np.exp(xx)))
#        elif xx< -500:
#            f.append(1)
#        elif xx> 500:
#            f.append(0)
#    return np.array(f)
#
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

#@jit(nopython=True)
def calc_Fermi0(x):
    """
    calculate the fermi function for a vector x. Sometimes smearing the fermi function can
    lead to better convergence (but also lead to ficticious result if beta is too small).
    """
    f=[]
    for xx in x:
        # This one is used to stablize selective Mott, but would lead to suprious OSMT if temperature is too high.
        #f.append(1./(1+np.exp(300*xx)))
        #f.append( np.exp(-300.*xx/2 - np.log(2) - np.log(np.cosh(300.*xx/2)) ) )
        # This one is important to get the correct phase diagram (especially for criyical t2/t1), but not stable in OSMP.
        if abs(xx)<1.e-9:
            #f.append(1./(1+np.exp(50000*xx)))
            f.append(0.5)
        elif xx<-1e-9:
            f.append(1)
        elif xx> 1e-9:
            f.append(0)
        #if xx <1.e-12:
        #    f.append(1)
        #elif xx>= 1.e-12:
        #    f.append(0)
    return np.array(f)

def calc_hole_Fermi0(x):
    """
    calculate the fermi function for a vector x. Sometimes smearing the fermi function can
    lead to better convergence (but also lead to ficticious result if beta is too small).
    """
    f=[]
    for xx in x:
        # This one is used to stablize selective Mott, but would lead to suprious OSMT if temperature is too high.
        #f.append(1./(1+np.exp(500*xx)))
        # This one is important to get the correct phase diagram (especially for criyical t2/t1), but not stable in OSMP.
        if abs(xx)<1.e-12:
            #f.append(1./(1+np.exp(500*xx)))
            f.append(0.5)
        elif xx< -1.e-12:
            f.append(0)
        elif xx> 1.e-12:
            f.append(1)
    return np.array(f)

def cut_small(mat, tol=1.0e-10):
    """ Utility function that sets all elements smaller than "tol"
    to zero in an numpy ndarray. """

    mat_return = np.zeros(mat.shape,dtype=mat.dtype)
    mat_return += mat.real * ( np.abs(mat.real) > tol )
    if mat.dtype == np.complex128:
        mat_return += 1.0j * mat.imag * ( np.abs(mat.imag) > tol )
    return mat_return

def spin_symmetrize(A, tol=1e-12):
    """Symmetrize spin up and dn, i.e compute matrix :math:`A^\mathrm{sym}` such that

    .. math::

      A^\mathrm{sym}_{ij} = \\frac{1}{2} (A_{2i,2j} + A_{2i+1,2j+1})

    Parameters
    ------------
    A : matrix or GfImFreq

    See also
    --------
    duplicate_in_spin_space
    """
    if "Gf" in str(type(A)):
        A_sym = type(A)(mesh = A.mesh, shape = (A.target_shape[0]/2, A.target_shape[1]/2))
        for i,j in it.product(list(range(A_sym.target_shape[0])), list(range(A_sym.target_shape[1]))):
            diff = A[2*i,2*j] - A[2*i+1,2*j+1]
            for iw in range(len(diff.mesh)):
                assert (abs(diff.data[iw,0,0])<tol), "symmetrizing matrix with spin differentiation for element %s, %s! : %s neq %s"%(i,j, A[2*i, 2*j], A[2*i+1, 2*j+1])
            A_sym[i,j] = 0.5*(A[2*i,2*j] + A[2*i+1,2*j+1])
    elif (type(A)==np.ndarray and A.ndim==3) or type(A)==list:
        A_sym = np.zeros((A.shape[0],A.shape[1]/2,A.shape[2]/2), dtype = A.dtype)
        for iw in range(len(A)):
            tmp = np.zeros((A[iw].shape[0]/2,A[iw].shape[1]/2), dtype = A.dtype)
            for i,j in it.product(list(range(tmp.shape[0])), list(range(tmp.shape[1]))):
                diff = A[iw,2*i,2*j] - A[iw,2*i+1,2*j+1]
                assert (abs(diff)<tol), "symmetrizing matrix with spin differentiation for element %s, %s! : %s neq %s"%(i,j, A[iw,2*i, 2*j], A[iw,2*i+1, 2*j+1])
                tmp[i,j] = 0.5*(A[iw,2*i,2*j] + A[iw,2*i+1,2*j+1])
            A_sym[iw,:,:] = tmp
    else:
        A_sym = np.zeros((A.shape[0]/2,A.shape[1]/2), dtype = A.dtype)
        for i,j in it.product(list(range(A_sym.shape[0])), list(range(A_sym.shape[1]))):
            assert (abs(A[2*i,2*j] - A[2*i+1,2*j+1])<tol), "symmetrizing matrix with spin differentiation for element %s, %s! : %s neq %s"%(i,j, A[2*i, 2*j], A[2*i+1, 2*j+1])
            A_sym[i,j] = 0.5*(A[2*i,2*j] + A[2*i+1,2*j+1])

    return A_sym

#########################################################################
# This interface is written by Dr Yongxin Yao in Ames Lab
#########################################################################
def gen_spci_input_file(U, E, D, LAM, na2, norb_mott, nval_bot, nval_top, sig ):
    '''
    Generate input files for spci.
    '''
    #print sig
    f = h5py.File('EMBED_HAMIL_1.h5','w')
    f['/D']=D.T
    f['/H1E']=E.T
    f['/LAMBDA']=LAM.T
    f['/V2E']=U.T
    f['/na2']=[na2]
    f['/norb_mott']=[norb_mott]
    f['/nval_bot']=[nval_bot]
    f['/nval_top']=[nval_top]
    f['/sigma_struct']=sig
    #f = h5py.File('HEmbed.h5','w')
    #f["/impurity_0/D"] = D.T
    #f["/impurity_0/H1E"] = E.T
    #f["/impurity_0/LAMBDA"] = LAM.T
    #f["/impurity_0/V2E"] = U.T
    #f["/impurity_0/m_struct"] = sig
    #f["/impurity_0/na2"]=[na2]
    #f.close()


def calc_Ek_T0(ek_list, R, lam):
    ek_tot = sum([np.sum( ( np.dot(R, np.dot(x, R.conj().T )) ) * \
                    calc_C( np.dot(R, np.dot(x, R.conj().T) ) + lam ).T ) for x in ek_list] )
    #
    #print ek_tot
    return 1.*(ek_tot).real/len(ek_list)
    #return 2*(ek_tot).real/len(ek_list)

def calc_Ek_T(ek_list, R, lam, T):
    ek_tot = sum([np.sum( ( np.dot(R, np.dot(x, R.conj().T )) ) * \
                    calc_nf( np.dot(R, np.dot(x, R.conj().T) ) + lam , T).T ) for x in ek_list] )
    #
    #print ek_tot
    return 1.*(ek_tot).real/len(ek_list)
    #return 2*(ek_tot).real/len(ek_list)

def calc_Sqp(ek_list, R, lam, T):
#    Sqp = -sum(map(lambda x: np.trace( np.dot(calc_nf( np.dot(R, np.dot(x, R.conj().T) ) + lam, T ).T, \
#               calc_logH( calc_nf( np.dot(R, np.dot(x, R.conj().T) ) + lam, T ).T ) ) + \
#               np.dot(np.eye(x.shape[0])-calc_nf( np.dot(R, np.dot(x, R.conj().T) ) + lam, T ).T, \
#               calc_logH(np.eye(x.shape[0])-calc_nf( np.dot(R, np.dot(x, R.conj().T) ) + lam, T ).T) ) )\
#               , ek_list) )/len(ek_list)
    Sqp = -sum([np.sum( np.dot(calc_C_T( np.dot(R, np.dot(x, R.conj().T) ) + lam, T ).T, \
               calc_logH( calc_C_T( np.dot(R, np.dot(x, R.conj().T) ) + lam, T ).T ) ) + \
               np.dot(np.eye(x.shape[0])-calc_C_T( np.dot(R, np.dot(x, R.conj().T) ) + lam, T ).T, \
               calc_logH(np.eye(x.shape[0])-calc_C_T( np.dot(R, np.dot(x, R.conj().T) ) + lam, T ).T) ) ) for x in ek_list] )/len(ek_list)
#    Sqp = 0
#    for x in ek_list:
#      print x, np.dot(R, np.dot(x, R.conj().T) ), calc_nf(np.dot(R, np.dot(x, R.conj().T) ) + lam, T).T
#      print calc_C(np.dot(R, np.dot(x, R.conj().T) ) + lam).T

    #
    #print Sqp
    return Sqp.real

########################################################################
#                  Wannier tight-binding tools                         #
########################################################################
#@jit(nopython=True)
#def AssembleHk(kx,ky,kz,hopp_dict, cutoff=0.005):
#    """This is tight-binding Hamiltonian for the two-band model
#    with hopping parameters listed in "wannier_hr.dat".
#    """
#    Hk = np.zeros((3,3), dtype=complex128)
#    for R, hmn in hopp_dict.items():
#        #print('R=',R)
#        for m in range(3):
#            for n in range(3):
#                if np.abs(hmn["h"][m,n])/float(hmn["deg"]) > cutoff:
#                    Hk[m,n] += hmn["h"][m,n]*np.exp(1j*(kx*R[0]+ky*R[1]+kz*R[2]))/float(hmn["deg"])
#    return Hk
@jit(nopython=True)
def AssembleHk(kx,ky,kz,Rs,hmns,degs, no, cutoff=0.00):
    """This is tight-binding Hamiltonian for the two-band model
    with hopping parameters listed in "wannier_hr.dat".
    """
    Hk = np.zeros((no,no), dtype=np.complex128)#numba.complex128)
    for i in range(Rs.shape[0]):
        #print('R=',R)
        for m in range(no):
            for n in range(no):
                if np.abs(hmns[i][m,n])/float(degs[i]) > cutoff:
                    Hk[m,n] += hmns[i][m,n]*np.exp(1j*(kx*Rs[i][0]+ky*Rs[i][1]+kz*Rs[i][2]))/float(degs[i])
    return Hk


def parse_hopping_from_wannier90_hr_dat(filename):
    # read in hamiltonian matrix, in eV
    f=open(filename,"r")
    ln=f.readlines()
    f.close()
    #
    # get number of wannier functions
    num_wan=int(ln[1])
    # get number of Wigner-Seitz points
    num_ws=int(ln[2])
    # get degenereacies of Wigner-Seitz points
    deg_ws=[]
    for j in range(3,len(ln)):
        sp=ln[j].split()
        for s in sp:
            deg_ws.append(int(s))
        if len(deg_ws)==num_ws:
            last_j=j
            break
        if len(deg_ws)>num_ws:
            raise Exception("Too many degeneracies for WS points!")
    deg_ws=np.array(deg_ws,dtype=int)
    # now read in matrix elements
    # Convention used in w90 is to write out:
    # R1, R2, R3, i, j, ham_r(i,j,R)
    # where ham_r(i,j,R) corresponds to matrix element < i | H | j+R >
    ham_r={} # format is ham_r[(R1,R2,R3)]["h"][i,j] for < i | H | j+R >
    ind_R=0 # which R vector in line is this?
    for j in range(last_j+1,len(ln)):
        sp=ln[j].split()
        # get reduced lattice vector components
        ham_R1=int(sp[0])
        ham_R2=int(sp[1])
        ham_R3=int(sp[2])
        # get Wannier indices
        ham_i=int(sp[3])-1
        ham_j=int(sp[4])-1
        # get matrix element
        ham_val=float(sp[5])+1.0j*float(sp[6])
        # store stuff, for each R store hamiltonian and degeneracy
        ham_key=(ham_R1,ham_R2,ham_R3)
        if (ham_key in ham_r)==False:
            ham_r[ham_key]={
                "h":np.zeros((num_wan,num_wan),dtype=complex),
                "deg":deg_ws[ind_R]
                }
            ind_R+=1
        ham_r[ham_key]["h"][ham_i,ham_j]=ham_val

    Rs = []
    hmns = []
    degs = []
    for R, hmn in ham_r.items():
        #print('R=',R)
        Rs.append(R)
        hmns.append(hmn["h"])
        degs.append(hmn["deg"])
    return np.array(Rs), np.array(hmns), np.array(degs)
    #return ham_r, num_wan


####################################################################################
#                      Multiorbital Coulomb interaction                            #
####################################################################################
def transform_U_matrix(U_matrix, T):
    r"""
    Transform a four-index interaction matrix into another basis.
    The transformation matrix is defined such that new creation operators :math:`b^\dagger` are related to
    the old ones :math:`a^\dagger` as

    .. math:: b_{i \sigma}^\dagger = \sum_j T_{ij} a^\dagger_{j \sigma}.

    Parameters
    ----------
    U_matrix : float numpy array
               The four-index interaction matrix in the original basis.
    T : real/complex numpy array, optional
        Transformation matrix for basis change.
        Must be provided if basis='other'.

    Returns
    -------
    U_matrix : float numpy array
               The four-index interaction matrix in the new basis.

    """
    # TODO: why is it not conjugate on the right side but transpose?
    return np.einsum("ij,kl,jlmo,mn,op",np.conj(T).T,np.conj(T).T,U_matrix,T,T)

def GetUMatrix(U, J, norb2, l, isrel=False, utrans=False):

    Ufullmat = np.zeros((norb2, norb2, norb2, norb2), dtype=np.complex128)

    if not isrel:
        Umat_spher = U_matrix_slater(l, radial_integrals=None, U_int=U, J_hund=J)
        #utrans = spherical_to_cubic(l, convention='wien2k')
        #print( 'spherical to cubic transform matrix' )
        #print( utrans )

        # transform to cubic
        Umat_spher = np.array(Umat_spher, dtype=np.complex128)
        Umat_cubic = unitary_transform_coulomb_matrix(Umat_spher, utrans.T) # transpose due to strange convention in triqs unitary transformation
        u_avg_spher, j_avg_spher = get_average_uj(Umat_spher)
        u_avg_cubic, j_avg_cubic = get_average_uj(Umat_cubic)

        Ufullmat[::2, ::2, ::2, ::2] = Umat_cubic  # up, up
        Ufullmat[1::2, 1::2, 1::2, 1::2] = Umat_cubic  # dn, dn
        Ufullmat[::2, ::2, 1::2, 1::2] = Umat_cubic  # up, dn
        Ufullmat[1::2, 1::2, ::2, ::2] = Umat_cubic  # dn, up

        print( 'U_avg_spher=', u_avg_spher, 'J_avg_spher=', j_avg_spher )
        print( 'U_avg_cubic=', u_avg_cubic, 'J_avg_cubic=', j_avg_cubic )
    else:
        #print( 'relativeistic Coulomb interaction not implemented!!' )
        #raise
        Umat_spher = U_matrix_slater(l, radial_integrals=None, U_int=U, J_hund=J)
        Umat_spher = np.array(Umat_spher, dtype=np.complex128)

        no = 2*l+1
        Ufullmat[:no, :no, :no, :no] = Umat_spher  # up, up
        Ufullmat[no:, no:, no:, no:] = Umat_spher  # dn, dn
        Ufullmat[:no, :no, no:, no:] = Umat_spher  # up, dn
        Ufullmat[no:, no:, :no, :no] = Umat_spher  # dn, up

        if utrans is not None: Ufullmat = unitary_transform_coulomb_matrix(Ufullmat, utrans)

    return Ufullmat


##############################################################################
#                           U_matrix from triqs                              #
##############################################################################


# Rotation matrices: complex harmonics to cubic harmonics
# Complex harmonics basis: ..., Y_{-2}, Y_{-1}, Y_{0}, Y_{1}, Y_{2}, ...
def spherical_to_cubic(l, convention=''):
    r"""
    Get the spherical harmonics to cubic harmonics transformation matrix.

    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated (l=2 for d shell, l=3 for f shell).
    convention : string, optional
                 The basis convention.
                 Takes the values

                 - '': basis ordered as ("xy","yz","z^2","xz","x^2-y^2"),
                 - 'wien2k': basis ordered as ("z^2","x^2-y^2","xy","yz","xz").

    Returns
    -------
    T : real/complex numpy array
        Transformation matrix for basis change.

    Note
    -------
    complex spherical harmonic arange as (l, l-1, ..., -l+1, l )
    T_ij: i index for cubic harmonic and j index for complex spherical harmonic.

    """
    if not convention in ('wien2k', ''):
        raise ValueError("Unknown convention: " + str(convention))

    size = 2 * l + 1
    T = np.zeros((size, size), dtype=complex)
    if convention == 'wien2k' and l != 2:
        raise ValueError("spherical_to_cubic: wien2k convention implemented only for l=2")
    if l == 0:
        cubic_names = ("s")
    elif l == 1:
        cubic_names = ("x", "y", "z")
        T[0, 0] = 1.0 / sqrt(2);   T[0, 2] = -1.0 / sqrt(2)
        T[1, 0] = 1j / sqrt(2);    T[1, 2] = 1j / sqrt(2)
        T[2, 1] = 1.0
    elif l == 2:
        if convention == 'wien2k':
            cubic_names = ("z^2", "x^2-y^2", "xy", "yz", "xz")
            T[0, 2] = 1.0
            T[1, 0] = 1.0 / sqrt(2);   T[1, 4] = 1.0 / sqrt(2)
            T[2, 0] = -1.0 / sqrt(2);   T[2, 4] = 1.0 / sqrt(2)
            T[3, 1] = 1.0 / sqrt(2);   T[3, 3] = -1.0 / sqrt(2)
            T[4, 1] = 1.0 / sqrt(2);   T[4, 3] = 1.0 / sqrt(2)
        else:
            cubic_names = ("xy", "yz", "z^2", "xz", "x^2-y^2")
            T[0, 0] = 1j / sqrt(2);    T[0, 4] = -1j / sqrt(2)
            T[1, 1] = 1j / sqrt(2);    T[1, 3] = 1j / sqrt(2)
            T[2, 2] = 1.0
            T[3, 1] = 1.0 / sqrt(2);   T[3, 3] = -1.0 / sqrt(2)
            T[4, 0] = 1.0 / sqrt(2);   T[4, 4] = 1.0 / sqrt(2)
    elif l == 3:
        cubic_names = ("x(x^2-3y^2)", "z(x^2-y^2)", "xz^2", "z^3", "yz^2", "xyz", "y(3x^2-y^2)")
        T[0, 0] = 1.0 / sqrt(2);    T[0, 6] = -1.0 / sqrt(2)
        T[1, 1] = 1.0 / sqrt(2);    T[1, 5] = 1.0 / sqrt(2)
        T[2, 2] = 1.0 / sqrt(2);    T[2, 4] = -1.0 / sqrt(2)
        T[3, 3] = 1.0
        T[4, 2] = 1j / sqrt(2);   T[4, 4] = 1j / sqrt(2)
        T[5, 1] = 1j / sqrt(2);   T[5, 5] = -1j / sqrt(2)
        T[6, 0] = 1j / sqrt(2);   T[6, 6] = 1j / sqrt(2)
    else: raise ValueError("spherical_to_cubic: implemented only for l=0,1,2,3")

    return np.matrix(T)

def TRIQS_angular_matrix_element(l, k, m1, m2, m3, m4):
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
    for q in range(-k,k+1):
        ang_mat_ele += three_j_symbol((l,-m1),(k,q),(l,m3))*three_j_symbol((l,-m2),(k,-q),(l,m4))*(-1.0 if (m1+q+m2) % 2 else 1.0)
    ang_mat_ele *= (2*l+1)**2 * (three_j_symbol((l,0),(k,0),(l,0))**2)
    return ang_mat_ele



# The interaction matrix in desired basis
# U^{spherical}_{m1 m2 m3 m4} = \sum_{k=0}^{2l} F_k angular_matrix_element(l, k, m1, m2, m3, m4)
# H = \frac{1}{2} \sum_{ijkl,\sigma \sigma'} U_{ijkl} a_{i \sigma}^\dagger a_{j \sigma'}^\dagger a_{l \sigma'} a_{k \sigma}.
def TRIQS_U_matrix(l, radial_integrals=None, U_int=None, J_hund=None, basis='spherical', T=None):
    r"""
    Calculate the full four-index U matrix being given either radial_integrals or U_int and J_hund.
    The convetion for the U matrix is that used to construct the Hamiltonians, namely:

    .. math:: H = \frac{1}{2} \sum_{ijkl,\sigma \sigma'} U_{ijkl} a_{i \sigma}^\dagger a_{j \sigma'}^\dagger a_{l \sigma'} a_{k \sigma}.

    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated (l=2 for d shell, l=3 for f shell).
    radial_integrals : list, optional
                       Slater integrals [F0,F2,F4,..].
                       Must be provided if U_int and J_hund are not given.
                       Preferentially used to compute the U_matrix if provided alongside U_int and J_hund.
    U_int : scalar, optional
            Value of the screened Hubbard interaction.
            Must be provided if radial_integrals are not given.
    J_hund : scalar, optional
             Value of the Hund's coupling.
             Must be provided if radial_integrals are not given.
    basis : string, optional
            The basis in which the interaction matrix should be computed.
            Takes the values

            - 'spherical': spherical harmonics,
            - 'cubic': cubic harmonics,
            - 'other': other basis type as given by the transformation matrix T.

    T : real/complex numpy array, optional
        Transformation matrix for basis change.
        Must be provided if basis='other'.
        The transformation matrix is defined such that new creation operators :math:`b^\dagger` are related to
        the old ones :math:`a^\dagger` as

        .. math:: b_{i \sigma}^\dagger = \sum_j T_{ij} a^\dagger_{j \sigma}.


    Returns
    -------
    U_matrix : float numpy array
               The four-index interaction matrix in the chosen basis.

    """

    # Check all necessary information is present and consistent
    if radial_integrals is None and (U_int is None and J_hund is None):
        raise ValueError("U_matrix: provide either the radial_integrals or U_int and J_hund.")
    if radial_integrals is None and (U_int is not None and J_hund is not None):
        radial_integrals = U_J_to_radial_integrals(l, U_int, J_hund)
    if radial_integrals is not None and (U_int is not None and J_hund is not None):
        if len(radial_integrals)-1 != l:
            raise ValueError("U_matrix: inconsistency in l and number of radial_integrals provided.")
        if (radial_integrals - U_J_to_radial_integrals(l, U_int, J_hund)).any() != 0.0:
            print("Warning: U_matrix: radial_integrals provided do not match U_int and J_hund. Using radial_integrals to calculate U_matrix.")

    # Full interaction matrix
    # Basis of spherical harmonics Y_{-2}, Y_{-1}, Y_{0}, Y_{1}, Y_{2}
    # U^{spherical}_{m1 m2 m3 m4} = \sum_{k=0}^{2l} F_k angular_matrix_element(l, k, m1, m2, m3, m4)
    U_matrix = np.zeros((2*l+1,2*l+1,2*l+1,2*l+1),dtype=float)

    m_range = range(-l,l+1)
    for n, F in enumerate(radial_integrals):
        k = 2*n
        for m1, m2, m3, m4 in product(m_range,m_range,m_range,m_range):
            U_matrix[m1+l,m2+l,m3+l,m4+l] += F * TRIQS_angular_matrix_element(l,k,m1,m2,m3,m4)

    # Transform from spherical basis if needed
    if basis == "cubic": T = spherical_to_cubic(l)
    if basis == "other" and T is None:
        raise ValueError("U_matrix: provide T for other bases.")
    if T is not None: U_matrix = transform_U_matrix(U_matrix, T)

    return U_matrix

##############################################################################
#   Below are Routines Copy from Yongxin's pyglib.mbody.coulomb_matrix       #
##############################################################################


def unitary_transform_coulomb_matrix(a, u):
    '''Perform a unitary transformation (u) on the Coulomb matrix (a).
    '''
    a_ = np.asarray(a).copy()
    m_range = range(a.shape[0])
    for i, j in it.product(m_range, m_range):
        a_[i, j, :, :] = u.T.conj().dot(a_[i, j, :, :].dot(u))
    a_ = a_.swapaxes(0, 2).swapaxes(1, 3)
    for i, j in it.product(m_range, m_range):
        a_[i, j, :, :] = u.T.conj().dot(a_[i, j, :, :].dot(u))
    return a_


#def U_matrix(mode, l, radial_integrals=None, U_int=None, J_hund=None, T=None):
#    if 'slater' in mode:
#        U_matrix = U_matrix_slater(l, radial_integrals, U_int, J_hund)
#    elif 'kanamori' in mode:
#        U_matrix = U_matrix_kanamori(2 * l + 1, U_int, J_hund)
#    else:
#        raise NameError(" unsupported mode!")
#    u_avg, j_avg = get_average_uj(U_matrix)
#
#    # add spin-components
#    norb = U_matrix.shape[0]
#    norb2 = norb * 2
#    Ufull_matrix = np.zeros((norb2, norb2, norb2, norb2), dtype=np.complex128)
#    if T is not None:
#        # spin block
#        Ufull_matrix[:norb, :norb, :norb, :norb] = U_matrix
#        Ufull_matrix[norb:, norb:, norb:, norb:] = U_matrix
#        Ufull_matrix[:norb, :norb, norb:, norb:] = U_matrix
#        Ufull_matrix[norb:, norb:, :norb, :norb] = U_matrix
#        print(" u-matrix: nnz in compl_sph_harm = {}".format(
#                np.count_nonzero(np.abs(Ufull_matrix) > 1.e-10)))
#        Ufull_matrix = unitary_transform_coulomb_matrix(Ufull_matrix, T)
#    else:  # spin index fast
#        Ufull_matrix[::2, ::2, ::2, ::2] = U_matrix  # up, up
#        Ufull_matrix[1::2, 1::2, 1::2, 1::2] = U_matrix  # dn, dn
#        Ufull_matrix[::2, ::2, 1::2, 1::2] = U_matrix  # up, dn
#        Ufull_matrix[1::2, 1::2, ::2, ::2] = U_matrix  # dn, up
#
#    print(" u-matrix: nnz in final basis = {}".format(
#            np.count_nonzero(np.abs(Ufull_matrix) > 1.e-10)))
#    return Ufull_matrix, u_avg, j_avg


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

##############################################################
#                  operators
##############################################################
pauli_matrix = {'x' : np.array([[0,1],[1,0]]),
                'y' : np.array([[0,-1j],[1j,0]]),
                'z' : np.array([[1,0],[0,-1]]),
                '+' : np.array([[0,2],[0,0]]),
                '-' : np.array([[0,0],[2,0]])}

def L_op(component, orb_names, basis='spherical', T=None):
    r"""
    Create a component of the orbital momentum vector operator.

    .. math::
        \hat L_{z,+,-} &= \sum_{ii'\sigma} a^\dagger_{i\sigma} L^{z,+,-}_{ii'} a_{i'\sigma},\\
        \hat L_x &= \frac{1}{2}(\hat L_+ + \hat L_-),\ \hat L_y = \frac{1}{2i}(\hat L_+ - \hat L_-),\\
        L^z_{ii'} &= i\delta_{i,i'}, \
        L^+_{ii'} = \delta_{i,i'+1}\sqrt{l(l+1)-i'(i'+1)}, \
        L^+_{ii'} = \delta_{i,i'-1}\sqrt{l(l+1)-i'(i'-1)}.

    Parameters
    ----------
    component : string
                Component to be created, one of 'x', 'y', 'z', '+', or '-'.
    orb_names : list of strings or int
                Names of the orbitals, e.g. [0,1,2] or ['t2g','eg'].
    basis : string, optional
            The basis in which the interaction matrix should be computed.
            Takes the values

            - 'spherical': spherical harmonics,
            - 'cubic': cubic harmonics (valid only for the integer orbital momenta, i.e. for odd sizes of orb_names),
            - 'other': other basis type as given by the transformation matrix T.

    T : real/complex numpy array, optional
        Transformation matrix for basis change.
        Must be provided if basis='other'.

    Returns
    -------
    L : Operator
        The component of the orbital momentum vector operator.

    """
    l = (len(orb_names)-1)/2.0
    L_melem_dict = {'z' : lambda m,mp: m if np.isclose(m,mp) else 0,
                    '+' : lambda m,mp: sqrt(l*(l+1)-mp*(mp+1)) if np.isclose(m,mp+1) else 0,
                    '-' : lambda m,mp: sqrt(l*(l+1)-mp*(mp-1)) if np.isclose(m,mp-1) else 0,
                    'x' : lambda m,mp: 0.5*(L_melem_dict['+'](m,mp) + L_melem_dict['-'](m,mp)),
                    'y' : lambda m,mp: -0.5j*(L_melem_dict['+'](m,mp) - L_melem_dict['-'](m,mp))}
    L_melem = L_melem_dict[component]
    orb_range = list(range(int(2*l+1)))
    L_matrix = np.array([[L_melem(o1-l,o2-l) for o2 in orb_range] for o1 in orb_range])

    # Transform from spherical basis if needed
    if basis == "cubic":
        if not np.isclose(np.mod(l,1),0):
            raise ValueError("L_op: cubic basis is only defined for the integer orbital momenta.")
        T = spherical_to_cubic(int(l))
    if basis == "other" and T is None: raise ValueError("L_op: provide T for other bases.")
    if T is not None: L_matrix = np.einsum("ij,jk,kl",np.conj(T),L_matrix,np.transpose(T))

    return L_matrix
