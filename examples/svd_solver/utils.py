import numpy as np
from numpy.random import uniform
from triqs_ghostGA.utility.e_list import EList_SemiCircular
from triqs.operators.util import U_matrix_kanamori as Umk

def set_gGA(nimp, nbath, U, nmesh=5000, R0=None, Lambda0=None):
    eloc = np.zeros((nimp, nimp))
    tmp_e = -U/2
    eloc[0,0] = tmp_e
    eloc[1,1] = tmp_e
    # construct ek with semicircular DOS
    e_list = EList_SemiCircular(nmesh=nmesh).e_list
    eks = []
    for e in e_list:
        tmp = np.array([[1.0*e]], dtype=np.complex128)
        tmp = np.kron(tmp,np.eye(2))
        eks.append(tmp)
    eks = np.array(eks)

    np.random.seed(1234)

    # random initial value for hybridization
    if R0 is None:
        R0 = np.array([np.random.uniform(-1, 0, size=nbath)]).T
    R0 = np.kron(R0, np.eye(2))

    if Lambda0 is None:
        Lambda0 = uniform(-1, 1, size=nbath**2).reshape((nbath, nbath))
    Lambda0 = np.kron(Lambda0, np.eye(2))

    ci_U = np.zeros((nimp, nimp, nimp, nimp))
    ci_U[1, 1, 0, 0] = U
    ci_U[0, 0, 1, 1] = U

    svd_U = Umk(nimp//2, U, 0, full_Uijkl=True)

    return eloc, eks, R0, Lambda0, ci_U, svd_U


def compute_Z(grisb, eks, dw_Z):
    grisb.compute_Gf_Sig(grisb.mu, eks, np.array([0, dw_Z]), dw_Z)
    dR = (grisb.Sig[1].real - grisb.Sig[0].real)/dw_Z
    Z = np.linalg.inv(np.eye(len(dR))-dR)[0][0]

    return Z
