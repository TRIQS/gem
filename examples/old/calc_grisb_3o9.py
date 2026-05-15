import numpy as np

from triqs_ghostGA.mps import *
from triqs.operators.util import U_matrix_kanamori as Uijkl, U_matrix_slater
np.set_printoptions(suppress=True, precision=6)
from triqs_ghostGA.utils_TH import get_semicircle_e_list, U_matrix_kanamori
from triqs_ghostGA.grisb import *

import time

def promote_to_spinful(up,dn):
    assert up.shape == dn.shape
    full=np.zeros((up.shape[0]*2,up.shape[1]*2),dtype=up.dtype)
    full[::2,::2]=up
    full[1::2,1::2]=dn
    return full


if __name__ == "__main__":
    do_ci_calc = False
    import h5py
    nimp, nbath, ntot = 6, 18, 24

    U = 12.0
    J = 3
    mu = 5
    t = 1

    #Build quartic part
    Utype = "Slater"    #"Kanamori" or "Slater"
    if Utype == "Kanamori":
        try:
            Utensor = Uijkl(nimp//2, U_int=U, J_hund=J, full_Uijkl=True)
        except:
            Utensor = U_matrix_kanamori(nimp//2, U_int=U, J_hund=J, full_Uijkl=True)
        finally:
            print("Kanamori UTensor construction breaks. The 4-index functionality is only present in triqs version >= 3.2.")
    elif Utype == "Slater":
        Utensor = U_matrix_slater(nimp//2, U_int=U, J_hund=J)


    #Build quadratic part
    np.random.seed(1234)
    eloc_spinless = np.zeros((nimp//2, nimp//2))
    # eloc_spinless = np.random.rand(nimp//2,nimp//2)
    # eloc_spinless += 0.5*eloc_spinless.conjugate().T
    eloc = promote_to_spinful(eloc_spinless, eloc_spinless)
    print(eloc)

    e_list = EList_SemiCircular(nmesh=5000).e_list
    eks = []
    for e in e_list:
        tmp = np.array(e*np.eye(nimp//2), dtype=np.complex128)
        tmp = np.kron(tmp, np.eye(2))
        eks.append(tmp)
    eks = np.array(eks)
    print(eks[-1])

    R0 = np.random.rand(nbath//2, nimp//2)
    R0 = np.kron(R0, np.eye(2))

    Lambda0 = np.zeros((nbath//2, nbath//2))
    Lambda0[0, 0], Lambda0[1, 1], Lambda0[2, 2] = 0.1, 0.1, 0.1
    Lambda0[3, 3], Lambda0[4, 4], Lambda0[5, 5] = 0.0, 0.0, 0.0
    Lambda0[6, 6], Lambda0[7, 7], Lambda0[8, 8] = -0.1, -0.1, -0.1
    Lambda0 = np.kron(Lambda0, np.eye(2))

    t = time.time()
    solver = ITensorMPSSolver(ntot, nimp, nbath, params={"use_Sz":True, "use_Ntot":True, "spin_pen":0.00})
    grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, edsolver=solver)
    grisb.run(itmax=1000, mu=mu, mix=0.5, tol=1e-5, beta=500, silence=False, spin_pen=0.00)
    t2 = time.time()
    print("time for run: ", t2-t)

    denMat = solver.calc_density_matrix()
    print('denMat_mps=')
    print(denMat)
