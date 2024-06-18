#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular
from triqs.operators.util import U_matrix_kanamori as Umk
import numpy as np
from triqs_ghostGA.version import *
#from triqs_ghostGA.mps import ITensorMPSSolver
from triqs_ghostGA.pyblock2_solver import *

class test_hemb_5o15_pyblock2(unittest.TestCase):

    def test_grisb_mps(self):

        # 2 orbital with 2 spins, 3 bath per orbital, total 16
        nimp, nbath, ntot = 10, 30, 40

        U, J = 1.2, 0.3
        nnom = 2.0
        eloc = np.zeros((nimp, nimp))
        tmp_e = -(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[0,0] = tmp_e
        eloc[1,1] = tmp_e
        eloc[2,2] = tmp_e
        eloc[3,3] = tmp_e

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=5000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[ 1.0*e, 0.0  ,   0.0,   0.0,   0.0],
                            [   0.0, 1.0*e,   0.0,   0.0,   0.0],
                            [   0.0,   0.0, 1.0*e,   0.0,   0.0],
                            [   0.0,   0.0,   0.0, 1.0*e,   0.0],
                            [   0.0,   0.0,   0.0,   0.0, 1.0*e]], dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        np.random.seed(1234)
        # random initial value for hybridization
        R0 = np.random.rand(nbath//2, nimp//2)
        R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.zeros((nbath//2, nbath//2))
        Lambda0[0, 0], Lambda0[1, 1], Lambda0[2, 2], Lambda0[3, 3], Lambda0[4, 4] = 0.5, 0.5, 0.5, 0.5, 0.5
        Lambda0[5, 5], Lambda0[6, 6], Lambda0[7, 7], Lambda0[8, 8], Lambda0[9, 9] = 0.0, 0.0, 0.0, 0.0, 0.0
        Lambda0[10, 10], Lambda0[11, 11], Lambda0[12, 12], Lambda0[13, 13], Lambda0[14, 14] = -0.5, -0.5, -0.5, -0.5, -0.5
        Lambda0 = np.kron(Lambda0, np.eye(2))

        Utensor = U_matrix_kanamori(nimp//2, U, J)

        solver = Pyblock2_N_SZ(ntot, nimp, nbath, 800)

        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, edsolver=solver)
        grisb.run(itmax=100, mix=50, tol=1e-3, beta=500, silence=True, spin_pen=0.1)

        name = "5o15_pyblock2"
        with HDFArchive("result_tests.h5", "r") as A:
            print("Compare denMat")
            ref_denM_eval, ref_denM_evec = np.linalg.eig(A[name]["denMat"])
            idx = ref_denM_eval.argsort()[::-1]
            ref_denM_eval = ref_denM_eval[idx]

            test_denM_eval, test_denM_evec = np.linalg.eig(grisb.denMat)
            idx = test_denM_eval.argsort()[::-1]
            test_denM_eval = test_denM_eval[idx]

            np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)

        # with HDFArchive("result_tests.h5", "a") as A:
        #     tmp_dir = {
        #         'denMat': grisb.denMat,
        #     }
        #     A[name] = tmp_dir


if __name__ == '__main__':
    unittest.main()
