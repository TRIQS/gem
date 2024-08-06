#!/usr/bin/env python

import unittest
import numpy as np
import os

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular

try:
    from triqs_ghostGA.solvers.pyblock2 import *
    have_block2 = True
except ImportError:
    have_block2 = False

class test_hemb_1o3_block2(unittest.TestCase):

    @unittest.skipIf(not have_block2, reason="Block2 solver is not installed.")
    def test_grisb_block2(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        nimp, nbath, ntot = 2, 6, 8

        U = 1.2
        nfix = 0.8
        nnom = 1.0
        eloc = np.zeros((nimp, nimp))
        tmp_e = -U/2
        eloc[0,0] = tmp_e
        eloc[1,1] = tmp_e

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=5000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]], dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        np.random.seed(1234)
        # random initial value for hybridization
        R0 = np.random.rand(nbath//2, nimp//2)
        R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.zeros((nbath//2, nbath//2))
        Lambda0[0, 0] = 0.1
        Lambda0[1, 1] = 0.0
        Lambda0[2, 2] = -0.1
        Lambda0 = np.kron(Lambda0, np.eye(2))

        Utensor = np.zeros((nimp, nimp, nimp, nimp))
        Utensor[1, 1, 0, 0] = U
        Utensor[0, 0, 1, 1] = U
        # Utensor = U_matrix_kanamori(nimp//2, U, 0)

        maxM = 300
        edsolver = Pyblock2_N_SZ(ntot, nimp, nbath, maxM)

        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, edsolver=edsolver)
        grisb.run(mu0=-0.2, nfix=nfix, itmax=100, mix=1, tol=1e-5, beta=500, silence=True, spin_pen=0.0, mu_tol=1e-8)

        name = "1o3_canonical_block2"
        # with HDFArchive(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:

        #     print("Compare docc")
        #     np.testing.assert_allclose(grisb.docc, A[name]["docc"], atol=1e-3)

        #     print("Compare denMat")
        #     ref_denM_eval, ref_denM_evec = np.linalg.eig(A[name]["denMat"])
        #     idx = ref_denM_eval.argsort()[::-1]
        #     ref_denM_eval = ref_denM_eval[idx]

        #     test_denM_eval, test_denM_evec = np.linalg.eig(grisb.denMat)
        #     idx = test_denM_eval.argsort()[::-1]
        #     test_denM_eval = test_denM_eval[idx]

        #     np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)

        #     print("Compare mu")
        #     np.testing.assert_allclose(grisb.mu, A[name]["mu"], atol=1e-2)

        with HDFArchive(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "a") as A:
            tmp_dir = {
                'docc': grisb.docc,
                'denMat': grisb.denMat,
                'mu': grisb.mu,
            }
            A[name] = tmp_dir


if __name__ == '__main__':
    unittest.main()
