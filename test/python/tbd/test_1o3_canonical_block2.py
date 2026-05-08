#!/usr/bin/env python

import unittest
import numpy as np
import h5py
import os

from triqs_ghostGA.gdmft import *

try:
    from triqs_ghostGA.solvers.pyblock2 import *
    have_block2 = True
except ImportError:
    have_block2 = False

class test_hemb_1o3_block2(unittest.TestCase):

    @unittest.skipIf(not have_block2, reason="Block2 solver is not installed.")
    def test_gdmft_block2(self):

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
        e_list = np.linspace(-1, 1, 5001)
        wks = np.sqrt(1 - e_list**2)
        wks /= np.sum(wks)
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

        maxM = 300
        edsolver = Pyblock2_N_SZ(ntot, nimp, nbath, maxM, spin_pen=0.0)

        grisb = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, wks=wks, R=R0, Lambda=Lambda0, edsolver=edsolver)
        grisb.run(mu=-0.2, n_target=nfix, itmax=100, mix=1, tol=1e-5, beta=500, silence=True)

        name = "1o3_canonical_block2"
        # with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:

        #     print("Compare docc")
        #     np.testing.assert_allclose(grisb.docc, A[name]["docc"][()], atol=1e-3)

        #     print("Compare denMat")
        #     ref_denM_eval, ref_denM_evec = np.linalg.eig(A[name]["denMat"][()])
        #     idx = ref_denM_eval.argsort()[::-1]
        #     ref_denM_eval = ref_denM_eval[idx]

        #     test_denM_eval, test_denM_evec = np.linalg.eig(grisb.Fragment.denMat)
        #     idx = test_denM_eval.argsort()[::-1]
        #     test_denM_eval = test_denM_eval[idx]

        #     np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)

        #     print("Compare mu")
        #     np.testing.assert_allclose(grisb.mu, A[name]["mu"][()], atol=1e-2)

        with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "a") as A:
            grp = A.require_group(name)
            grp["docc"] = grisb.docc
            grp["denMat"] = grisb.Fragment.denMat
            grp["mu"] = grisb.mu


if __name__ == '__main__':
    unittest.main()
