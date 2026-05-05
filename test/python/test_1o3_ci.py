#!/usr/bin/env python

import unittest

from triqs_ghostGA.gdmft import *
import numpy as np
from triqs_ghostGA.solvers.ci import CI
import os


class test_hemb_ci_1o3(unittest.TestCase):

    def test_gdmft_ci(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        B = 3
        nimp = 2
        nbath = nimp*B
        ntot = nimp+nbath

        # construct ek with semicircular DOS
        e_list = np.linspace(-1, 1, 5001)
        wks = np.sqrt(1 - e_list**2)
        wks /= np.sum(wks)
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        # random initial value for hybridization
        R0 = np.random.rand(nbath//2, nimp//2)
        R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.zeros((nbath//2, nbath//2))
        Lambda0 = np.diag([0.6, 0, -0.6])
        Lambda0 = np.kron(Lambda0, np.eye(2))

        U = 2.4
        eloc = np.zeros((nimp, nimp))
        eloc[0,0] = -U/2.
        eloc[1,1] = -U/2.

        Utensor = np.zeros((nimp, nimp, nimp, nimp))
        Utensor[0,0,1,1] = U
        Utensor[1,1,0,0] = U

        # test CI solver
        edsolver = CI(ntot, use_Ntot=True,
                      use_Sz=True, dtype=np.complex128)
        grisb = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, wks=wks, edsolver=edsolver)
        grisb.run(itmax=30, mix=0.2, tol=1e-5, beta=500,
                  silence=True)

        name = "1o3_ci"


        with HDFArchive(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:

            print("Compare docc")
            np.testing.assert_allclose(grisb.docc, A[name]["docc"], atol=1e-3)

            print("Compare denMat")
            ref_denM_eval, ref_denM_evec = np.linalg.eig(A[name]["denMat"])
            idx = ref_denM_eval.argsort()[::-1]
            ref_denM_eval = ref_denM_eval[idx]

            test_denM_eval, test_denM_evec = np.linalg.eig(grisb.Fragment.denMat)
            idx = test_denM_eval.argsort()[::-1]
            test_denM_eval = test_denM_eval[idx]

            np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)

        # with HDFArchive(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "a") as A:
        #     tmp_dir = {
        #         'docc': grisb.docc,
        #         'denMat': grisb.denMat,
        #     }
        #     A[name] = tmp_dir


if __name__ == '__main__':
    unittest.main()
