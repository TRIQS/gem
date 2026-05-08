#!/usr/bin/env python

import unittest

from triqs_ghostGA.gdmft import *
from triqs_ghostGA.utility.utilities import U_matrix_kanamori
import numpy as np
import h5py
from triqs_ghostGA.solvers.pyblock2 import *
import os


class test_hemb_5o15_pyblock2(unittest.TestCase):

    def test_gdmft_pyblock2(self):

        # 5 orbital with 2 spins, 3 bath per orbital, total 40
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
        e_list = np.linspace(-1, 1, 5001)
        wks = np.sqrt(1 - e_list**2)
        wks /= np.sum(wks)
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

        grisb = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, wks=wks, R=R0, Lambda=Lambda0,
                      edsolver=solver, spin_pen=0.1)
        grisb.run(itmax=100, mix=50, tol=1e-3, beta=500, silence=True)

        name = "5o15_pyblock2"
        with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:
            print("Compare denMat")
            denmat_true = A[name]["denMat"][...,0] + 1j*A[name]["denMat"][...,1]
            ref_denM_eval, ref_denM_evec = np.linalg.eig(denmat_true)
            idx = ref_denM_eval.argsort()[::-1]
            ref_denM_eval = ref_denM_eval[idx]

            test_denM_eval, test_denM_evec = np.linalg.eig(grisb.Fragment.denMat)
            idx = test_denM_eval.argsort()[::-1]
            test_denM_eval = test_denM_eval[idx]

            np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)

        # with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "a") as A:
        #     grp = A.require_group(name)
        #     grp["denMat"] = grisb.Fragment.denMat


if __name__ == '__main__':
    unittest.main()
