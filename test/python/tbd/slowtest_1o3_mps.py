#!/usr/bin/env python

import unittest

from triqs_ghostGA.gdmft import *
from triqs_ghostGA.utility.utilities import U_matrix_kanamori as Umk
import numpy as np
import h5py
from triqs_ghostGA.solvers.mps import ITensorMPSSolver as MPS
import os


class test_hemb_1o3_mps(unittest.TestCase):

    def test_gdmft_mps(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        nimp, nbath, ntot = 2, 6, 8

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

        Utensor = Umk(nimp//2, U, 0, full_Uijkl=True)

        # test julia MPS solver
        params={"use_Sz":True, "use_Ntot":True, "spin_pen":0.05}
        edsolver = MPS(ntot, nimp, nbath, params=params)
        grisb = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, wks=wks, R=R0,
                      Lambda=Lambda0, edsolver=edsolver, spin_pen=0.05)
        grisb.run(itmax=30, mix=1, tol=1e-5, beta=500, silence=True)

        name = "1o3_mps"
        # with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:

        #     print("Compare denMat")
        #     ref_denM_eval, ref_denM_evec = np.linalg.eig(A[name]["denMat"][()])
        #     idx = ref_denM_eval.argsort()[::-1]
        #     ref_denM_eval = ref_denM_eval[idx]

        #     test_denM_eval, test_denM_evec = np.linalg.eig(grisb.Fragment.denMat)
        #     idx = test_denM_eval.argsort()[::-1]
        #     test_denM_eval = test_denM_eval[idx]

        #     np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)

        with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "a") as A:
            grp = A.require_group(name)
            grp["denMat"] = grisb.Fragment.denMat


if __name__ == '__main__':
    unittest.main()
