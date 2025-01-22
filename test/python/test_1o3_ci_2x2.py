#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_TB_2D
import numpy as np
from triqs_ghostGA.solvers.ci import CI
import os

from triqs.lattice.tight_binding import TBLattice
from triqs.lattice import TightBinding
from triqs.lattice import BravaisLattice, BrillouinZone

class test_hemb_ci_1o3_2x2(unittest.TestCase):

    def test_grisb_ci(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        nimp, nbath, ntot = 2, 6, 8

        # construct ek with semicircular DOS
        t = -1.0

        H_r = TBLattice(
            units=[
                (1,0,0), # basis vector in the x-direction
                (0,1,0), # basis vector in the y-direction
            ],
            orbital_positions = [(0, 0, 0)],
            hoppings={
                (+1, 0)  : [[t]],   # hopping in the +x direction
                (-1, 0)  : [[t]],   # hopping in the -x direction
                (0, +1)  : [[t]],   # hopping in the +y direction
                (0, -1)  : [[t]],   # hopping in the -y direction
            })

        e_list = EList_TB_2D(H_r, n_fourier=50, n_k=200).e_list[:, 0, 0].real

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
                      use_Sz=True, dtype=np.complex128,
                      spin_pen=0.05)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, edsolver=edsolver)
        grisb.run(itmax=30, mix=1, tol=1e-5, beta=500, silence=True)

        name = "1o3_ci_2x2"


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

        with HDFArchive(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "a") as A:
            tmp_dir = {
                'docc': grisb.docc,
                'denMat': grisb.denMat,
            }
            A[name] = tmp_dir


if __name__ == '__main__':
    unittest.main()
