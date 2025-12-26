#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.gdmft import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular
import numpy as np
from triqs_ghostGA.solvers.ci import CI
import os


class test_hemb_ci_1o3(unittest.TestCase):

    def test_gdmft_ci(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        nimp, nbath, ntot = 2, 6, 8

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=5000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        # random initial value for hybridization
        np.random.seed(1234)
        R0 = np.random.rand(nbath//2, nimp//2) + 0j
        R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.zeros((nbath//2, nbath//2))
        Lambda0 = np.diag([0.6, 0, -0.6+0j])
        Lambda0 = np.kron(Lambda0, np.eye(2))

        D0 = R0*0.5
        Lambda_c0 = Lambda0
  
        print('R0=')
        print(R0)
        print('Lambda0=')
        print(Lambda0)
        print('D0=')
        print(D0)
        print('Lambda_c0=')
        print(Lambda_c0)

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
        gdmft = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, D=D0, Lambda_c=Lambda_c0, edsolver=edsolver)
        gdmft.run_dmft(itmax=50, mix=1, tol=1e-5, beta=500,
                       silence=True, spin_pen=0.05, method_min='BFGS')


if __name__ == '__main__':
    unittest.main()
