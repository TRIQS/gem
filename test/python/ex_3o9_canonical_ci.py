#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utils_TH import get_semicircle_e_list, U_matrix_kanamori
import numpy as np
from triqs_ghostGA.version import *
from triqs_ghostGA.solvers.ci import CI


class test_hemb_3o9_ci(unittest.TestCase):

    def test_grisb_ci(self):

        # 2 orbital with 2 spins, 3 bath per orbital, total 16
        nimp, nbath, ntot = 6, 18, 24

        U, J = 3.0, 3.0/4.
        nfix = 2.0
        nnom = 2.0
        eloc = np.zeros((nimp, nimp))
        tmp_e = -1.3151383176669014#-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[0,0] = tmp_e
        eloc[1,1] = tmp_e
        eloc[2,2] = tmp_e
        eloc[3,3] = tmp_e
        eloc[4,4] = tmp_e
        eloc[5,5] = tmp_e

        # construct ek with semicircular DOS
        e_list = get_semicircle_e_list(nmesh=5000)
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e, 0.0, 0.0], [0.0, 1.0*e, 0.0], [0.0, 0.0, 1.0*e]], dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        np.random.seed(1234)
        # random initial value for hybridization
        R0 = np.random.rand(nbath//2, nimp//2)
        R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.zeros((nbath//2, nbath//2))
        Lambda0[0, 0], Lambda0[1, 1], Lambda0[2,2] = 0.1, 0.1, 0.1
        Lambda0[3, 3], Lambda0[4, 4], Lambda0[5,5] = 0.0, 0.0, 0.0
        Lambda0[6, 6], Lambda0[7, 7], Lambda0[8,8] = -0.1, -0.1, -0.1
        Lambda0 = np.kron(Lambda0, np.eye(2))


        Utensor = U_matrix_kanamori(nimp//2, U, J)

        edsolver=CI(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, edsolver=edsolver)
        grisb.run_canonical(mu0=0.0, nfix=nfix, itmax=100, mix=0.5, tol=5e-6, beta=500, silence=True, spin_pen=0.00, num_eig=2)

        docc0 = grisb.docc[0]
        docc1 = grisb.docc[1]
        Z = grisb.R.conj().T.dot(grisb.R)
        docc = grisb.docc
        R0 = grisb.R
        Lambda0 = grisb.Lambda

        # self.assertAlmostEqual(docc0.real , 0.1442486503727918, 4, 'incorrect double occupancy')


if __name__ == '__main__':
    unittest.main()
