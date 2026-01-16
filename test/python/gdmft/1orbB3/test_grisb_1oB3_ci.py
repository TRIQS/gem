#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
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
        #np.random.seed(123)
        #R0 = np.random.rand(nbath//2, nimp//2) + 0j
        #R0 = np.kron(R0, np.eye(2))

        #Lambda0 = np.zeros((nbath//2, nbath//2))
        #Lambda0 = np.diag([0.6, 0.6, 0, 0, -0.6, -0.6+0j])
        #Lambda0 = np.kron(Lambda0, np.eye(2))

        #D0 = R0*0.5
        #Lambda_c0 = Lambda0
        np.random.seed(1234)
        R0 = np.random.rand(nbath//2,nimp//2)*(0.2+0j)
        R0 = np.kron(R0,np.eye(2))
        Lambda0 = np.zeros((nbath//2,nbath//2),dtype=complex)
        Lambda0[0,0] = 1.0
        Lambda0[1,1] = 0.0
        Lambda0[2,2] =-1.0
        Lambda0 = np.kron(Lambda0,np.eye(2))
        D0 = np.random.rand(nbath//2,nimp//2)
        D0 = np.kron(D0,np.eye(2))
        Lambda_c0 = Lambda0

  
        print('R0=')
        print(R0)
        print('Lambda0=')
        print(Lambda0)
        print('D0=')
        print(D0)
        print('Lambda_c0=')
        print(Lambda_c0)

        U = 1.5
        eloc = np.zeros((nimp, nimp))
        tmp_e = -U/2.
        eloc[0,0] = tmp_e
        eloc[1,1] = tmp_e

        Utensor = U_matrix_kanamori(nimp//2, U, 0)

        # test CI solver
        edsolver = CI(ntot, use_Ntot=True,
                      use_Sz=True, dtype=np.complex128)
        gdmft = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, edsolver=edsolver)
        gdmft.run(itmax=100, mix=0.3, tol=1e-5, beta=500,
                       silence=True)#method='minimize')

        oms = np.linspace(-5,5,200)
        gdmft.compute_Gf_Sig(0.0, eks, oms, 0.05)
        Gloc = np.sum(gdmft.Gf, axis=0)/gdmft.Gf.shape[0]

        import matplotlib.pyplot as plt
        plt.plot(oms, -Gloc[:,0,0].imag/np.pi)
        plt.show()


if __name__ == '__main__':
    unittest.main()
