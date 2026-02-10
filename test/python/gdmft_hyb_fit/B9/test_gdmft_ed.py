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
        nimp, nbath, ntot = 2, 18, 20

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=1000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        # random initial value for hybridization
        #np.random.seed(1234)
        R0 = np.ones((nbath//2,nimp//2))*0.5
        R0 = np.kron(R0,np.eye(2))
        Lambda0 = np.zeros((nbath//2,nbath//2),dtype=complex)
        Lambda0_diag = np.linspace(-2.0,2.0,nbath//2)
        print('Lambda0_diag=',Lambda0_diag)
        for i in range(Lambda0.shape[0]):
            Lambda0[i,i] = Lambda0_diag[i]
        Lambda0 = np.kron(Lambda0,np.eye(2))

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

        U = 2.0
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
        gdmft.run_dmft_hyb(itmax=200, mix=0.5, tol=1e-5, beta=500,
                       silence=True, spin_pen=0.05, num_eig=10)

        oms = np.linspace(-5,5,200)
        gdmft.compute_Gf_Sig(0.0, eks, oms, 0.05)
        Gloc = np.sum(gdmft.Gf, axis=0)/gdmft.Gf.shape[0]

        np.savetxt('G_U%.2f_ED.dat'%(U), np.vstack((oms, Gloc[:,0,0].real, Gloc[:,0,0].imag)).T)

        import matplotlib.pyplot as plt
        plt.plot(oms, -Gloc[:,0,0].imag/np.pi)
        plt.show()

if __name__ == '__main__':
    unittest.main()
