#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.gdmft import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular
import numpy as np
import time
from triqs_ghostGA.solvers.ci import CI
import os


class test_hemb_ci_1o3(unittest.TestCase):

    def test_gdmft_ci(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        B = 3
        nimp  = 2
        nbath = nimp*B
        ntot  = nimp+nbath

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=20000).e_list
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
        R0 = np.array([[0.8],[0.2],[0.2]])[:B,:]
        R0 = np.kron(R0,np.eye(2))
        Lambda0 = np.zeros((nbath//2,nbath//2),dtype=complex)

        if(B==3):
            Lambda0[0,0] = 0.0
            Lambda0[1,1] = 1.0
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

        
        U = 1.3
        J = U/4.
        eloc = np.zeros((nimp,nimp))
        Utensor = np.zeros((nimp,nimp,nimp,nimp))
        eloc[0,0] =-U/2.
        eloc[1,1] =-U/2.
        Utensor[0,0,1,1] = U
        Utensor[1,1,0,0] = U

        # test CI solver
        T_list=[]
        D_list=[]
        edsolver = CI(ntot, use_Ntot=True,
                      use_Sz=True, dtype=np.complex128, thermal=True)
        gdmft = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, D=D0, Lambda_c=Lambda_c0, edsolver=edsolver)
        
        for T in np.array([5,4,3,2,1,0.9,0.8,0.7,0.6,0.5,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.08,0.06,0.04,0.02,0.01,0.002]):
            print("")
            print(f" ***** Doing T={T} ***** ")
            time.sleep(2)
            gdmft.run_dmft(itmax=100, mix=1.0, tol=1e-4, beta=1.0/T, n_target=None,
                           silence=False, spin_pen=0.0, method='minimize')
            T_list.append(T)
            D_list.append(gdmft.docc)
            np.savetxt(f'Tlist_B{B}.dat',np.array(T_list).real)
            np.savetxt(f'Dlist_B{B}.dat',np.array(D_list).real)

        oms = np.linspace(-5,5,200)
        gdmft.compute_energy()
        print('docc:',gdmft.docc)
        print('Epot:',gdmft.epot)
        print('Ekin:',gdmft.ekin)
        print('Etot:',gdmft.etot)
        
        gdmft.compute_Gf_Sig(0.0, eks, oms, 0.05)
        Gloc = np.sum(gdmft.Gf, axis=0)/gdmft.Gf.shape[0]

        import matplotlib.pyplot as plt
        plt.plot(oms, -Gloc[:,0,0].imag/np.pi)
        plt.show()


if __name__ == '__main__':
    unittest.main()
