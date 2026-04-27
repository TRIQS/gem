#!/usr/bin/env python

import unittest

from triqs_ghostGA.gdmft import *
from triqs_ghostGA.utility.e_list import EList_SemiCircular
import numpy as np
from triqs_ghostGA.solvers.ci import CI
import os
import time

import matplotlib.pyplot as plt

class test_hemb_ci_1o3(unittest.TestCase):

    def test_gdmft_ci(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        B = 3
        nimp = 2
        nbath = nimp*B
        ntot = nimp+nbath

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=5000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        Lambda0=None
        R0=None
        U_list=np.linspace(1.0,3.2,12)
        Z_list=[]
        for iU,U in enumerate(U_list):
            eloc = np.zeros((nimp, nimp))
            eloc[0,0] = -U/2.
            eloc[1,1] = -U/2.

            Utensor = np.zeros((nimp, nimp, nimp, nimp))
            Utensor[0,0,1,1] = U
            Utensor[1,1,0,0] = U

            # test CI solver
            edsolver = CI(ntot, use_Ntot=True, spin_pen=0.1,
                        use_Sz=True, dtype=np.complex128)
            grisb = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, edsolver=edsolver, 
                          Lambda=Lambda0, R=R0 , spin_pen=0.1)
            grisb.run(itmax=100, mix=0.2, tol=1e-5, beta=500,
                    silence=True)
        
            Z = grisb.Fragment.compute_Z()

            print(f'Done with U={U} returning Z={np.diag(Z.real)}')
            time.sleep(1)

            Lambda0=grisb.Lambda.copy()
            R0=grisb.R.copy()
            Z_list.append(Z[0,0].real)

        plt.figure()
        plt.plot(U_list,Z_list)
        plt.savefig('Z_vs_U_B{B}.png',dpi=100)
        plt.show()




if __name__ == '__main__':
    unittest.main()
