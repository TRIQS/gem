#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.gdmft import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular
import numpy as np
from triqs_ghostGA.solvers.fed import Fed
import os


class test_hemb_ci_1o3(unittest.TestCase):

    def test_gdmft_fed(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        nimp, nbath, ntot = 2, 6, 8

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=2000).e_list
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

        U = 1.3
        eloc = np.zeros((nimp, nimp))
        eloc[0,0] = -U/2.
        eloc[1,1] = -U/2.

        Utensor = np.zeros((nimp, nimp, nimp, nimp))
        Utensor[0,0,1,1] = U
        Utensor[1,1,0,0] = U

        Ts = np.logspace(-2,0,26)

        # test Fed solver
        edsolver = Fed(ntot, nimp, nbath)
        gdmft = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, D=D0, Lambda_c=Lambda_c0, edsolver=edsolver)

        for T in Ts:
            gdmft.run_dmft(itmax=100, mix=0.3, tol=5e-5, beta=1./T,
                           silence=True, spin_pen=0.0, method='minimize')
    
            Z = gdmft.R.conj().T.dot(gdmft.R)
            docc = gdmft.edsolver.calc_double_occ(0,T)
            gdmft.compute_energy(beta=1./T,mu=0.0)
            Etot = gdmft.etot
            Ekin = gdmft.ekin
            E2loc = gdmft.E2loc
            Epot = gdmft.epot # local one + two-body energy 

            fo = open('T_Z_docc_Etot_Ekin_E2loc_Epot_U%.3f.dat'%(U),'a')
            print(T, Z[0,0].real, docc.real, Etot.real+U/2., Ekin.real, E2loc.real, Epot.real, file=fo)
            fo.close()

            oms = np.linspace(-5,5,200)
            gdmft.compute_Gf_Sig(0.0, eks, oms, 0.05)
            Gloc = np.sum(gdmft.Gf, axis=0)/gdmft.Gf.shape[0]
    
            np.savetxt('G_T%.3f.dat'%(T), np.vstack((oms, Gloc[:,0,0].real, Gloc[:,0,0].imag)).T)

        import matplotlib.pyplot as plt
        plt.plot(oms, -Gloc[:,0,0].imag/np.pi)
        plt.show()

if __name__ == '__main__':
    unittest.main()
