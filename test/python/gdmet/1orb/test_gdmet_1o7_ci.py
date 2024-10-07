#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.gdmet import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular
import numpy as np
from triqs_ghostGA.solvers.ci import CI
import os


class test_hemb_ci_1o7(unittest.TestCase):

    def test_grisb_ci(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        nimp, nbath, ntot = 2, 14, 16

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=1000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        # random initial value for hybridization
        np.random.seed(12)
        R0 = np.random.rand(nbath//2, nimp//2)
        R0 = np.kron(R0, np.eye(2))
            
        # Diagonal initialization
        #Lambda0 = np.zeros((nbath//2, nbath//2))
        #Lambda0 = np.diag([1.8, 1.2, 0.8, 0, -0.8, -1.2, -1.8])
        #Lambda0 = np.kron(Lambda0, np.eye(2))
        # Random Hermitian initialization
        Lambda0 = np.random.rand(nbath//2,nbath//2)*3.0
        Lambda0 = (Lambda0 + Lambda0.conj().T)/2
        Lambda0 = np.kron(Lambda0,np.eye(2))

        #U = 4.0
        Us = np.arange(0.2, 4.2, 0.2)
        for U in Us:
            eloc = np.zeros((nimp, nimp))
            eloc[0,0] = -U/2.
            eloc[1,1] = -U/2.
    
            Utensor = np.zeros((nimp, nimp, nimp, nimp))
            Utensor[0,0,1,1] = U
            Utensor[1,1,0,0] = U
    
            # test CI solver
            edsolver = CI(ntot, use_Ntot=True,
                          use_Sz=True, dtype=np.complex128)
            grisb = Gdmet(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                          Lambda=Lambda0, edsolver=edsolver)
            grisb.run(itmax=100, mix=0.3, tol=1e-5, beta=1000,
                  silence=True, spin_pen=0.0, num_eig=3)

            Z = grisb.R.conj().T.dot(grisb.R)
            docc0 = grisb.docc[0]
            R0 = grisb.R
            Lambda0 = grisb.Lambda
            mu0 = grisb.mu

            Nom = 10
            oms = np.linspace(-0.1,0.1,Nom)
            grisb.compute_Gf_Sig(0.0, eks, oms, 0.005)
            # plot DOS
            #Gfk = grisb.Gf
            #print(Gfk.shape)    
            #Gf = np.sum(Gfk, axis=0)/Gfk.shape[0]
            #import matplotlib.pyplot as plt
            #plt.plot(oms, -Gf[:,0,0].imag/np.pi, 'b-')
            #plt.show()

            Z0test = 1/(1-(grisb.Sig[Nom//2+1,0,0]-grisb.Sig[Nom//2,0,0]).real/(oms[Nom//2+1]-oms[Nom//2]))
            print('U=', U, 'docc=', docc0, 'Z=', Z0test)
            fo = open('U_docc_Z_gdmet_1o7_ci.dat','a')
            print(U, docc0.real, Z0test, file=fo)
            fo.close()


        #name = "1o7_ci"

        #with HDFArchive(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:

        #    print("Compare docc")
        #    np.testing.assert_allclose(grisb.docc, A[name]["docc"], atol=1e-3)

        #    print("Compare denMat")
        #    ref_denM_eval, ref_denM_evec = np.linalg.eig(A[name]["denMat"])
        #    idx = ref_denM_eval.argsort()[::-1]
        #    ref_denM_eval = ref_denM_eval[idx]

        #    test_denM_eval, test_denM_evec = np.linalg.eig(grisb.denMat)
        #    idx = test_denM_eval.argsort()[::-1]
        #    test_denM_eval = test_denM_eval[idx]

        #    np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)

        # with HDFArchive(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "a") as A:
        #     tmp_dir = {
        #         'docc': grisb.docc,
        #         'denMat': grisb.denMat,
        #     }
        #     A[name] = tmp_dir


if __name__ == '__main__':
    unittest.main()
