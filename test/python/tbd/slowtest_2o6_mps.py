#!/usr/bin/env python

import unittest

from triqs_ghostGA.gdmft import *
from triqs_ghostGA.utility.utilities import U_matrix_kanamori as Umk
import numpy as np
import h5py
from triqs_ghostGA.solvers.mps import ITensorMPSSolver
import os


class test_hemb_2o6_mps(unittest.TestCase):

    def test_gdmft_mps(self):

        # 2 orbital with 2 spins, 3 bath per orbital, total 16
        nimp, nbath, ntot = 4, 12, 16

        U, J = 1.2, 0.3
        nnom = 2.0
        eloc = np.zeros((nimp, nimp))
        tmp_e = -(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[0,0] = tmp_e
        eloc[1,1] = tmp_e
        eloc[2,2] = tmp_e
        eloc[3,3] = tmp_e

        # construct ek with semicircular DOS
        e_list = np.linspace(-1, 1, 5001)
        wks = np.sqrt(1 - e_list**2)
        wks /= np.sum(wks)
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e, 0.0], [0.0, 1.0*e]], dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        np.random.seed(1234)
        # random initial value for hybridization
        R0 = np.random.rand(nbath//2, nimp//2)
        R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.zeros((nbath//2, nbath//2))
        Lambda0[0, 0], Lambda0[1, 1] = 0.2, 0.2
        Lambda0[2, 2], Lambda0[3, 3] = 0.0, 0.0
        Lambda0[4, 4], Lambda0[5, 5] = -0.2, -0.2
        Lambda0 = np.kron(Lambda0, np.eye(2))

        Utensor = Umk(nimp//2, U, J, full_Uijkl=True)

        solver = ITensorMPSSolver(ntot, nimp, nbath, params={"use_Sz":True,"use_Ntot":True,"spin_pen":0.05})
        solver.schedule=[]
        solver.add_to_schedule(nsweeps=2,maxdim=128,cutoff=1e-10,noise=1e-8)
        solver.add_to_schedule(nsweeps=3,maxdim=256,cutoff=1e-12,noise=1e-10)
        solver.add_to_schedule(nsweeps=2,maxdim=512,cutoff=1e-14,noise=0.0)
        solver.set_tolerances(tol_vals=(1e-6,1e-4))

        grisb = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, wks=wks, R=R0, Lambda=Lambda0,
                      edsolver=solver, spin_pen=0.1)
        grisb.run(itmax=100, mix=1, tol=1e-5, beta=500, silence=True)

        name = "2o6_mps"
        with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:
            print("Compare denMat")
            denmat_true = A[name]["denMat"][...,0] + 1j*A[name]["denMat"][...,1]
            ref_denM_eval, ref_denM_evec = np.linalg.eig(denmat_true)
            idx = ref_denM_eval.argsort()[::-1]
            ref_denM_eval = ref_denM_eval[idx]

            test_denM_eval, test_denM_evec = np.linalg.eig(grisb.Fragment.denMat)
            idx = test_denM_eval.argsort()[::-1]
            test_denM_eval = test_denM_eval[idx]

            np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)

        # with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "a") as A:
        #     grp = A.require_group(name)
        #     grp["denMat"] = grisb.Fragment.denMat


if __name__ == '__main__':
    unittest.main()
