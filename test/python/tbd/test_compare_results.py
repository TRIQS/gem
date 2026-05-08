#!/usr/bin/env python

import unittest

from triqs_ghostGA.gdmft import *
import numpy as np
import h5py
import os


class test_compare_results(unittest.TestCase):

    def test_1o3_halffilled(self):

        with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:
            ci_docc = A["1o3_ci"]["docc"]['0'][0] + 1j*A["1o3_ci"]["docc"]['0'][1]
            ci_denM = A["1o3_ci"]["denMat"][...,0] + 1j*A["1o3_ci"]["denMat"][...,1]
            ci_denM_eval, ci_denM_evec = np.linalg.eig(ci_denM)
            idx = ci_denM_eval.argsort()[::-1]
            ci_denM_eval = ci_denM_eval[idx]

            ci_diis_docc = A["1o3_ci_diis"]["docc"]['0'][0] + 1j*A["1o3_ci_diis"]["docc"]['0'][1]
            ci_diis_denM = A["1o3_ci_diis"]["denMat"][...,0] + 1j*A["1o3_ci_diis"]["denMat"][...,1]
            ci_diis_denM_eval, ci_diis_denM_evec = np.linalg.eig(ci_diis_denM)
            idx = ci_diis_denM_eval.argsort()[::-1]
            ci_diis_denM_eval = ci_diis_denM_eval[idx]

            mps_denM = A["1o3_mps"]["denMat"][...,0] + 1j*A["1o3_mps"]["denMat"][...,1]
            mps_denM_eval, mps_denM_evec = np.linalg.eig(mps_denM)
            idx = mps_denM_eval.argsort()[::-1]
            mps_denM_eval = mps_denM_eval[idx]

        print("Comparing results across solvers for 1o3 half-filled test.")
        np.testing.assert_allclose(ci_docc, ci_diis_docc, atol=1e-3)
        np.testing.assert_allclose(ci_denM[:2, :2], ci_diis_denM[:2, :2], atol=1e-3)
        np.testing.assert_allclose(ci_denM[:2, :2], mps_denM[:2, :2], atol=1e-3)

        np.testing.assert_allclose(ci_denM_eval, ci_diis_denM_eval, atol=1e-3)
        np.testing.assert_allclose(ci_denM_eval, mps_denM_eval, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
