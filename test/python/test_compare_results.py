#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular
import numpy as np
from triqs_ghostGA.ci import CI


class test_compare_results(unittest.TestCase):

    def test_1o3_halffilled(self):

        with HDFArchive("result_tests.h5", "r") as A:
            ci_docc = A["1o3_ci"]["docc"]
            ci_denM = A["1o3_ci"]["denMat"]
            ci_denM_eval, ci_denM_evec = np.linalg.eig(A["1o3_ci"]["denMat"])
            idx = ci_denM_eval.argsort()[::-1]
            ci_denM_eval = ci_denM_eval[idx]

            ci_diis_docc = A["1o3_ci_diis"]["docc"]
            ci_diis_denM = A["1o3_ci_diis"]["denMat"]
            ci_diis_denM_eval, ci_diis_denM_evec = np.linalg.eig(A["1o3_ci_diis"]["denMat"])
            idx = ci_diis_denM_eval.argsort()[::-1]
            ci_diis_denM_eval = ci_diis_denM_eval[idx]

            ftps_denM = A["1o3_ftps"]["denMat"]
            ftps_denM_eval, ftps_denM_evec = np.linalg.eig(A["1o3_ftps"]["denMat"])
            idx = ftps_denM_eval.argsort()[::-1]
            ftps_denM_eval = ftps_denM_eval[idx]

            mps_denM = A["1o3_mps"]["denMat"]
            mps_denM_eval, mps_denM_evec = np.linalg.eig(A["1o3_mps"]["denMat"])
            idx = mps_denM_eval.argsort()[::-1]
            mps_denM_eval = mps_denM_eval[idx]

        print("Comparing results across solvers for 1o3 half-filled test.")
        np.testing.assert_allclose(ci_docc, ci_diis_docc, atol=1e-3)
        np.testing.assert_allclose(ci_denM[:2, :2], ci_diis_denM[:2, :2], atol=1e-3)
        np.testing.assert_allclose(ci_denM[:2, :2], ftps_denM[:2, :2], atol=1e-3)
        np.testing.assert_allclose(ci_denM[:2, :2], mps_denM[:2, :2], atol=1e-3)

        np.testing.assert_allclose(ci_denM_eval, ci_diis_denM_eval, atol=1e-3)
        np.testing.assert_allclose(ci_denM_eval, ftps_denM_eval, atol=1e-3)
        np.testing.assert_allclose(ci_denM_eval, mps_denM_eval, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
