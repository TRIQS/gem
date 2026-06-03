import unittest
from pathlib import Path

import numpy as np

from gem.utility.delta_fit import (
    pack_params,
    residual_LcD,
    jacobian_LcD,
    build_H,
    F_of_H,
    solve_F_dF_LcD_with_movement,
    update_hybridization_thermal_penalty,
)

# --- Configuration ---
size = 1
B = 3
Bsize = int(B * size)
beta = 100.0
noise = 1e-3
data_dir = Path(__file__).parent / "delta_fit" / "input_data" / "B3"


class TestFitLcDB3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not data_dir.exists():
            raise unittest.SkipTest(f"Missing test data directory: {data_dir}")

    def setUp(self):
        np.random.seed(4201)

        # --- Read reference solution (B=3, Norb=1) ---
        self.L = np.loadtxt(data_dir / "lambda.real")
        self.L = 0.5 * (self.L + self.L.T.conj())

        self.R = np.loadtxt(data_dir / "R.real").reshape((B, size))

        self.Lc_trg = np.loadtxt(data_dir / "lambdac.real")
        self.Lc_trg = 0.5 * (self.Lc_trg + self.Lc_trg.T.conj())

        self.D_trg = np.loadtxt(data_dir / "V.real").reshape((B, size))

        # --- Build targets ---
        H = build_H(self.L, self.Lc_trg, self.D_trg, self.R)
        Delta_trg = F_of_H(H, beta).T

        self.F11_trg = Delta_trg[:Bsize, :Bsize]
        F12_trg = Delta_trg[:Bsize, Bsize:]
        self.F12D_trg = F12_trg @ self.D_trg

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _gauge_invariant_error(self, Lc, D):
        eig_trg, U_trg = np.linalg.eigh(self.Lc_trg)
        eig_sol, U_sol = np.linalg.eigh(Lc)

        Dg_trg = np.abs(U_trg.T.conj() @ self.D_trg)
        Dg_sol = np.abs(U_sol.T.conj() @ D)

        return np.sum(np.abs(abs(Dg_trg) - abs(Dg_sol))) + np.sum(np.abs(eig_sol - eig_trg))

    def _perturb_initial_guess(self):
        Lc_pert = 2.0 * (-0.5 + np.random.rand(Bsize, Bsize)) + 2j * (-0.5 + np.random.rand(Bsize, Bsize))
        Lc_pert = 0.5 * (Lc_pert + Lc_pert.T.conj())

        D_pert = 2.0 * (np.random.rand(Bsize, size) - 0.5) + 2j * (np.random.rand(Bsize, size) - 0.5)

        Lc_0 = self.Lc_trg + noise * Lc_pert
        D_0 = self.D_trg + noise * D_pert
        return Lc_0, D_0

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_target_is_stationary_point(self):
        x_trg = pack_params(self.Lc_trg, self.D_trg)

        res0 = residual_LcD(x_trg, beta, self.L, self.R, self.F11_trg, self.F12D_trg)
        jac0 = jacobian_LcD(x_trg, beta, self.L, self.R, self.F11_trg, self.F12D_trg)

        self.assertLess(
            float(np.sum(np.abs(res0))),
            1e-8,
            msg="Residual at the target parameters should be ~0",
        )

        self.assertTrue(
            np.all(np.isfinite(jac0)),
            msg="Jacobian at the target parameters should be finite",
        )

    def test_root_finding_without_derivatives(self):
        Lc_0, D_0 = self._perturb_initial_guess()
        _, Lc_sol, D_sol = solve_F_dF_LcD_with_movement( beta, self.L,self.R, Lc_0, D_0,
                                                     self.F11_trg, self.F12D_trg, alpha=1e-10, use_analytic_jac=False )
 

        x_sol = pack_params(Lc_sol, D_sol)
        res = residual_LcD(x_sol, beta, self.L, self.R, self.F11_trg, self.F12D_trg)

        self.assertLess(
            float(np.sum(np.abs(res))),
            1e-6,
            msg="Residual should be small after root finding (method='F')",
        )

        self.assertLess(
            float(self._gauge_invariant_error(Lc_sol, D_sol)),
            1e-5,
            msg="Solution should be close to target in gauge-invariant quantities (method='F')",
        )

    def test_root_finding_with_derivatives(self):
        Lc_0, D_0 = self._perturb_initial_guess()
        _, Lc_sol, D_sol = solve_F_dF_LcD_with_movement( beta, self.L,self.R, Lc_0,D_0,
                                                     self.F11_trg, self.F12D_trg, alpha=1e-10, use_analytic_jac=True )
 

        x_sol = pack_params(Lc_sol, D_sol)
        res = residual_LcD(x_sol, beta, self.L, self.R, self.F11_trg, self.F12D_trg)

        self.assertLess(
            float(np.sum(np.abs(res))),
            1e-6,
            msg="Residual should be small after root finding (method='dF')",
        )

        self.assertLess(
            float(self._gauge_invariant_error(Lc_sol, D_sol)),
            1e-5,
            msg="Solution should be close to target in gauge-invariant quantities (method='dF')",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
