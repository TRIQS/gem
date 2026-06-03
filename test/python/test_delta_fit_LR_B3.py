import unittest
from pathlib import Path

import numpy as np

from gem.utility.delta_fit import (
    pack_params,
    residual_LR,
    jacobian_LR,
    build_H,
    F_of_H,
    solve_F_dF_LR_with_movement,
)

# --- Configuration ---
size = 1
B = 3
Bsize = int(B * size)
beta = 100
noise = 3e-3
data_dir = Path(__file__).parent / "delta_fit" / "input_data" / "B3"


class TestSelfEnergySolverB3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not data_dir.exists():
            raise unittest.SkipTest(f"Missing test data directory: {data_dir}")

    def setUp(self):
        np.random.seed(4201)

        self.Lambda_target = np.loadtxt(data_dir / "lambda.real")
        self.Lambda_target = 0.5 * (self.Lambda_target + self.Lambda_target.T.conj())

        self.R_target = np.loadtxt(data_dir / "R.real").reshape((Bsize, size))

        self.x_target = pack_params(self.Lambda_target, self.R_target)

        self.Lambda_c = np.loadtxt(data_dir / "lambdac.real")
        self.Lambda_c = 0.5 * (self.Lambda_c + self.Lambda_c.T.conj())

        self.D = np.loadtxt(data_dir / "V.real").reshape((Bsize, size))

        H = build_H(self.Lambda_target, self.Lambda_c, self.D, self.R_target)
        Delta_target = F_of_H(H, beta).T

        self.D11_target = Delta_target[:Bsize, :Bsize]
        self.D22_target = Delta_target[Bsize:, Bsize:]
        D12_target = Delta_target[:Bsize, Bsize:]
        self.RTD12_target = self.R_target.T @ D12_target

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _gauge_invariant_error(self, Lam_sol, R_sol):
        Lg_trg, Ut = np.linalg.eigh(self.Lambda_target)
        Lg_sol, Us = np.linalg.eigh(Lam_sol)

        Rg_trg = np.abs(Ut.T.conj() @ self.R_target)
        Rg_sol = np.abs(Us.T.conj() @ R_sol)

        return float(np.sum(np.abs(Rg_trg - Rg_sol)) + np.sum(np.abs(Lg_sol - Lg_trg)))

    def _perturb_initial_guess(self):
        Lambda_pert = 2.0 * (-0.5 + np.random.rand(Bsize, Bsize)) + 2j * (-0.5 + np.random.rand(Bsize, Bsize))
        Lambda_pert = 0.5 * (Lambda_pert + Lambda_pert.T.conj())

        R_pert = 2.0 * (np.random.rand(Bsize, size) - 0.5) + 2j * (np.random.rand(Bsize, size) - 0.5)

        Lambda_0 = self.Lambda_target + noise * Lambda_pert
        R_0 = self.R_target + noise * R_pert
        return Lambda_0, R_0

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_target_is_stationary_point(self):
        res0 = residual_LR(self.x_target, beta, self.Lambda_c, self.D, self.D22_target, self.RTD12_target)
        jac0 = jacobian_LR(self.x_target, beta, self.Lambda_c, self.D, self.D22_target, self.RTD12_target)

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
        Lambda_0, R_0 = self._perturb_initial_guess()

        _, Lam_sol, R_sol = solve_F_dF_LR_with_movement(
            beta, self.Lambda_c, self.D, Lambda_0, R_0,
            self.D22_target, self.RTD12_target, alpha=1e-10, use_analytic_jac=False,
        )

        x_sol = pack_params(Lam_sol, R_sol)
        res = residual_LR(x_sol, beta, self.Lambda_c, self.D, self.D22_target, self.RTD12_target)

        self.assertLess(
            float(np.sum(np.abs(res))),
            1e-6,
            msg="Residual should be small after root finding (method='F')",
        )

        self.assertLess(
            float(self._gauge_invariant_error(Lam_sol, R_sol)),
            1e-5,
            msg="Solution should be close to target in gauge-invariant quantities (method='F')",
        )

    def test_root_finding_with_derivatives(self):
        Lambda_0, R_0 = self._perturb_initial_guess()

        _, Lam_sol, R_sol = solve_F_dF_LR_with_movement(
            beta, self.Lambda_c, self.D, Lambda_0, R_0,
            self.D22_target, self.RTD12_target, alpha=1e-10, use_analytic_jac=True,
        )

        x_sol = pack_params(Lam_sol, R_sol)
        res = residual_LR(x_sol, beta, self.Lambda_c, self.D, self.D22_target, self.RTD12_target)

        self.assertLess(
            float(np.sum(np.abs(res))),
            1e-6,
            msg="Residual should be small after root finding (method='dF')",
        )

        self.assertLess(
            float(self._gauge_invariant_error(Lam_sol, R_sol)),
            1e-5,
            msg="Solution should be close to target in gauge-invariant quantities (method='dF')",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
