import unittest
import numpy as np
import time

from triqs_ghostGA.utility.delta_fit import (
    pack_params,
    residual_LR,
    jacobian_LR,
    new_self_energy,
    build_H,
    F_of_H,
)


class TestSelfEnergySolverB3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # --- Configuration ---
        cls.size = 1
        cls.B = 3
        cls.Bsize = cls.B * cls.size
        cls.beta = 300
        cls.noise = 1e-2
        cls.fold_data = "input_data/B3"

        # --- Read reference solution ---
        cls.Lambda_target = np.loadtxt(f"{cls.fold_data}/lambda.real")
        cls.Lambda_target = 0.5 * (
            cls.Lambda_target + cls.Lambda_target.T.conj()
        )

        cls.R_target = np.loadtxt(
            f"{cls.fold_data}/R.real"
        ).reshape((cls.Bsize, cls.size))

        cls.x_target = pack_params(
            cls.Lambda_target, cls.R_target
        )

        cls.Lambda_c = np.loadtxt(f"{cls.fold_data}/lambdac.real")
        cls.Lambda_c = 0.5 * (
            cls.Lambda_c + cls.Lambda_c.T.conj()
        )

        cls.D = np.loadtxt(f"{cls.fold_data}/V.real").reshape(
            (cls.Bsize, cls.size)
        )

        # --- Build target hybridization ---
        H = build_H(
            cls.Lambda_target,
            cls.Lambda_c,
            cls.D,
            cls.R_target,
        )

        Delta_target = F_of_H(H, cls.beta).T

        cls.D11_target = Delta_target[:cls.Bsize, :cls.Bsize]
        cls.D22_target = Delta_target[cls.Bsize:, cls.Bsize:]
        D12_target = Delta_target[:cls.Bsize, cls.Bsize:]
        cls.RTD12_target = cls.R_target.T @ D12_target

    # ------------------------------------------------------------------
    # Test: reference point is exact root
    # ------------------------------------------------------------------
    def test_reference_residual_is_zero(self):
        residual0 = residual_LR(
            self.x_target,
            self.beta,
            self.Lambda_c,
            self.D,
            self.D22_target,
            self.RTD12_target,
        )

        tot_res = np.sum(np.abs(residual0))

        self.assertLess(
            tot_res,
            1e-10,
            msg=f"Residual at target point should be zero, got {tot_res:.3e}",
        )

        print("[OK] Reference solution is an exact root")

    # ------------------------------------------------------------------
    # Utility: noisy initial condition
    # ------------------------------------------------------------------
    def make_noisy_start(self):
        np.random.seed(123)

        Lambda_0 = 2.0 * (
            np.random.rand(self.Bsize, self.Bsize) - 0.5
            + 1j * (np.random.rand(self.Bsize, self.Bsize) - 0.5)
        )
        Lambda_0 = 0.5 * (Lambda_0 + Lambda_0.T.conj())

        R_0 = 2.0 * (
            np.random.rand(self.Bsize, self.size) - 0.5
            + 1j * (np.random.rand(self.Bsize, self.size) - 0.5)
        )

        Lambda_0 = self.Lambda_target + self.noise * Lambda_0
        R_0 = self.R_target + self.noise * R_0

        return Lambda_0, R_0

    # ------------------------------------------------------------------
    # Test: solver without derivatives
    # ------------------------------------------------------------------
    def test_solver_without_derivatives(self):
        Lambda_0, R_0 = self.make_noisy_start()

        t0 = time.time()
        Lam_sol, R_sol = new_self_energy(
            Lambda_0,
            R_0,
            self.Lambda_c,
            self.D,
            self.D22_target,
            self.RTD12_target,
            beta=self.beta,
            method="F",
        )
        t1 = time.time()

        x_sol = pack_params(Lam_sol, R_sol)
        res = residual_LR(
            x_sol,
            self.beta,
            self.Lambda_c,
            self.D,
            self.D22_target,
            self.RTD12_target,
        )

        res_norm = np.sum(np.abs(res))
        self.assertLess(res_norm, 1e-5)

        # Gauge-invariant comparison
        Lg_trg, Ut = np.linalg.eigh(self.Lambda_target)
        Lg_sol, Us = np.linalg.eigh(Lam_sol)
        Rg_trg = np.abs(Ut.T.conj() @ self.R_target)
        Rg_sol = np.abs(Us.T.conj() @ R_sol)

        error = (
            np.sum(np.abs(Rg_trg - Rg_sol))
            + np.sum(np.abs(Lg_sol - Lg_trg))
        )

        self.assertLess(error, 1e-5)

        print(
            f"[OK] Solver without derivatives converged "
            f"(time={t1-t0:.2f}s, error={error:.2e})"
        )

    # ------------------------------------------------------------------
    # Test: solver with derivatives
    # ------------------------------------------------------------------
    def test_solver_with_derivatives(self):
        Lambda_0, R_0 = self.make_noisy_start()

        t0 = time.time()
        Lam_sol, R_sol = new_self_energy(
            Lambda_0,
            R_0,
            self.Lambda_c,
            self.D,
            self.D22_target,
            self.RTD12_target,
            beta=self.beta,
            method="dF",
        )
        t1 = time.time()

        x_sol = pack_params(Lam_sol, R_sol)
        res = residual_LR(
            x_sol,
            self.beta,
            self.Lambda_c,
            self.D,
            self.D22_target,
            self.RTD12_target,
        )

        res_norm = np.sum(np.abs(res))
        self.assertLess(res_norm, 1e-5)

        Lg_trg, Ut = np.linalg.eigh(self.Lambda_target)
        Lg_sol, Us = np.linalg.eigh(Lam_sol)
        Rg_trg = np.abs(Ut.T.conj() @ self.R_target)
        Rg_sol = np.abs(Us.T.conj() @ R_sol)

        error = (
            np.sum(np.abs(Rg_trg - Rg_sol))
            + np.sum(np.abs(Lg_sol - Lg_trg))
        )

        self.assertLess(error, 1e-6)

        print(
            f"[OK] Solver with derivatives converged "
            f"(time={t1-t0:.2f}s, error={error:.2e})"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
