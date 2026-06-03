import unittest
import numpy as np

from gem.utility.delta_fit import (
    pack_params,
    residual_LcD,
    jacobian_LcD,
    build_H,
    F_of_H,
    solve_F_dF_LcD_with_movement,
)

# --- Configuration ---
size = 1
B = 3
Bsize = int(B * size)
beta = 100.0
noise = 1e-3


def random_hermitian(n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Random complex Hermitian matrix of shape (n, n)."""
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * scale + np.diag(np.linspace(-1.5,1.5,B,endpoint=True))
    return 0.5 * (A + A.T.conj())


def random_complex(shape, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Random complex array."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * scale + np.ones( (B,1) )/np.sqrt(B)


class TestFitLcDB3_random_generated(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(420)

        # --- Generate a random consistent target problem ---
        # Lambda (Hermitian B*size x B*size)
        self.L = random_hermitian(Bsize, rng, scale=0.1)

        # Lambda_c target (Hermitian B*size x B*size)
        self.Lc_trg = random_hermitian(Bsize, rng, scale=0.1)

        # R (complex B x size) -> for size=1, Bx1
        self.R = random_complex((B, size), rng, scale=0.1)

        # D target (complex B*size x size) -> for size=1, Bx1
        self.D_trg = random_complex((Bsize, size), rng, scale=0.1)

        # --- Build targets F11 and F12D from the target parameters ---
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

        return np.sum(np.abs(Dg_trg - Dg_sol)) + np.sum(np.abs(eig_sol - eig_trg))

    def _perturb_initial_guess(self):
        rng = np.random.default_rng(420)

        Lc_pert = random_hermitian(Bsize, rng, scale=0.1)
        D_pert = random_complex((Bsize, size), rng, scale=0.1)

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
            1e-10,
            msg="Residual at the target parameters should be ~0",
        )

        self.assertTrue(
            np.all(np.isfinite(jac0)),
            msg="Jacobian at the target parameters should be finite",
        )

    def test_root_finding_without_derivatives(self):
        Lc_0, D_0 = self._perturb_initial_guess()

        _, Lc_sol, D_sol = solve_F_dF_LcD_with_movement(
            beta, self.L, self.R, Lc_0, D_0, self.F11_trg, self.F12D_trg, alpha=1e-10, use_analytic_jac=False
        )

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

        _, Lc_sol, D_sol = solve_F_dF_LcD_with_movement(
            beta, self.L, self.R, Lc_0, D_0, self.F11_trg, self.F12D_trg, alpha=1e-10, use_analytic_jac=True
        )

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
