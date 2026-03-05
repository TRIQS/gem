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

# -------------------------
# Random generators
# -------------------------
def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random unitary from QR of a complex Gaussian matrix."""
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph.conj()

def random_hermitian_bounded_spectrum(
    n: int, rng: np.random.Generator, eig_max: float
) -> np.ndarray:
    """Hermitian with eigenvalues in [-eig_max, +eig_max]."""
    U = random_unitary(n, rng)
    eig = rng.uniform(-eig_max, eig_max, size=n)
    return U @ np.diag(eig) @ U.T.conj()

def random_complex(shape, rng: np.random.Generator, scale: float) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * scale


class TestSelfEnergySolverB3_random_generated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.size = 1
        cls.B = 3
        cls.Bsize = cls.B * cls.size

        # beta=4 is mild, so we can be less strict than your beta=300 case
        cls.beta = 100.0

        # keep this small-ish to help the solver converge reliably
        cls.noise = 1e-3

    def setUp(self):
        rng = np.random.default_rng(4201)

        # -------------------------
        # Generate a consistent target problem
        # -------------------------
        # Target Lambda, Hermitian (Bsize x Bsize)
        self.Lambda_target = random_hermitian_bounded_spectrum(
            self.Bsize, rng, eig_max=1.5
        )

        # Target R, complex (Bsize x size) -> Bx1
        self.R_target = random_complex((self.Bsize, self.size), rng, scale=0.5)

        self.x_target = pack_params(self.Lambda_target, self.R_target)

        # Fixed Lambda_c (Hermitian) and D (complex)
        self.Lambda_c = random_hermitian_bounded_spectrum(
            self.Bsize, rng, eig_max=0.6
        )
        self.D = random_complex((self.Bsize, self.size), rng, scale=0.5)

        # -------------------------
        # Build target hybridization data from the target parameters
        # -------------------------
        H = build_H(self.Lambda_target, self.Lambda_c, self.D, self.R_target)
        Delta_target = F_of_H(H, self.beta).T

        self.D11_target = Delta_target[: self.Bsize, : self.Bsize]
        self.D22_target = Delta_target[self.Bsize :, self.Bsize :]

        D12_target = Delta_target[: self.Bsize, self.Bsize :]
        self.RTD12_target = self.R_target.T @ D12_target

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
        tot_res = float(np.sum(np.abs(residual0)))

        self.assertLess(
            tot_res,
            1e-10,
            msg=f"Residual at target point should be zero, got {tot_res:.3e}",
        )

        jac0 = jacobian_LR(
            self.x_target,
            self.beta,
            self.Lambda_c,
            self.D,
            self.D22_target,
            self.RTD12_target,
        )
        self.assertTrue(np.all(np.isfinite(jac0)), "Jacobian at target should be finite")

    # ------------------------------------------------------------------
    # Utility: noisy initial condition
    # ------------------------------------------------------------------
    def make_noisy_start(self):
        rng = np.random.default_rng(123)

        Lam_pert = random_hermitian_bounded_spectrum(self.Bsize, rng, eig_max=1.0)
        R_pert = random_complex((self.Bsize, self.size), rng, scale=1.0)

        Lambda_0 = self.Lambda_target + self.noise * Lam_pert
        R_0 = self.R_target + self.noise * R_pert
        return Lambda_0, R_0

    def _gauge_invariant_error(self, Lam_sol, R_sol) -> float:
        Lg_trg, Ut = np.linalg.eigh(self.Lambda_target)
        Lg_sol, Us = np.linalg.eigh(Lam_sol)

        Rg_trg = np.abs(Ut.T.conj() @ self.R_target)
        Rg_sol = np.abs(Us.T.conj() @ R_sol)

        return float(np.sum(np.abs(Rg_trg - Rg_sol)) + np.sum(np.abs(Lg_sol - Lg_trg)))

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
        res_norm = float(np.sum(np.abs(res)))
        self.assertLess(res_norm, 1e-6, msg=f"Residual too large: {res_norm:.3e}")

        error = self._gauge_invariant_error(Lam_sol, R_sol)
        self.assertLess(error, 1e-6, msg=f"Gauge-invariant error too large: {error:.3e}")

        print(f"[OK] Solver without derivatives converged (time={t1-t0:.2f}s, error={error:.2e})")

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
        res_norm = float(np.sum(np.abs(res)))
        self.assertLess(res_norm, 1e-8, msg=f"Residual too large: {res_norm:.3e}")

        error = self._gauge_invariant_error(Lam_sol, R_sol)
        self.assertLess(error, 1e-8, msg=f"Gauge-invariant error too large: {error:.3e}")

        print(f"[OK] Solver with derivatives converged (time={t1-t0:.2f}s, error={error:.2e})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
