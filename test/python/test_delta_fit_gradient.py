import unittest
import numpy as np

from gem.utility.delta_fit import (
    pack_params,
    residual_LR,
    residual_LcD,
    jacobian_LR,
    jacobian_LcD,
    build_H,
    F_of_H,
)

# --- Configuration ---
size = 2
p = 1
beta = 200.0
n = size


class TestJacobianConsistency(unittest.TestCase):

    def setUp(self):
        np.random.seed(420)

        # --- Generate consistent random matrices ---
        self.D = np.random.randn(n, p) + 1j * np.random.randn(n, p)

        self.Lc = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        self.Lc = 0.5 * (self.Lc + self.Lc.conj().T)

        self.L = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        self.L = 0.5 * (self.L + self.L.conj().T)

        self.R = np.random.randn(n, p) + 1j * np.random.randn(n, p)


        self.x_LR = pack_params(self.L, self.R)
        self.x_LcD = pack_params(self.Lc, self.D)

        # --- Build targets ---
        H = build_H(self.L, self.Lc, self.D, self.R)
        Delta_trg = F_of_H(H, beta)

        self.F11_trg = Delta_trg[:n, :n]
        self.F22_trg = Delta_trg[n:, n:]
        F12_trg = Delta_trg[:n, n:]

        self.F12D_trg = F12_trg @ self.D
        self.RTF12_trg = self.R.T @ F12_trg

    # ------------------------------------------------------------------
    # Utility: finite-difference Jacobian
    # ------------------------------------------------------------------
    def finite_difference_jacobian(self, residual, x0, eps=1e-6):
        jac = []

        for i in range(len(x0)):
            x_p = np.array(x0, copy=True)
            x_m = np.array(x0, copy=True)
            x_p[i] += eps
            x_m[i] -= eps
            
            f_p = residual(x_p)
            f_m = residual(x_m)
            
            jac.append((f_p - f_m) / (2 * eps))
        
        return np.column_stack(jac)


    # ------------------------------------------------------------------
    # Test Jacobian wrt (Lambda, R)
    # ------------------------------------------------------------------
    def test_jacobian_LR(self):
        print("\n--- CHECK GRADIENT WITH RESPECT TO LAMBDA AND R ---")

        ana_jac = jacobian_LR(
            self.x_LR, beta, self.Lc, self.D, self.F22_trg, self.RTF12_trg
        )

        num_jac = self.finite_difference_jacobian(
            lambda x: residual_LR(
                x, beta, self.Lc, self.D, self.F22_trg, self.RTF12_trg
            ),
            self.x_LR,
        )

        try:
            np.testing.assert_allclose(
                ana_jac,
                num_jac,
                atol=1e-8,
                err_msg="Jacobian mismatch for (Lambda, R)",
            )
        except AssertionError:
            diff = np.abs(ana_jac - num_jac)
            bad_row, bad_col = np.unravel_index(np.argmax(diff), diff.shape)

            iu = np.triu_indices(n)
            iu_strict = np.triu_indices(n, k=1)
            n_re = len(iu[0])
            n_im = len(iu_strict[0])

            if bad_col < n_re:
                p_type = f"Lambda Real (element {bad_col})"
            elif bad_col < n_re + n_im:
                p_type = f"Lambda Imag (element {bad_col - n_re})"
            elif bad_col < n_re + n_im + n * p:
                p_type = f"R Real (element {bad_col - (n_re + n_im)})"
            else:
                p_type = f"R Imag (element {bad_col - (n_re + n_im + n * p)})"

            self.fail(
                f"\nLargest error at Row {bad_row}, Col {bad_col} ({p_type})\n"
                f"Analytical value: {ana_jac[bad_row, bad_col]:.6e}\n"
                f"Numerical value:  {num_jac[bad_row, bad_col]:.6e}"
            )

        print("[OK] Jacobian consistency test passed for (Lambda, R)")

    # ------------------------------------------------------------------
    # Test Jacobian wrt (Lambda_c, D)
    # ------------------------------------------------------------------
    def test_jacobian_LcD(self):
        print("\n--- CHECK GRADIENT WITH RESPECT TO LAMBDA_C AND D ---")

        ana_jac = jacobian_LcD(
            self.x_LcD, beta, self.L, self.R, self.F11_trg, self.F12D_trg
        )

        num_jac = self.finite_difference_jacobian(
            lambda x: residual_LcD(
                x, beta, self.L, self.R, self.F11_trg, self.F12D_trg
            ),
            self.x_LcD,
        )

        try:
            np.testing.assert_allclose(
                ana_jac,
                num_jac,
                atol=1e-8,
                err_msg="Jacobian mismatch for (Lambda_c, D)",
            )
        except AssertionError:
            diff = np.abs(ana_jac - num_jac)
            bad_row, bad_col = np.unravel_index(np.argmax(diff), diff.shape)

            iu = np.triu_indices(n)
            iu_strict = np.triu_indices(n, k=1)
            n_re = len(iu[0])
            n_im = len(iu_strict[0])

            if bad_col < n_re:
                p_type = f"Lambda_c Real (element {bad_col})"
            elif bad_col < n_re + n_im:
                p_type = f"Lambda_c Imag (element {bad_col - n_re})"
            elif bad_col < n_re + n_im + n * p:
                p_type = f"D Real (element {bad_col - (n_re + n_im)})"
            else:
                p_type = f"D Imag (element {bad_col - (n_re + n_im + n * p)})"

            self.fail(
                f"\nLargest error at Row {bad_row}, Col {bad_col} ({p_type})\n"
                f"Analytical value: {ana_jac[bad_row, bad_col]:.6e}\n"
                f"Numerical value:  {num_jac[bad_row, bad_col]:.6e}"
            )

        print("[OK] Jacobian consistency test passed for (Lambda_c, D)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
