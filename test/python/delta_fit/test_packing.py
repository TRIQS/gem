import unittest
import numpy as np
from gem.utility.delta_fit import pack_params, unpack_params


class TestPackUnpackParams(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)

        self.m, self.n = 23, 7

        self.R = np.random.rand(self.m, self.n) + 1j * np.random.rand(self.m, self.n)
        L = np.random.rand(self.m, self.m) + 1j * np.random.rand(self.m, self.m)
        self.L = L + L.T.conj()  # Hermitian

    def test_pack_unpack_consistency(self):
        x = pack_params(self.L, self.R)
        Lu, Ru = unpack_params(x, self.m, self.n)

        np.testing.assert_allclose(
            Lu,
            self.L,
            atol=1e-12,
            err_msg="Unpacked L does not match original L"
        )

        np.testing.assert_allclose(
            Ru,
            self.R,
            atol=1e-12,
            err_msg="Unpacked R does not match original R"
        )

        # Success message (printed only if assertions pass)
        print(
            f"[OK] test_pack_unpack_consistency "
            f"(m={self.m}, n={self.n})"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
