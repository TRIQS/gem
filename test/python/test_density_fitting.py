'''The density fitting must reach the target filling.

Checked both for the quasiparticle problem and for the impurity problem.
'''

import unittest
import numpy as np

import scf
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED


def _make_system():
    B = 3
    nimp = 2
    nbath = nimp * B
    ntot = nimp + nbath

    e_list = np.linspace(-1, 1, 5001)
    wks =np.sqrt(1 - e_list**2)
    wks /= np.sum(wks)
    eks = []
    for e in e_list:
        tmp = np.array([[1.0 * e]], dtype=np.complex128)
        tmp = np.kron(tmp, np.eye(2))
        eks.append(tmp)
    eks = np.array(eks)

    U = 1.5
    eloc = np.zeros((nimp, nimp))
    eloc[0, 0] = -U / 2.
    eloc[1, 1] = -U / 2.

    Utensor = np.zeros((nimp, nimp, nimp, nimp))
    Utensor[0, 0, 1, 1] = U
    Utensor[1, 1, 0, 0] = U

    edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
    lattice = Lattice(eks, wk_list=wks, verbose=0)
    fragment = Fragment(nimp, nbath, eloc, Utensor, edsolver, verbose=0)
    return lattice, fragment


N_TARGET = 0.85
N_TOLERANCE = 1e-3
RUN_KWARGS = dict(itmax=150, mix=0.2, tol=1e-5, T=2e-3,
                  n_target=N_TARGET, n_tolerance=N_TOLERANCE)


class TestDensityFittingQP(unittest.TestCase):

    def test_density_fit_qp(self):
        lattice, fragment = _make_system()
        scf.run_scf(lattice, fragment, **RUN_KWARGS, n_fit_method='qp')
        np.testing.assert_allclose(
            np.trace(fragment.denMat[:fragment.nimp, :fragment.nimp]).real,
            N_TARGET, atol=N_TOLERANCE,
            err_msg="QP density fitting did not converge to n_target"
        )


class TestDensityFittingImp(unittest.TestCase):

    def test_density_fit_imp(self):
        lattice, fragment = _make_system()
        scf.run_scf(lattice, fragment, **RUN_KWARGS, n_fit_method='imp')
        np.testing.assert_allclose(
            np.trace(fragment.denMat[:fragment.nimp, :fragment.nimp]).real,
            N_TARGET, atol=N_TOLERANCE,
            err_msg="Impurity density fitting did not converge to n_target"
        )


if __name__ == '__main__':
    unittest.main()
