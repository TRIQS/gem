#!/usr/bin/env python

import unittest
import numpy as np

from gem.gdmft import *
#from gem.utility.e_list import EList_SemiCircular
from gem.solvers.simple_ed import SimpleED


def _make_grisb():
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
    return Gdmft(ntot, nimp, nbath, eks, eloc, Utensor, wks=wks, edsolver=edsolver)


N_TARGET = 0.85
N_TOLERANCE = 1e-3
RUN_KWARGS = dict(itmax=150, mix=0.2, tol=1e-5, T=2e-3, silence=True,
                  n_target=N_TARGET, n_tolerance=N_TOLERANCE)


class TestDensityFittingQP(unittest.TestCase):

    def test_density_fit_qp(self):
        grisb = _make_grisb()
        grisb.run(**RUN_KWARGS, n_fit_method='qp')
        np.testing.assert_allclose(
            grisb.nfill, N_TARGET, atol=N_TOLERANCE,
            err_msg="QP density fitting did not converge to n_target"
        )


class TestDensityFittingImp(unittest.TestCase):

    def test_density_fit_imp(self):
        grisb = _make_grisb()
        grisb.run(**RUN_KWARGS, n_fit_method='imp')
        np.testing.assert_allclose(
            grisb.nfill, N_TARGET, atol=N_TOLERANCE,
            err_msg="Impurity density fitting did not converge to n_target"
        )


if __name__ == '__main__':
    unittest.main()
