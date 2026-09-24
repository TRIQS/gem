'''Regression test of the ghost-GA cycle with SimpleED, two orbitals at B=1.

Kanamori interaction; the reference is the ``2o2_ci`` group of
``result_tests.h5``.
'''

import unittest

import scf
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.utilities import U_matrix_kanamori
import numpy as np
import h5py
from gem.solvers.simple_ed import SimpleED
import os


class test_hemb_2o2_simple_ed(unittest.TestCase):

    def test_scf_simple_ed(self):

        # 2 orbital with 2 spins, 1 bath per orbital, total 4
        B=1
        nimp = 4
        nbath = B * nimp
        ntot = nimp + nbath

        U, J = 1.2, 0.3
        nnom = 2.0
        eloc = np.zeros((nimp, nimp))
        tmp_e = -(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[0,0] = tmp_e
        eloc[1,1] = tmp_e
        eloc[2,2] = tmp_e
        eloc[3,3] = tmp_e

        # construct ek with semicircular DOS
        e_list = np.linspace(-1, 1, 1000)
        wks = np.sqrt(1 - e_list**2)
        wks /= np.sum(wks)
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e, 0.0], [0.0, 1.0*e]], dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        np.random.seed(1234)
        # random initial value for hybridization
        L0 = np.zeros((nbath//2,nbath//2), dtype=np.complex128)
        R0 = np.zeros((nbath//2,nimp//2), dtype=np.complex128)

        # Build Ltmp
        for ix in range(2, B + 1):  # Fortran: ix = 2, ngh4b+1
            stride = (ix - 1) * nimp//2  # Python uses 0-based indexing
            block = np.eye(nimp//2) * (1.2 - 0.4 ** ((ix - 2) // 2)) * ((-1.0) ** ix)
            L0[stride:stride + nimp//2, stride:stride + nimp//2] = block

        # Build Rtmp
        for ix in range(1, B+1):  # Fortran: ix = 1, ngh4b+1
            stride = (ix - 1) * nimp//2
            block = 0.9 * np.eye(nimp//2) * (0.4 ** (ix // 2))
            R0[ stride:stride + nimp//2 , :] = block

        R0 = np.kron(R0, np.eye(2))
        L0 = np.kron(L0, np.eye(2))


        Utensor = U_matrix_kanamori(nimp//2, U, J)

        edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
        lattice = Lattice(eks, wk_list=wks, verbose=4)
        fragment = Fragment(nimp, nbath, eloc, Utensor, edsolver, verbose=4,
                            R=R0, Lambda=L0)
        # the cycle oscillates for the first ~25 iterations before settling on
        # the symmetric fixed point, so itmax must be well above it
        _, docc, _ = scf.run_scf(lattice, fragment, itmax=100, mix=0.5,
                                 tol=1e-5, T=1e-3, spin_sym=True,
                                 orb_sym=False, verbose=True)

        name = "2o2_ci"
        with h5py.File(os.path.dirname(os.path.abspath(__file__)) + "/result_tests.h5", "r") as A:

            print("Compare docc")
            docc_true = [A[name]["docc"][str(i)][0] + 1j*A[name]["docc"][str(i)][1] for i in range(len(docc))]
            np.testing.assert_allclose(docc, docc_true, atol=1e-3)

            print("Compare denMat")
            denmat_true = A[name]["denMat"][...,0] + 1j*A[name]["denMat"][...,1]
            ref_denM_eval, ref_denM_evec = np.linalg.eig(denmat_true)
            idx = ref_denM_eval.argsort()[::-1]
            ref_denM_eval = ref_denM_eval[idx]

            test_denM_eval, test_denM_evec = np.linalg.eig(fragment.denMat)
            idx = test_denM_eval.argsort()[::-1]
            test_denM_eval = test_denM_eval[idx]

            np.testing.assert_allclose(test_denM_eval, ref_denM_eval, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
