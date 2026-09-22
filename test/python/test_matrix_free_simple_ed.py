#!/usr/bin/env python
"""
Compare the matrix-free SimpleED path against the stored-CSC one.

solver_params['matrix_free'] = True never builds the Hamiltonian, the
c^dag_i c_j operators or the spin operators; ARPACK gets a LinearOperator and
the observables are accumulated by numba kernels instead. The two paths share
the bit algebra in gem/solvers/utilities/simple_ed_matvec.py, so they must
agree to machine precision, not merely to physical accuracy.
"""
import unittest
import numpy as np

from gem.solvers.simple_ed import SimpleED
from gem.solvers.utilities.simple_ed_matvec import (
    EmbeddingHamiltonian, build_index_table, one_body_terms, two_body_terms)

from test_symmetries_simple_ed import _make_hemb_inputs, _solve


def _kanamori_inputs(U=1.2, J=0.3, seed=7):
    """2 orbitals x 2 spins with a full Kanamori interaction.

    Density-density alone leaves the spin-flip / pair-hopping terms of
    two_body_target untested, and those are the ones with non-trivial signs.
    """
    from gem.utilities import U_matrix_kanamori

    nimp, nbath = 4, 4
    ntot = nimp + nbath
    rng = np.random.default_rng(seed)

    V2E = U_matrix_kanamori(2, U, J)
    eloc = np.diag([-0.6, -0.6, -0.5, -0.5]).astype(np.complex128)
    Lambdac = np.diag(rng.normal(size=nbath)).astype(np.complex128)
    D = rng.normal(size=(nbath, nimp)).astype(np.complex128)
    return ntot, nimp, nbath, eloc, D, Lambdac, V2E


def _solve_gs(solver, nimp, eloc, D, Lambdac, V2E, T, mu=0.0):
    """_solve plus the two scalars Lattice.compute_functional reads."""
    dm, e1, e2, docc = _solve(solver, nimp, eloc, D, Lambdac, V2E, T, mu=mu)
    return dm, e1, e2, docc, solver.gs_ene, solver.Zpart


def _assert_same(self, ref, got, atol, tag):
    names = ('density matrix (real)', 'density matrix (imag)', 'E1loc',
             'E2loc', 'docc', 'gs_ene', 'Zpart')
    values = [(ref[0].real, got[0].real), (ref[0].imag, got[0].imag)]
    values += [(ref[i], got[i]) for i in range(1, 6)]
    for name, (a, b) in zip(names, values):
        np.testing.assert_allclose(
            a, b, atol=atol, err_msg=f"{tag}: {name} disagrees")


class TestMatrixFreeOperator(unittest.TestCase):
    """The LinearOperator must reproduce the stored matrix element by element."""

    def test_matvec_and_dense_match_stored(self):
        ntot, nimp, _, eloc, D, Lambdac, V2E = _make_hemb_inputs()

        # rank-local check on one sector: use_mpi=False so every rank owns it
        stored = SimpleED(ntot, use_Ntot=True, use_Sz=True, N_sector=4,
                          Sz_sector=0.0, dtype=np.complex128,
                          solver_params={'use_mpi': False})
        Ham_list = stored.build_Hemb(D, eloc, Lambdac, V2E, mu=0.0, debug=True)
        Ham_csc = Ham_list[0]

        basis = stored.basis_list[0]
        index = build_index_table(basis, ntot)
        op = EmbeddingHamiltonian(basis, index, ntot,
                                  one_body_terms(stored.h1e),
                                  two_body_terms(V2E))

        self.assertEqual(op.shape, Ham_csc.shape)
        np.testing.assert_allclose(op.to_dense(), Ham_csc.toarray(), atol=1e-14,
                                   err_msg="dense matrix-free H disagrees")

        rng = np.random.default_rng(0)
        for _ in range(5):
            v = (rng.normal(size=op.shape[1])
                 + 1j * rng.normal(size=op.shape[1]))
            np.testing.assert_allclose(op @ v, Ham_csc @ v, atol=1e-13,
                                       err_msg="matvec disagrees")

        # and with the binary-search fallback instead of the lookup table
        op_bs = EmbeddingHamiltonian(basis, build_index_table(basis, ntot,
                                                              max_norb=0),
                                     ntot, one_body_terms(stored.h1e),
                                     two_body_terms(V2E))
        np.testing.assert_allclose(op_bs.to_dense(), Ham_csc.toarray(),
                                   atol=1e-14,
                                   err_msg="binary-search fallback disagrees")


class TestMatrixFreeObservables(unittest.TestCase):
    """End-to-end: every quantity the rest of GEM reads back."""

    def _compare(self, inputs, T, atol=1e-10, **params):
        ntot, nimp, _, eloc, D, Lambdac, V2E = inputs
        out = []
        for matrix_free in (False, True):
            sp = dict(params)
            sp['matrix_free'] = matrix_free
            solver = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                              dtype=np.complex128, solver_params=sp)
            out.append(_solve_gs(solver, nimp, eloc, D, Lambdac, V2E, T))
        _assert_same(self, out[0], out[1], atol,
                     f"T={T}, params={params}")

    def test_density_density_T0(self):
        self._compare(_make_hemb_inputs(), T=0.0)

    def test_density_density_finite_T(self):
        self._compare(_make_hemb_inputs(), T=0.25)

    def test_kanamori_T0(self):
        self._compare(_kanamori_inputs(), T=0.0)

    def test_kanamori_finite_T(self):
        self._compare(_kanamori_inputs(), T=0.3)

    def test_arpack_branch(self):
        """dense_cutoff below the largest sector forces eigsh on the operator.

        num_eig must stay under dim-1 for the sectors that reach ARPACK, hence
        the cutoff of 25 rather than 0.
        """
        inputs = _make_hemb_inputs(B=4)
        self._compare(inputs, T=0.0, atol=1e-9,
                      dense_cutoff=25, num_eig=1, tol=0.0)
        self._compare(inputs, T=0.3, atol=1e-9,
                      dense_cutoff=25, num_eig=15, tol=0.0)

    def test_no_symmetry(self):
        """Full 2^ntot Fock space as a single block."""
        ntot, nimp, _, eloc, D, Lambdac, V2E = _make_hemb_inputs()
        out = []
        for matrix_free in (False, True):
            solver = SimpleED(ntot, use_Ntot=False, use_Sz=False,
                              dtype=np.complex128,
                              solver_params={'matrix_free': matrix_free})
            out.append(_solve_gs(solver, nimp, eloc, D, Lambdac, V2E, 0.25))
        _assert_same(self, out[0], out[1], 1e-10, "no symmetry")

    def test_binary_search_fallback(self):
        """mf_lookup_max_norb=0 drops the lookup table for np.searchsorted."""
        ntot, nimp, _, eloc, D, Lambdac, V2E = _make_hemb_inputs()
        out = []
        for params in ({'matrix_free': False},
                       {'matrix_free': True, 'mf_lookup_max_norb': 0}):
            solver = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                              dtype=np.complex128, solver_params=params)
            out.append(_solve_gs(solver, nimp, eloc, D, Lambdac, V2E, 0.25))
        _assert_same(self, out[0], out[1], 1e-10, "binary-search fallback")


class TestMatrixFreeUnsupported(unittest.TestCase):

    def setUp(self):
        self.inputs = _make_hemb_inputs()

    def _solver(self, **params):
        ntot = self.inputs[0]
        params['matrix_free'] = True
        return SimpleED(ntot, use_Ntot=True, use_Sz=True,
                        dtype=np.complex128, solver_params=params)

    def test_spin_penalty_rejected(self):
        _, _, _, eloc, D, Lambdac, V2E = self.inputs
        for key in ('spin_pen', 'sz_pen', 'sx_pen', 'sy_pen'):
            solver = self._solver(**{key: 0.5})
            with self.assertRaises(NotImplementedError):
                solver.build_Hemb(D, eloc, Lambdac, V2E)

    def test_full_spectrum_at_finite_T_rejected(self):
        """T>0 with num_eig unset needs the whole spectrum, which needs H."""
        _, _, _, eloc, D, Lambdac, V2E = self.inputs
        solver = self._solver(dense_cutoff=0)
        solver.build_Hemb(D, eloc, Lambdac, V2E)
        with self.assertRaises(ValueError):
            solver.solve_Hemb(T=0.25)

    def test_operator_builders_rejected(self):
        solver = self._solver()
        for name in ('build_denmat_op', 'build_S2_op'):
            with self.assertRaises(NotImplementedError):
                getattr(solver, name)()


if __name__ == '__main__':
    unittest.main()
