'''Round trip of ``Fragment.save_fragment`` / ``Fragment.load_fragment``.

The file holds ``nimp``, ``nbath``, ``eloc``, ``Utensor`` and the four parameter
matrices. The solver is not saved and must be supplied at load.
'''

import os
import tempfile
import unittest

import h5py
import numpy as np

from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED

NIMP, B, U = 2, 3, 2.0
NBATH, NTOT = NIMP * B, NIMP * (B + 1)
SAVED = ('eloc', 'Utensor', 'Lambda', 'R', 'Lambda_c', 'D')


def make_solver():
    return SimpleED(NTOT, use_Ntot=True, use_Sz=True,
                    N_sector=NTOT // 2, Sz_sector=0)


def make_fragment(complex_params=False):
    '''A fragment whose four matrices are all different.

    The constructor defaults are symmetric enough that a bug swapping, say,
    ``Lambda`` and ``Lambda_c`` would go unnoticed.
    '''
    eloc = np.array([[-U / 2, 0.1], [0.1, -U / 2]])
    Utensor = np.zeros((NIMP,) * 4)
    Utensor[0, 0, 1, 1] = Utensor[1, 1, 0, 0] = U

    rng = np.random.default_rng(1234)
    M, A = rng.normal(size=(NBATH, NBATH)), rng.normal(size=(NBATH, NBATH))
    Lambda = M + M.T
    Lambda_c = A + A.T + np.diag(np.arange(NBATH, dtype=float))
    R = rng.normal(size=(NBATH, NIMP))
    D = R + 1.0

    if complex_params:
        Lambda = Lambda + 1j * (M - M.T)          # still Hermitian
        R, D = R + 0.3j * A[:, :NIMP], D - 0.2j * A[:, :NIMP]

    return Fragment(NIMP, NBATH, eloc, Utensor, make_solver(),
                    Lambda=Lambda, R=R, Lambda_c=Lambda_c, D=D, verbose=0)


class test_save_load_fragment(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'fragment.h5')
        self.addCleanup(self.tmp.cleanup)

    def save_and_load(self, frag, solver=None):
        frag.save_fragment(self.path)
        return Fragment.load_fragment(self.path, solver or frag.solver)

    def test_roundtrip(self):
        '''Everything comes back unchanged, real or complex.'''
        for complex_params in (False, True):
            with self.subTest(complex=complex_params):
                frag = make_fragment(complex_params)
                back = self.save_and_load(frag)

                for name in ('nimp', 'nbath', 'ntot', 'Bgh'):
                    # ints, not numpy scalars: Fragment.__init__ type-checks them
                    self.assertIsInstance(getattr(back, name), int, name)
                    self.assertEqual(getattr(back, name), getattr(frag, name), name)

                for name in SAVED:
                    x, y = getattr(frag, name), getattr(back, name)
                    self.assertEqual(np.iscomplexobj(x), np.iscomplexobj(y), name)
                    # exact: nothing is recomputed on the way back
                    np.testing.assert_array_equal(y, x, err_msg=name)

    def test_saves_the_live_eloc(self):
        '''``eloc`` is written as it stands, not as it was at construction.

        The two-sublattice examples nudge it with a staggered field mid-loop.
        '''
        frag = make_fragment()
        frag.eloc = frag.eloc + 0.05 * np.diag([-1.0, 1.0])
        np.testing.assert_array_equal(self.save_and_load(frag).eloc, frag.eloc)

    def test_load_takes_the_solver_it_is_given(self):
        other = make_solver()
        back = self.save_and_load(make_fragment(), solver=other)
        self.assertIs(back.solver, other)

    def test_load_works_from_class_and_instance(self):
        '''``load_fragment`` is a staticmethod, so both spellings work.'''
        frag = make_fragment()
        frag.save_fragment(self.path)
        np.testing.assert_array_equal(
            Fragment.load_fragment(self.path, frag.solver).Lambda,
            frag.load_fragment(self.path, frag.solver).Lambda)

    def test_overwrite(self):
        frag = make_fragment()
        frag.save_fragment(self.path)
        with self.assertRaises(FileExistsError):
            frag.save_fragment(self.path, overwrite=False)

        frag.Lambda = frag.Lambda + 0.5 * np.eye(NBATH)   # the default replaces
        np.testing.assert_array_equal(self.save_and_load(frag).Lambda, frag.Lambda)

    def test_load_rejects_an_inconsistent_file(self):
        '''The validation in ``Fragment.__init__`` still applies at load.'''
        frag = make_fragment()
        frag.save_fragment(self.path)
        with h5py.File(self.path, 'r+') as f:
            del f['nbath']
            f['nbath'] = NBATH + 1          # no longer a multiple of nimp
        with self.assertRaises(ValueError):
            Fragment.load_fragment(self.path, frag.solver)

    def test_same_impurity_solution(self):
        '''Physics, not just bytes.'''
        frag = make_fragment()
        back = self.save_and_load(frag, solver=make_solver())
        frag.solve_impurity(0.0)
        back.solve_impurity(0.0)

        np.testing.assert_allclose(back.denMat, frag.denMat, atol=1e-12)
        np.testing.assert_allclose(back.nfill, frag.nfill, atol=1e-12)
        np.testing.assert_allclose(back.compute_energy(), frag.compute_energy(),
                                   atol=1e-12)
        np.testing.assert_allclose(back.compute_Z(), frag.compute_Z(), atol=1e-10)

    def test_restart_continues_the_scf_cycle(self):
        '''The warm-start use case: save mid-cycle, reload, keep iterating.'''
        e = np.linspace(-1, 1, 201)
        wks = np.sqrt(1 - e**2)
        wks /= wks.sum()
        eks = e[:, None, None] * np.eye(NIMP, dtype=np.complex128)

        def step(lattice, frag):
            lattice.solve_qp([frag], T=0.0, Tsmearing=1e-3)
            frag.update_hybridization(use_Sz=True)
            frag.solve_impurity(0.0)
            frag.update_self_energy(use_Sz=True)

        frag, lattice = make_fragment(), Lattice(eks, wk_list=wks)
        for _ in range(2):      # so the saved state is a generic point
            step(lattice, frag)

        back = self.save_and_load(frag, solver=make_solver())
        back_lattice = Lattice(eks, wk_list=wks)

        for _ in range(3):
            step(lattice, frag)
            step(back_lattice, back)

        for name in ('Lambda', 'R', 'Lambda_c', 'D'):
            np.testing.assert_allclose(getattr(back, name), getattr(frag, name),
                                       atol=1e-10, err_msg=name)
        np.testing.assert_allclose(back.denMat, frag.denMat, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
