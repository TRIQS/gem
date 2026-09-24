'''Multi-rank test of the lattice: the k-point distribution.

Run by ctest under ``mpirun``. Every k sum covers the contiguous block owned by
the rank and is closed by an allreduce; on one rank that block is the whole
range and the allreduce is a no-op. The reference is the same lattice built
with ``use_mpi=False``, run redundantly on this rank. Both fragments use a
serial solver, so the k split is the only thing that varies.
'''

import unittest

import numpy as np

import mpi_harness as H
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED

NIMP, B = 2, 1
NBATH, NTOT = NIMP * B, NIMP * (B + 1)
NK = 101          # not divisible by the rank counts, so the blocks are uneven
U, MU, T = 2.0, 1.0, 0.5
# a reduction re-associates the k sum, so the parallel result is not bit-identical
# to the serial one; a missing allreduce is off by O(1), far above this
ATOL = 1e-9


def make_lattice(use_mpi):
    e = np.linspace(-1, 1, NK)
    wks = np.sqrt(1 - e**2)
    wks /= wks.sum()
    eks = e[:, None, None] * np.eye(NIMP, dtype=np.complex128)
    return Lattice(eks, wk_list=wks, use_mpi=use_mpi)


def make_fragment():
    Utensor = np.zeros((NIMP,) * 4)
    Utensor[0, 0, 1, 1] = Utensor[1, 1, 0, 0] = U
    solver = SimpleED(NTOT, use_Ntot=True, use_Sz=True,
                      N_sector=None, Sz_sector=None,
                      solver_params={'use_mpi': False})   # isolate the k split
    return Fragment(NIMP, NBATH, np.zeros((NIMP, NIMP)), Utensor, solver,
                    verbose=0)


def converged(use_mpi):
    '''Three ghost-GA iterations, left in a state compute_functional accepts.'''
    lattice, frag = make_lattice(use_mpi), make_fragment()
    for _ in range(3):
        lattice.solve_qp([frag], T=T)
        frag.update_hybridization(T=T)
        frag.solve_impurity(MU, T=T)
        frag.update_self_energy(T=T)
    lattice.solve_qp([frag], T=T)       # compute_functional needs this first
    frag.solve_impurity(MU, T=T)
    return lattice, frag


@unittest.skipUnless(H.CAN_RUN, H.NEEDS_RANKS)
class test_mpi_lattice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.par_lat, cls.par = converged(True)
        cls.ser_lat, cls.ser = converged(False)

    def test_kpoints_are_actually_split(self):
        '''Otherwise everything below would agree for the wrong reason.'''
        self.assertLess(self.par_lat.k1 - self.par_lat.k0, NK)
        self.assertEqual((self.ser_lat.k0, self.ser_lat.k1), (0, NK))

    def test_reductions_match_serial(self):
        for name in ('Delta_p_tot', 'ERD_tot'):
            np.testing.assert_allclose(getattr(self.par_lat, name),
                                       getattr(self.ser_lat, name),
                                       atol=ATOL, rtol=0, err_msg=name)
        for name in ('Delta_qp', 'Lambda', 'R', 'Lambda_c', 'D'):
            np.testing.assert_allclose(getattr(self.par, name),
                                       getattr(self.ser, name),
                                       atol=ATOL, rtol=0, err_msg=name)

    def test_energies_match_serial(self):
        np.testing.assert_allclose(self.par_lat.compute_ekin([self.par], T=T),
                                   self.ser_lat.compute_ekin([self.ser], T=T),
                                   atol=ATOL, rtol=0)
        np.testing.assert_allclose(
            self.par_lat.compute_functional([self.par], T=T),
            self.ser_lat.compute_functional([self.ser], T=T), atol=ATOL, rtol=0)

    def test_fit_mu_matches_serial(self):
        '''The bisection runs on a reduced quantity: one branch for all ranks.'''
        np.testing.assert_allclose(
            self.par_lat.fit_mu(1.0, [self.par], T=T, mu_old=0.0, mode='imp'),
            self.ser_lat.fit_mu(1.0, [self.ser], T=T, mu_old=0.0, mode='imp'),
            atol=ATOL, rtol=0)

    def test_guarded_save_and_load(self):
        '''Rank 0 writes; the k split is redone per rank at load.'''
        path = 'test_mpi_lattice_tmp.h5'
        if H.RANK == 0:                   # the guard save_lattice requires
            self.par_lat.save_lattice(path)
        H.COMM.Barrier()

        back = Lattice.load_lattice(path)
        np.testing.assert_array_equal(back.eks, self.par_lat.eks)
        np.testing.assert_array_equal(back.wks, self.par_lat.wks)
        self.assertEqual((back.k0, back.k1), (self.par_lat.k0, self.par_lat.k1))
        H.COMM.Barrier()
        if H.RANK == 0:
            import os
            os.remove(path)


if __name__ == '__main__':
    H.run(__import__(__name__))
