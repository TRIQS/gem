'''Multi-rank test of the fragment: the SimpleED sector distribution.

Run by ctest under ``mpirun``. The solver splits its ``(N, Sz)`` sectors over
the ranks and reduces the ground state, the partition function and every
expectation value back; on one rank those reductions are no-ops, so this is the
only place they are exercised. The reference is the same fragment built with
``use_mpi=False``, solved redundantly on this rank. No lattice is involved:
the sector split is the only thing that varies.
'''

import unittest

import numpy as np

import mpi_harness as H
from gem.fragment import Fragment
from gem.solvers.simple_ed import SimpleED

NIMP, B = 2, 1
NBATH, NTOT = NIMP * B, NIMP * (B + 1)
U, MU, T = 2.0, 1.0, 0.5      # T > 0 and every sector kept, so all ranks work


def make_fragment(use_mpi):
    Utensor = np.zeros((NIMP,) * 4)
    Utensor[0, 0, 1, 1] = Utensor[1, 1, 0, 0] = U
    solver = SimpleED(NTOT, use_Ntot=True, use_Sz=True,
                      N_sector=None, Sz_sector=None,   # all sectors, to spread
                      solver_params={'use_mpi': use_mpi})
    return Fragment(NIMP, NBATH, np.zeros((NIMP, NIMP)), Utensor, solver,
                    verbose=0)


def solved(use_mpi):
    frag = make_fragment(use_mpi)
    frag.solve_impurity(MU, T=T)
    return frag


@unittest.skipUnless(H.CAN_RUN, H.NEEDS_RANKS)
class test_mpi_fragment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.par, cls.ser = solved(True), solved(False)

    def test_sectors_are_actually_split(self):
        '''Otherwise everything below would agree for the wrong reason.'''
        self.assertGreater(len(self.par.solver.sectors), H.SIZE)
        self.assertLess(len(self.par.solver.my_sectors),
                        len(self.par.solver.sectors))

    def test_reductions_match_serial(self):
        for name, atol in (('denMat', 1e-12), ('nfill', 1e-12)):
            np.testing.assert_allclose(getattr(self.par, name),
                                       getattr(self.ser, name),
                                       atol=atol, rtol=0, err_msg=name)
        for name in ('gs_ene', 'Zpart', 'Tstates', 'deg'):
            np.testing.assert_allclose(getattr(self.par.solver, name),
                                       getattr(self.ser.solver, name),
                                       atol=1e-12, rtol=0, err_msg=name)

    def test_observables_match_serial(self):
        np.testing.assert_allclose(self.par.compute_energy(),
                                   self.ser.compute_energy(), atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.par.solver.calc_double_occ(0),
                                   self.ser.solver.calc_double_occ(0),
                                   atol=1e-12, rtol=0)

    def test_guarded_save_and_load(self):
        '''Rank 0 writes, every rank reads back what it wrote.'''
        path = 'test_mpi_fragment_tmp.h5'
        frag = self.ser
        if H.RANK == 0:                   # the guard save_fragment requires
            frag.save_fragment(path)
        H.COMM.Barrier()

        back = Fragment.load_fragment(path, frag.solver)
        for name in ('Lambda', 'R', 'Lambda_c', 'D', 'eloc'):
            np.testing.assert_array_equal(getattr(back, name),
                                          getattr(frag, name), err_msg=name)
        H.COMM.Barrier()
        if H.RANK == 0:
            import os
            os.remove(path)


if __name__ == '__main__':
    H.run(__import__(__name__))
