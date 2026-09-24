'''Round trip of ``Lattice.save_lattice`` / ``Lattice.load_lattice``.

The file holds ``ek_list``, ``wk_list`` and the verbosity. The MPI settings are
not saved and are given at load.
'''

import os
import tempfile
import unittest

import h5py
import numpy as np

from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.mpi import SerialComm
from gem.solvers.simple_ed import SimpleED

NIMP, NK = 2, 51


def make_lattice(verbose=0):
    '''Semicircular DOS, with weights that are deliberately not uniform.'''
    e = np.linspace(-1, 1, NK)
    wks = np.sqrt(1 - e**2)
    wks /= wks.sum()
    eks = e[:, None, None] * np.eye(NIMP, dtype=np.complex128)
    return Lattice(eks, wk_list=wks, verbose=verbose)


class test_save_load_lattice(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'lattice.h5')
        self.addCleanup(self.tmp.cleanup)

    def save_and_load(self, lat, **kwargs):
        lat.save_lattice(self.path)
        return Lattice.load_lattice(self.path, **kwargs)

    def test_roundtrip(self):
        '''Everything comes back unchanged.'''
        lat = make_lattice(verbose=1)
        back = self.save_and_load(lat)

        self.assertIsInstance(back.verb, int)
        self.assertEqual(back.verb, lat.verb)
        for name in ('eks', 'wks'):
            x, y = getattr(lat, name), getattr(back, name)
            self.assertEqual(np.iscomplexobj(x), np.iscomplexobj(y), name)
            np.testing.assert_array_equal(y, x, err_msg=name)

    def test_load_works_from_class_and_instance(self):
        '''``load_lattice`` is a staticmethod, so both spellings work.'''
        lat = make_lattice()
        lat.save_lattice(self.path)
        np.testing.assert_array_equal(Lattice.load_lattice(self.path).eks,
                                      lat.load_lattice(self.path).eks)

    def test_mpi_settings_are_given_at_load(self):
        '''They are not in the file, so the loader's arguments decide them.

        Checked on the communicator rather than on ``(k0, k1)``, which is the
        whole range either way when the test runs on a single rank.
        '''
        back = self.save_and_load(make_lattice(), use_mpi=False)
        self.assertIsInstance(back.comm, SerialComm)
        self.assertEqual((back.k0, back.k1), (0, NK))

    def test_overwrite(self):
        lat = make_lattice()
        lat.save_lattice(self.path)
        with self.assertRaises(FileExistsError):
            lat.save_lattice(self.path, overwrite=False)

        lat.eks = 2.0 * lat.eks                          # the default replaces
        np.testing.assert_array_equal(self.save_and_load(lat).eks, lat.eks)

    def test_load_rejects_an_inconsistent_file(self):
        '''The validation in ``Lattice.__init__`` still applies at load.'''
        make_lattice().save_lattice(self.path)
        with h5py.File(self.path, 'r+') as f:
            del f['wk_list']
            f['wk_list'] = np.ones(NK)                   # no longer summing to 1
        with self.assertRaises(ValueError):
            Lattice.load_lattice(self.path)

    def test_same_quasiparticle_solution(self):
        '''Physics, not just bytes: the reloaded lattice solves the same H_qp.'''
        lat = make_lattice()
        back = self.save_and_load(lat)

        nbath = NIMP * 3
        Utensor = np.zeros((NIMP,) * 4)
        Utensor[0, 0, 1, 1] = Utensor[1, 1, 0, 0] = 2.0
        solvers = [SimpleED(NIMP + nbath, use_Ntot=True, use_Sz=True,
                            N_sector=(NIMP + nbath) // 2, Sz_sector=0)
                   for _ in range(2)]
        frags = [Fragment(NIMP, nbath, np.diag([-1.0, -1.0]), Utensor, s, verbose=0)
                 for s in solvers]

        for lattice, frag in zip((lat, back), frags):
            lattice.solve_qp([frag], T=0.0, Tsmearing=1e-3)

        np.testing.assert_allclose(frags[1].Delta_qp, frags[0].Delta_qp, atol=1e-12)
        np.testing.assert_allclose(back.compute_ekin([frags[1]], T=0.0, Tsmearing=1e-3),
                                   lat.compute_ekin([frags[0]], T=0.0, Tsmearing=1e-3),
                                   atol=1e-12)


if __name__ == '__main__':
    unittest.main()
