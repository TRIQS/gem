#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular
from h5 import *
import numpy as np
from triqs.utility import mpi
from triqs.lattice.tight_binding import TBLattice
from triqs.gf import *
from triqs.lattice import *
from triqs.operators import *
from triqs_ghostGA.version import *


class test_basic_features(unittest.TestCase):

    # Basic test just loading the lattice solver
    def test_loading(self):

        BL = BravaisLattice(units=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        BZ = BrillouinZone(BL)
        nk = 10
        mk = MeshBrZone(BZ, nk)
        ekup = Gf(mesh=mk, target_shape=[1, 1])
        ekdn = Gf(mesh=mk, target_shape=[1, 1])

        for k in mk:
            ekup[k][0, 0] = 1*(np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]))
            ekdn[k][0, 0] = 1*(np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]))

        h0_k = BlockGf(name_list=['up', 'down'], block_list=[ekup, ekdn])

        gf_struct = [('up', 1), ('down', 1)]
        h_int = 3*n('up', 0)*n('down', 0)

        S = LatticeSolver(h0_k=h0_k, gf_struct=gf_struct)
        S.solve_ForkTPS(h_int=h_int)

        pass

    # Print version and hash
    def test_version_prints(self):
        show_version()
        show_git_hash()


if __name__ == '__main__':
    unittest.main()
