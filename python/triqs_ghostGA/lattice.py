# Copyright (c) 2022 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Olivier Gingras and Tsung-Han Lee

import copy
import numpy as np
from triqs.gf import *
import triqs.utility.mpi as mpi
from h5.formats import register_class
from .utility.utils import *


class LatticeSolver(object):

    """ Ghost-Gutzwiller approximation Lattice solver for local interactions.

    Parameters
    ----------

    h0_k : TRIQS BlockGF on a Brillouin zone mesh
        Single-particle dispersion.

    """

    def __init__(self, h0_k, gf_struct):

        self.h0_k = h0_k.copy()
        self.gf_struct = gf_struct
        # self.rho = {bl: np.zeros((bl_size, bl_size)) for bl, bl_size in gf_struct}

        self.git_hash = "@PROJECT_GIT_HASH@"

    def solve_ForkTPS(self, h_int, N_target=None, mu=None, tol=None):
        """ Solve the embedded Hamiltonian using the ForkTPS solver.

        Parameters
        ----------

        h_int : TRIQS Operator instance
            Local interaction Hamiltonian

        N_target : optional, float
            target density per site. Can only be provided if mu is not provided

        mu: optional, float
            chemical potential. Can only be provided if N_target is not provided. Default is 0 if N_target is not provided

        tol: optional, float
            Convergence tolerance to pass to root finder

        """

    def __reduce_to_dict__(self):
        store_dict = {'h0_k': self.h0_k, 'n_k': self.n_k}
        if hasattr(self, 'mu'):
            store_dict['mu'] = self.mu
        if hasattr(self, 'last_solve_params'):
            params_copy = copy.deepcopy(self.last_solve_params)
            if params_copy['mu'] is None:
                params_copy['mu'] = 'none'
            if params_copy['tol'] is None:
                params_copy['tol'] = 'none'
            store_dict['last_solve_params'] = params_copy

        return store_dict

    @classmethod
    def __factory_from_dict__(cls, name, D):

        instance = cls(D['h0_k'])

        instance.n_k = D['n_k']
        instance.git_hash = D['git_hash']
        if 'mu' in D:
            instance.mu = D['mu']
        if 'last_solve_params' in D:
            params = copy.deepcopy(D['last_solve_params'])
            if params['mu'] == 'none':
                params['mu'] = None
            if params['tol'] == 'none':
                params['tol'] = None
            instance.last_solve_params = params
        return instance


register_class(LatticeSolver)
