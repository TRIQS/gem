# Solver based on EDIpack, a generic and interoperable high-performance Lanczos-based
# ED solver for quantum impurity problems ( https://github.com/EDIpack/EDIpack )
#
# When using this solver please cite the following articles:
# - L.Crippa et al, SciPost Phys. Codebases 58 (2025)
# - A.Amarcci et al, Computer Physics Communications, Volume 273, 108261 (2022)

from edipack2triqs.solver import EDIpackSolver, LanczosParams
import mpi4py
from mpi4py import MPI
import numpy




class SolverEDIpack(object):

    def __init__(self, nimp, nbath, use_Ntot=False, use_Sz=False, use_SU2=False, thermal=False, solver_params=None):
        self.solver_params = solver_params if solver_params is not None else {}
        # Solver parameters, using BATH_MODE=hybrid
        self.ediNspin = 1 if use_SU2 else 2
        self.ediNorb  = nimp//2
        self.ediNbath = nbath//2


        # Fundamental sets for impurity degrees of freedom
        fops_imp_up = [('up', i) for i in range(self.ediNorb)]
        fops_imp_dn = [('dn', i) for i in range(self.ediNorb)]


        # Fundamental sets for bath degrees of freedom
        fops_bath_up = [('B_up', i) for i in range(self.ediNbath)]
        fops_bath_dn = [('B_dn', i) for i in range(self.ediNbath)]

        self.solver = EDIpackSolver(H_loc + H_int + H_bath,
                       fops_imp_up, fops_imp_dn, fops_bath_up, fops_bath_dn,
                       lanczos_params=LanczosParams(dim_threshold=16))
        
        #INTERACTION?
        #BETA?

#MANDATORY FUNCTIONS
    def build_Hemb(self, D, H1E, Lambda, V2E):
        raise NotImplementedError

    def solve_Hemb(self, num_eig=None, verbose=None, beta=500.0):
        raise NotImplementedError

    def calc_density_matrix(self):
        raise NotImplementedError

    def compute_E1loc(self):
        raise NotImplementedError

    def compute_E2loc(self):
        raise NotImplementedError

#AUXILIARY FUNCTIONS
