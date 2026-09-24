#######################################################
# Template for solvers to solve the embedding Hamiltonian in GEM.
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
#######################################################
import numpy as np

from .gem_solver import gemSolver

class SolverTemplate(gemSolver): # MANDATORY: every GEM solver inherits from gemSolver
    '''
    Generic Solver class. The aim of this object that, given a general impurity Hamiltonian, it should be able to solve it an return the density matrix.

    Only what appears below is used by the Fragment: the methods marked MANDATORY,
    plus the gs_ene and Zpart attributes set by solve_Hemb. Everything else
    (dimensions, symmetry sectors, whether the calculation is thermal, bond
    dimensions, ...) is solver-specific: it is decided here, at initialization,
    or read from self.solver_params, and the Fragment never sets it.
    '''
    def __init__(self,
                 norb, # SOLVER-SPECIFIC: dimensions, symmetry flags, ... are decided here, the Fragment does not pass them
                 solver_params=None, # MANDATORY: dict of solver-specific parameters; keys depend on the solver
                 ):
        '''
        Initialization of the Solver object.
        The signature is free: only the caller who builds the solver sees it,
        not the Fragment.

        :param norb:            int. Number of orbitals. Defines the dimensions of the Hamiltonian.
        :param solver_params:   dict. Solver-specific parameters. Keys depend on the solver.
        '''
        #things that re relevant for the solver
        # MANDATORY: sets self.type and self.solver_params (a copy of the dict above)
        super().__init__(solver_params=solver_params, solver_type="SolverTemplate")
        self.norb = norb


    def build_Hemb(self,
                   D, # MANDATORY: the hybridization matrix
                   eloc, # MANDATORY: the impurity one-body term, already containing -mu
                   Lambdac, # MANDATORY: the bath one-body term
                   V2E, # MANDATORY: the two-body interaction in the impurity
                   verbose=0, # SOLVER-SPECIFIC: the Fragment calls build_Hemb with the four arguments above only
                   ):
        '''
        Construct the embedded Hamiltonian.
        The four matrices are the only things the Fragment passes, positionally.

        Solver-specific parameters are read from self.solver_params with sensible defaults,
        e.g.: my_param = self.solver_params.get('my_param', default_value).
        This includes the penalty terms used to enforce symmetries ('spin_pen', 'sz_pen',
        'sx_pen', 'sy_pen' in SimpleED): Fragment.solve_impurity does not pass them.

        :param D:           array. D matrix.
        :param eloc:        array. Local part of the Hamiltonian.
        :param Lambdac:     array. Lambda_c matrix.
        :param V2E:         array. Two-body interaction of the impurity (Fragment).
        :param verbose:     int. Level of verbosity (default 0).
        '''
        print("build_Hemb not implemented yet")


    def solve_Hemb(self,
                   verbose=1, # MANDATORY: verbose level
                   T=0.0 # MANDATORY: electronic temperature
                   ):
        '''
        Solve the embedded Hamiltonian. Either for the ground state or also some excited states, if not all.
        These two keyword arguments are the only ones the Fragment passes.

        Solver-specific parameters are read from self.solver_params with sensible defaults,
        e.g.: my_param = self.solver_params.get('my_param', default_value).
        This includes how many eigenvectors to solve for ('num_eig' in SimpleED)
        and the tolerance of the diagonalization ('tol'): Fragment.solve_impurity
        does not pass them.

        :param verbose:     int. Level of verbosity (default 1).
        :param T:           float. Electronic temperature (default 0 for ground states).
        '''
        self.gs_ene = None # MANDATORY: ground state energy
        self.Zpart = None # MANDATORY: partition function for thermal calculations divided by exp(gs_ene/T) so that is 1 at zero temperature
        print("solve_Hemb not implemented yet")
        #HERE YOU SHOULD SOLVE



    def calc_density_matrix(self):
        '''
        Compute the denstiy matrix. Takes no argument.

        Return:
          denmat: numpy.array. Densty matrix, <c^\dagger_i c_j>, of the system (impurity+bath).
        '''
        print("calc_density_matrix not implemented yet")
        denMat = None
        return denMat


    def compute_E1loc(self,
                      nimp, # number of impurity spin-orbital levels
                      ):
        '''
        Compute the local one-body energy E1loc = Tr[eloc * denmat_impurity].
        The one-body Hamiltonian is the one stored by build_Hemb, so it already
        contains eloc - mu.
        OPTIONAL: only needed by Fragment.compute_energy.

        :param nimp:    int. Number of impurity spin-orbital levels.

        Return:
          E1loc: float. Local one-body energy.
        '''
        # THIS IS A POSSIBLE IMPLEMENTATION GIVEN THE DENSITY MATRIX AND THE LOCAL ONE-BODY HAMILTONIAN
        denMat = self.calc_density_matrix()
        E1loc = np.trace(self.h1e[:nimp,:nimp].dot(denMat[:nimp,:nimp].T))
        return E1loc

    def compute_E2loc(self):
        '''
        Compute the local two-body energy from a given set of thermal states.
        Works also at zero temperature for ground states.
        Takes no argument: eloc, D and Lambda_c are the ones stored by build_Hemb.

        Return:
          E2loc: float. Local two-body energy.
        '''
        print("compute_E2loc not implemented yet")
        #HERE IS A POSSIBLE IMPLEMENTATION GIVEN THE DENSITY MATRIX, THE ONE-BODY HAMILTONIAN AND THE GROUND STATE ENERGY
        #N.B. it only holds for the ground state: at T>0 the two-body term has to be
        #averaged over the thermal states, as SimpleED.compute_E2loc does.
        denMat = self.calc_density_matrix()
        E1tot = np.trace(self.h1e.dot(denMat.T)) #eloc - mu, Lambda_c and D blocks at once
        E2loc = self.gs_ene - E1tot
        return E2loc

    def calc_double_occ(self,
                        i, # index of the impurity level
                        ):
        '''
        Compute the double occupancy <n_up n_dn> of the impurity level i.
        OPTIONAL: only needed by Gdmft.run.

        :param i:   int. Index of the impurity level.

        Return:
          docc: float. Double occupancy of the impurity level i.
        '''
        print("calc_double_occ not implemented yet")
        docc = None
        return docc


