#######################################################
# Template for solvers to solve the embedding Hamiltonian in GEM.
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
#######################################################
import numpy as np

class SolverTemplate(object):
    '''
    Generic Solver class. The aim of this object that, given a general impurity Hamiltonian, it should be able to solve it an return the density matrix.
    '''
    def __init__(self,
                 norb,
                 use_Ntot=False,
                 use_Sz=False, # eventually flags to use symmetries
                 thermal=False, # flag to indicate if the calculations is thermal or not
                 solver_params=None, # dict of solver-specific parameters; keys depend on the solver
                 ):
        '''
        Initialization of the Solver object.

        :param norb:            int. Number of orbitals. Defines the dimensions of the Hamiltonian.
        :param use_Ntot:        bool. Whether or not the number of fermions is conserved (default False).
        :param use_Sz:          bool. Whether or not the Sz symmetry is enforced (default False).
        :param thermal:         bool. Flag to indicate whether the calculation is thermal or not.
        :param solver_params:   dict. Solver-specific parameters. Keys depend on the solver.
        '''
        #things that re relevant for the solver
        self.type = "SolverTemplate"
        self.solver_params = solver_params if solver_params is not None else {}


    def build_Hemb(self,
                   D, # MANDATORY: the hybridization matrix
                   eloc, # MANDATORY: the impurity one-body term
                   Lambdac, # MANDATORY: the bath one-body term
                   V2E, # MANDATORY: the two-body interaction in the impurity
                   verbose=0, # MANDATORY: verbose level
                   ):
        '''
        Construct the embedded Hamiltonian.

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
                   tol=1e-8,  # MANDATORY: tolerance for convergence
                   T=0.0 # MANDATORY: inverse temperature
                   ):
        '''
        Solve the embedded Hamiltonian. Either for the ground state or also some excited states, if not all.
        Solver-specific parameters are read from self.solver_params with sensible defaults,
        e.g.: my_param = self.solver_params.get('my_param', default_value).
        This includes how many eigenvectors to solve for ('num_eig' in SimpleED):
        Fragment.solve_impurity does not pass it.

        :param verbose:     int. Level of verbosity (default 1).
        :param tol:         float. Tolerance for convergence (default 1e-8).
        :param T:           float. Electronic temperature (default 0 for ground states).
        '''
        self.gs_ene = None # MANDATORY: ground state energy
        self.Zpart = None # MANDATORY: partition function for thermal calculations divided by exp(gs_ene/T) so that is 1 at zero temperature
        print("solve_Hemb not implemented yet")
        #HERE YOU SHOULD SOLVE



    def calc_density_matrix(self):
        '''
        Compute the denstiy matrix.

        Return:
          denmat: numpy.array. Densty matrix, <c^\dagger_i c_j>, of the system (impurity+bath).
        '''
        print("calc_density_matrix not implemented yet")
        denMat = None
        return denMat


    def compute_E1loc(self,eloc,mu=0.0):
        '''
        Compute the local one-body energy E1loc = Tr[eloc * denmat_impurity].

        :param eloc:    array. Local part of the Hamiltonian.
        :param mu:      float. Chemical potential (default 0).

        Return:
          E1loc: float. Local one-body energy.
        '''
        # THIS IS A POSSIBLE IMPLEMENTATION GIVEN THE DENSITY MATRIX AND THE LOCAL ONE-BODY HAMILTONIAN
        denMat = self.calc_density_matrix()
        E1loc = np.trace((eloc - mu * np.eye(self.nimp)).dot(denMat[:self.nimp,:self.nimp].T))
        return E1loc

    def compute_E2loc(self,eloc,D,Lambdac,mu=0.0):
        '''
        Compute the local energy, including local one- and two-body terms from a given set of thermal states.
        Works also at zero temperature for ground states.

        :param eloc:    array. Local part of the Hamiltonian.
        :param D:       array. D matrix.
        :param Lambdac: array. Lambda_c matrix.
        :param mu:      float. Chemical potential (default 0).

        Return:
          Eloc: float. Total local energy.
        '''
        print("compute_E2loc not implemented yet")
        #HERE IS A POSSIBLE IMPLEMENTATION GIVEN THE DENSITY MATRIX, THE ONE-BODY HAMILTONIAN AND THE GROUND STATE ENERGY
        denMat = self.calc_density_matrix()
        E1tot = 0.0
        E1tot+=np.trace((eloc - mu * np.eye(self.nimp)).dot(denMat[:self.nimp,:self.nimp].T)) #eloc part
        E1tot+=np.trace(Lambdac.dot(denMat[self.nimp:,:self.nimp].T)) #bath part
        E1tot+=np.trace(D.dot(denMat[self.nimp:,self.nimp:].T)) #hybridization part
        E1tot+=np.trace(D.T.conjg().dot(denMat[:self.nimp,self.nimp:].T)) #hybridization part
        E2loc = self.gs_ene - E1tot
        return E2loc


