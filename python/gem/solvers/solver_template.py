#######################################################
# Template for solvers to solve the embedding Hamiltonian in gem.
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
#######################################################
import numpy as np

class SolverTemplate(object):
    '''
    Your class aim to solve general impurity Hamiltonian.
    '''
    def __init__(self,
                 norb,
                 use_Ntot=False, use_Sz=False, # eventually flags to use symmetries
                 thermal=False, # flag to indicate if the calculations is thermal or not
                 solver_params=None, # dict of solver-specific parameters; keys depend on the solver
                 ):
        #things that re relevant for the solver
        self.type = "SolverTemplate"
        self.solver_params = solver_params if solver_params is not None else {}
    
    def build_Hemb(self,
                   D, # MANDATORY: the hybridization matrix
                   eloc, # MANDATORY: the impurity one-body term
                   Lambdac, # MANDATORY: the bath one-body term
                   V2E, # MANDATORY: the two-body interaction in the impurity
                   verbose=0, # MANDATORY: verbose level
                   spin_pen=0, sz_pen=0, sx_pen=0, sy_pen=0 # eventually penalty terms to enforce symmetries
                   ):
        print("build_Hemb not implemented yet")

    def solve_Hemb(self,
                   num_eig=1, # MANDATORY: number of eigenvalues to compute
                   verbose=1, # MANDATORY: verbose level
                   tol=1e-8,  # MANDATORY: tolerance for convergence
                   T=0.0 # MANDATORY: inverse temperature
                   ):
        '''
        diagonalize the Hamiltonian.
        Solver-specific parameters are read from self.solver_params with sensible defaults,
        e.g.: my_param = self.solver_params.get('my_param', default_value)
        '''
        self.gs_ene = None # MANDATORY: ground state energy
        self.Zpart = None # MANDATORY: partition function for thermal calculations divided by exp(gs_ene/T) so that is 1 at zero temperature
        print("solve_Hemb not implemented yet")
        #HERE YOU SHOULD SOLVE


    
    def calc_density_matrix(self):
        '''
        Compute denstiy matrix.
        Return:
          denmat: numpy.array. Densty matrix, <c^\dagger_i c_j>, of the system (impurity+bath).
        '''
        print("calc_density_matrix not implemented yet")
        denMat = None
        return denMat

    
    def compute_E1loc(self,eloc,mu=0.0):
        '''
        Compute the local one-body energy E1loc = Tr[eloc * denmat_impurity]
        Return:
          E1loc: float. Local one-body energy.
        '''
        # THIS IS A POSSIBLE IMPLEMENTATION GIVEN THE DENSITY MATRIX AND THE LOCAL ONE-BODY HAMILTONIAN
        denMat = self.calc_density_matrix()
        E1loc = np.trace((eloc - mu * np.eye(self.nimp)).dot(denMat[:self.nimp,:self.nimp].T))
        return E1loc
    
    def compute_E2loc(self,eloc,D,Lambdac,mu=0.0):
        '''
        Compute local energy including local one and two-body term from a given set od thermal states
        Works also at zero Temperature
        Input:
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

    