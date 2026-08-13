###########################################
# Definition of Fragment objects
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
###########################################

import warnings

import numpy as np
from .utilities import Hermitian_list, funcMat, denR, calc_Lambda, calc_Lambda_c
from .delta_fit import update_self_energy_thermal_penalty, update_hybridization_thermal_penalty
from .solvers.gem_solver import gemSolver


class Fragment():
    '''
    Class for the embedded correlated space.

    It contains the matrices parameterizing the self-energy ( R and Lambda )
    and the hybridization ( D and Lambda_c ), as well as the local Hamiltonian parameters.

    It takes care of solving the correlated embedding problem and updating the self-energy and hybridization parameters.
    '''
# nimp and nbath or nimp and B?
# Think of passing dict instead of all those parameters?
# N.B. penalties and sectors are decided at solver level, thermal?
    def __init__(self,
                 nimp: int, nbath: int,
                 eloc: np.ndarray, Utensor: np.ndarray,
                 solver,
                 Lambda=None,R=None,Lambda_c=None,D=None,
                 verbose=0
                  ):
        """  
        Initialize the Fragment class with the given parameters.

        :param nimp: int. Number of impurity spin-orbital levels.
        :param nbath: int. Number of bath spin-orbital levels.
        :param eloc: ndarray. Local one-body electronic Hamiltonian.
        :param Utensor: ndarray. Tensor of local electron-electron interactions.
        :param solver: object. Solver for the impurity problem.    
        :param Lambda: ndarray, optional. Self-energy parameters Lambda.
        :param R: ndarray, optional. Self-energy parameters R.
        :param Lambda_c: ndarray, optional. Hybridization parameters Lambda_c.
        :param D: ndarray, optional. Hybridization parameters D.
        :param verbose: int, optional. Level of verbosity.

        """

        if not isinstance(nimp, int): raise TypeError(f"nimp must be int, got {type(nimp)}")
        if not isinstance(nbath, int): raise TypeError(f"nbath must be int, got {type(nbath)}")
        if not isinstance(eloc, np.ndarray): raise TypeError(f"eloc must be ndarray, got {type(eloc)}")
        if not isinstance(Utensor, np.ndarray): raise TypeError(f"Utensor must be ndarray, got {type(Utensor)}")
        if not isinstance(verbose, int): raise TypeError(f"verbose must be int, got {type(verbose)}")

        if nbath % nimp != 0:
            raise ValueError(f"nbath must be a multiple of nimp, got nbath={nbath}, nimp={nimp}")

        self.nimp = nimp
        self.nbath = nbath
        self.ntot = nimp + nbath
        self.Bgh = nbath//nimp

        assert eloc.shape == (nimp, nimp), f"eloc must be ({nimp},{nimp}), got {eloc.shape}"
        assert Utensor.shape == (nimp,)*4, f"Utensor must be ({nimp},{nimp},{nimp},{nimp}), got {Utensor.shape}"

        self.eloc = eloc.copy()
        self.Utensor = Utensor.copy()

        #CHECK THE SOLVER CLASS
        if not isinstance(solver, gemSolver):
            raise TypeError(
                f"solver must be a gemSolver, got {type(solver)}. Every GEM solver "
                "must inherit from gem.solvers.gem_solver.gemSolver, see "
                "gem.solvers.solver_template.SolverTemplate for a skeleton.")
        missing = solver.missing_methods()
        if missing:
            warnings.warn(
                f"The solver {solver.type} does not implement {', '.join(missing)}: "
                "solve_impurity will fail.")
        self.solver = solver

        #Self-energy parameters
        if Lambda is None:
            self.Lambda = np.kron(np.diag(np.tanh(np.arange(self.Bgh)-(self.Bgh-1)/nimp)), np.eye(nimp))
        else:
            self.Lambda = np.array(Lambda)
            if self.Lambda.shape != (nbath, nbath):
                raise ValueError(f"Lambda must be ({nbath},{nbath}), got {self.Lambda.shape}")
        if R is None:
            self.R = np.kron(np.ones((self.Bgh, 1))/np.sqrt(self.Bgh), np.eye(nimp))
        else:
            self.R = np.array(R)
            if self.R.shape != (nbath, nimp):
                raise ValueError(f"R must be ({nbath},{nimp}), got {self.R.shape}")

        #Hybridization parameters
        if Lambda_c is None:
            self.Lambda_c = np.kron(np.diag(np.tanh(np.arange(self.Bgh)-(self.Bgh-1)/nimp)), np.eye(nimp))
        else:
            self.Lambda_c = np.array(Lambda_c)
            if self.Lambda_c.shape != (nbath, nbath):
                raise ValueError(f"Lambda_c must be ({nbath},{nbath}), got {self.Lambda_c.shape}")
        if D is None:
            self.D = np.kron(np.ones((self.Bgh, 1))/np.sqrt(self.Bgh), np.eye(nimp))
        else:
            self.D = np.array(D)
            if self.D.shape != (nbath, nimp):
                raise ValueError(f"D must be ({nbath},{nimp}), got {self.D.shape}")

        self.verb = verbose

        #Create Hermitian list here and store
        #spinfull and spinless versions
        self.H_list,self.tH_list=Hermitian_list(nbath)
        self.Hs_list,self.tHs_list=Hermitian_list(nbath//2)

        if(self.verb>2):
            print('initial R matrix =')
            print(self.R)
            print('initial Lambda matrix =')
            print(self.Lambda)
            print('initial D matrix =')
            print(self.D)
            print('initial Lambda_c matrix =')
            print(self.Lambda_c)
        print("##### END OF FRAGMENT INITIALIZATION #####")

    def solve_impurity(self, mu, T=0.0):
        """
        Solve embedding problem using the solver from Fragment

        :param mu: float. Chemical potential.
        :param T: float, optional. Temperature (default: 0.0).
        """
        h1e = np.zeros((self.ntot,self.ntot), dtype=np.complex128)
        h1e[:self.nimp,:self.nimp] = self.eloc - mu*np.eye(self.nimp)
        h1e[:self.nimp,self.nimp:] = self.D.T
        h1e[self.nimp:,self.nimp:] = -self.Lambda_c
        h1e[self.nimp:,:self.nimp] = self.D.conj()

        if(self.verb>0):
            print(f"Solving embedding problem with solver  {self.solver.type}")
            print(" Temperature T =", T)
        
        self.solver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor)
        
        if(T>=0.0):
            self.solver.solve_Hemb(T=T, verbose=self.verb )
        else:
            raise ValueError("Temperature T must be non-negative")

        self.denMat = self.solver.calc_density_matrix()
        fdagf = self.denMat[self.nimp:,self.nimp:]
        self.Delta_aim = np.eye(self.nbath) - fdagf
        self.nfill  = np.trace( self.denMat[:self.nimp,:self.nimp] )
        self.E2loc  = self.solver.compute_E2loc()

    def update_self_energy(self, T=0.0, move_pen=1e-6, use_Sz=False):
        '''
        This function update the self-energy parameters Lambda and R

        :param T: float, optional. Temperature (default: 0.0).
        :param move_pen: float, optional. Penalty for moving the self-energy parameters (default: 1e-6).
        :param use_Sz: bool, optional. Whether to use Sz as a good quantum number (default: False).

        Return:
            R: ndarray. Updated self-energy parameter R.
            Lambda: ndarray. Updated self-energy parameter Lambda.
        '''
        cdagf = self.denMat[:self.nimp,self.nimp:]
        fdagf = self.denMat[self.nimp:,self.nimp:]
        self.Delta_aim = np.eye(self.nbath) - fdagf

        sstep = 2 if use_Sz else 1
        L_s=[]; R_s=[]
        if T > 0.0:
            for spin in range(sstep):
                L_new, R_new = update_self_energy_thermal_penalty(self.Lambda[spin::sstep,spin::sstep], self.R[spin::sstep,spin::sstep],
                                                                self.Lambda_c[spin::sstep,spin::sstep], self.D[spin::sstep,spin::sstep],
                                                                fdagf[spin::sstep,spin::sstep], cdagf[spin::sstep,spin::sstep],
                                                                beta=1/T, alpha=move_pen, method="dF")
                L_s.append(L_new); R_s.append(R_new)
        elif T == 0.0:
            hlist = self.Hs_list if use_Sz else self.H_list
            for spin in range(sstep):
                R_new = np.transpose( cdagf[spin::sstep,spin::sstep].dot( funcMat(self.Delta_aim[spin::sstep,spin::sstep], denR)) )
                L_new = calc_Lambda( R_new, self.Lambda_c[spin::sstep,spin::sstep],
                                    self.Delta_aim[spin::sstep,spin::sstep], self.D[spin::sstep,spin::sstep], hlist )
                L_s.append(L_new); R_s.append(R_new)
        else:
            raise ValueError("Temperature T must be non-negative")
        
        self.R = np.kron( R_s[0], np.eye(sstep) )
        self.Lambda = np.kron( L_s[0], np.eye(sstep) )
        if(sstep==2):
            self.R[1::2,1::2] = R_s[1]
            self.Lambda[1::2,1::2] = L_s[1]
        return self.R, self.Lambda

    def update_hybridization(self, T=0.0, move_pen=1e-6, use_Sz=False):
        '''
        This function update the hybridization parameters Lambda_c and D

        :param T: float, optional. Temperature (default: 0.0).
        :param move_pen: float, optional. Penalty for moving the hybridization parameters (default: 1e-6).
        :param use_Sz: bool, optional. Whether to use Sz as a good quantum number (default: False).

        Return:
            D: ndarray. Updated hybridization parameter D.
            Lambda_c: ndarray. Updated hybridization parameter Lambda_c.
        '''
        sstep = 2 if use_Sz else 1
        D_s=[]; Lc_s=[]
        if T > 0.0:
            for spin in range(sstep):
                Lc_new, D_new = update_hybridization_thermal_penalty(self.Lambda_c[spin::sstep,spin::sstep], self.D[spin::sstep,spin::sstep],
                                                                     self.Lambda[spin::sstep,spin::sstep], self.R[spin::sstep,spin::sstep],
                                                                     self.Delta_qp[spin::sstep,spin::sstep], self.ERD.T[spin::sstep,spin::sstep],
                                                                     beta=1/T, alpha=move_pen, method="dF")
                Lc_s.append(Lc_new); D_s.append(D_new)
        elif T == 0.0:
            hlist = self.Hs_list if use_Sz else self.H_list
            for spin in range(sstep):
                D_new = np.dot(funcMat(self.Delta_qp[spin::sstep,spin::sstep], denR),np.transpose(self.ERD[spin::sstep,spin::sstep]))
                Lc_new = calc_Lambda_c(self.R[spin::sstep,spin::sstep], self.Lambda[spin::sstep,spin::sstep],
                                       self.Delta_qp[spin::sstep,spin::sstep], D_new, hlist)
                Lc_s.append(Lc_new); D_s.append(D_new)
        else:
            raise ValueError("Temperature T must be non-negative")
        
        self.D = np.kron( D_s[0], np.eye(sstep) )
        self.Lambda_c = np.kron( Lc_s[0], np.eye(sstep) )
        if(sstep==2):
            self.D[1::2,1::2] = D_s[1]
            self.Lambda_c[1::2,1::2] = Lc_s[1]
        return self.D, self.Lambda_c
    
    def compute_energy(self):
        '''
        Compute the energy contributions of the fragment using the density matrix and the Hamiltonian parameters.

        Return:
          E: float. Energy of the fragment.
        '''
        self.E1loc = self.solver.compute_E1loc(self.nimp)
        self.E2loc = self.solver.compute_E2loc()
        E = self.E1loc + self.E2loc
        return E
    
    def compute_self_energy(self, z, mu=0.0):
        """
        Compute the self-energy from the analytical formula.

        :param z: complex. Real or matsubara frequency for self-energy
        :param mu: float, optional. Chemical potential (default: 0.0).
        """
        m, nu = self.R.shape
        I_m = np.eye(m, dtype=complex)
        I_nu = np.eye(nu, dtype=complex)
        Ainv = np.linalg.inv(z*I_m - self.Lambda)
        M = self.R.conj().T @ Ainv @ self.R
        
        return (z)*I_nu - np.linalg.inv(M) - self.eloc + mu*I_nu


    def compute_Z(self, mu=0.0, z0=0.0, h=1e-4):
        """
        Compute the quasiparticle weight Z from the self-energy parameters in the local case.

        :param mu: float, optional. Chemical potential (default: 0.0).
        :param z0: float, optional. Frequency at which to compute Z (default: 0.0).
        :param h: float, optional. Step size for finite difference (default: 1e-4).

        Return:
            Z: float. Quasiparticle weight.
        """
        m, nu = self.R.shape
        I_m = np.eye(m, dtype=complex)
        I_nu = np.eye(nu, dtype=complex)

        def Sigma(z):
            Ainv = np.linalg.inv(z*I_m - self.Lambda)
            M = self.R.conj().T @ Ainv @ self.R
            return (z)*I_nu - np.linalg.inv(M) - self.eloc + mu*I_nu

        # centered finite difference derivative
        dSigma = (Sigma(z0 + h) - Sigma(z0 - h)) / (2*h)

        return np.linalg.inv(I_nu - dSigma)
    
    ###### ROUTINES TO IMPOSE SYMMETRY #####
    def impose_spin_SU2_symmetry(self):
        """
        Impose spin SU(2) symmetry on the self-energy and hybridization parameters by averaging over spin components.
        """
        self.R = 0.5*np.kron( self.R[::2,::2] + self.R[1::2,1::2], np.eye(2) )
        self.Lambda = 0.5*np.kron( self.Lambda[::2,::2] + self.Lambda[1::2,1::2], np.eye(2) )
        self.Lambda_c = 0.5*np.kron( self.Lambda_c[::2,::2] + self.Lambda_c[1::2,1::2], np.eye(2) )
        self.D = 0.5*np.kron( self.D[::2,::2] + self.D[1::2,1::2], np.eye(2) )

    def impose_orbital_symmetry(self):
        """
        Impose orbital symmetry on the self-energy and hybridization parameters by averaging over orbital components.
        """
        n_orb = self.nimp//2
        R_new = np.zeros((self.nbath//n_orb, self.nimp//n_orb), dtype=np.complex128)
        Lambda_new = np.zeros((self.nbath//n_orb, self.nbath//n_orb), dtype=np.complex128)
        D_new = np.zeros((self.nbath//n_orb, self.nimp//n_orb), dtype=np.complex128)
        Lambda_c_new = np.zeros((self.nbath//n_orb, self.nbath//n_orb), dtype=np.complex128)

        for i in range(n_orb):
            idx_i = slice(i*2, (i+1)*2)
            idx_b = slice(i*(self.nbath//n_orb), (i+1)*(self.nbath//n_orb))
            R_new += self.R[idx_b, idx_i]
            Lambda_new += self.Lambda[idx_b, idx_b]
            D_new += self.D[idx_b, idx_i]
            Lambda_c_new += self.Lambda_c[idx_b, idx_b]
        self.R = np.kron(np.eye(n_orb), R_new)/n_orb
        self.Lambda = np.kron(np.eye(n_orb), Lambda_new)/n_orb
        self.Lambda_c = np.kron(np.eye(n_orb), Lambda_c_new)/n_orb 
        self.D = np.kron(np.eye(n_orb), D_new)/n_orb
