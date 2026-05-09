import numpy as np
from .utility.utilities import Hermitian_list, funcMat, denR, calc_Lambda, calc_Lambda_c
from .utility.delta_fit import update_self_energy_thermal_penalty, update_hybridization_thermal_penalty


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

        #Checks?
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
        self.solver = solver

        #Self-energy parameters
        if Lambda is None:
            self.Lambda = np.kron(np.diag(np.tanh(np.arange(self.Bgh)-(self.Bgh-1)/nimp)), np.eye(nimp))
        else:
            self.Lambda = np.array(Lambda)
            if self.Lambda.shape != (nbath, nbath):
                raise ValueError(f"Lambda must be ({nbath},{nbath}), got {self.Lambda.shape}")
        if R is None:
            self.R = np.kron(np.random.rand(self.Bgh, 1), np.eye(nimp))
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
            self.D = np.kron(np.random.rand(self.Bgh, 1), np.eye(nimp))
        else:
            self.D = np.array(D)
            if self.D.shape != (nbath, nimp):
                raise ValueError(f"D must be ({nbath},{nimp}), got {self.D.shape}")

        self.verb = verbose

        #Create Hermitian list here and store
        #maybe with a variable nspin being 1 or 2 so that [::nspin] always stride properly
        self.H_list,self.tH_list=Hermitian_list(nbath)

        if(self.verb>1):
            print('initial R matrix =')
            print(self.R)
            print('initial Lambda matrix =')
            print(self.Lambda)
            print('initial D matrix =')
            print(self.D)
            print('initial Lambda_c matrix =')
            print(self.Lambda_c)
        print("##### END OF FRAGMENT INITIALIZATION #####")

    def solve_impurity(self, mu, T=0.0, num_eig=1, spin_pen=0.0):
        """
        Solve embedding problem using the solver from Fragment

        :param mu: float. Chemical potential.
        :param T: float. Temperature.
        :param num_eig: int. Number of eigenvalues to compute.
        :param spin_pen: float. Penalty for spin singlet symmetry breaking.
        """
        h1e = np.zeros((self.ntot,self.ntot), dtype=np.complex128)
        h1e[:self.nimp,:self.nimp] = self.eloc - mu*np.eye(self.nimp)
        h1e[:self.nimp,self.nimp:] = self.D.T
        h1e[self.nimp:,self.nimp:] = -self.Lambda_c
        h1e[self.nimp:,:self.nimp] = self.D.conj()

        if(self.verb>0):
            print(f"Solving embedding problem with solver  {self.solver.type}")
            print(" Temperature T =", T)
        
        self.solver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)
        
        if(T>=0.0):
            self.solver.solve_Hemb(num_eig=num_eig, verbose=self.verb , T=T)
        else:
            raise ValueError("Temperature T must be non-negative")

        self.denMat = self.solver.calc_density_matrix()
        fdagf = self.denMat[self.nimp:,self.nimp:]
        self.Delta_aim = np.eye(self.nbath) - fdagf
        self.nfill  = np.trace( self.denMat[:self.nimp,:self.nimp] )
        self.E2loc  = self.solver.compute_E2loc()

    def update_self_energy(self, T=0.0, move_pen=1e-6):
        '''
        This function update the self-energy parameters Lambda and R

        :param T: float. Temperature.
        :param move_pen: float. Penalty for moving the self-energy parameters.

        Return:
            R: ndarray. Updated self-energy parameter R.
            Lambda: ndarray. Updated self-energy parameter Lambda.
        '''
        cdagf = self.denMat[:self.nimp,self.nimp:]
        fdagf = self.denMat[self.nimp:,self.nimp:]
        self.Delta_aim = np.eye(self.nbath) - fdagf

        if T > 0.0:
            L_new, R_new = update_self_energy_thermal_penalty(self.Lambda, self.R, self.Lambda_c, self.D,
                                                              fdagf, cdagf,
                                                              beta=1/T, alpha=move_pen, method="dF")
        elif T == 0.0:
            R_new = np.transpose( cdagf.dot( funcMat(self.Delta_aim, denR)) )
            L_new = calc_Lambda( R_new, self.Lambda_c, self.Delta_aim, self.D, self.H_list )
        else:
            raise ValueError("Temperature T must be non-negative")
        self.R = R_new.copy()
        self.Lambda = L_new.copy()
        return self.R, self.Lambda

    def update_hybridization(self, T=0.0, move_pen=1e-6):
        '''
        This function update the hybridization parameters Lambda_c and D

        :param T: float. Temperature.
        :param move_pen: float. Penalty for moving the hybridization parameters.

        Return:
            D: ndarray. Updated hybridization parameter D.
            Lambda_c: ndarray. Updated hybridization parameter Lambda_c.
        '''
        if T > 0.0:
            Lc_new, D_new = update_hybridization_thermal_penalty(self.Lambda_c, self.D, self.Lambda, self.R,
                                                                 self.Delta_qp, self.ERD.T,
                                                                 beta=1/T, alpha=move_pen, method="dF")
        elif T == 0.0:
            D_new = np.dot(funcMat(self.Delta_qp, denR),np.transpose(self.ERD))
            Lc_new = calc_Lambda_c(self.R, self.Lambda, self.Delta_qp, D_new, self.H_list)
        else:
            raise ValueError("Temperature T must be non-negative")
        self.D = D_new.copy()
        self.Lambda_c = Lc_new.copy()
        return self.D, self.Lambda_c
    
    def compute_energy(self):
        '''
        Compute the energy contributions of the fragment using the density matrix and the Hamiltonian parameters.

        Return:
          E: float. Energy of the fragment.
        '''
        self.E1loc = self.solver.compute_E1loc()
        self.E2loc = self.solver.compute_E2loc()
        E = self.E1loc + self.E2loc
        return E

    def compute_Z(self, mu=0.0, z0=0.0, h=1e-8):
        """
        Compute the quasiparticle weight Z from the self-energy parameters in the local case.

        :param mu: float. Chemical potential.
        :param z0: float. Frequency at which to compute Z.
        :param h: float. Step size for finite difference.

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