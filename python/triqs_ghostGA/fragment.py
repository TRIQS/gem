import numpy as np
from .utility.utilities import Hermitian_list, funcMat, denR, calc_Lambda, calc_Lambda_c
from .utility.delta_fit import update_self_energy_thermal_penalty, update_hybridization_thermal_penalty


class Fragment():
    '''
    Class for the embedded correlated space
    '''
# nimp and nbath or nimp and B?
# Think of passing dict instead of all those parameters?
# N.B. penalties and sectors are decided at solver level, thermal?
    def __init__(self, \
                 nimp: int, nbath: int, T: float, \
                 eloc: np.ndarray, Utensor: np.ndarray, \
                 solver, \
                 Lambda=None,R=None,Lambda_c=None,D=None, \
                 Thermal=False, spin_sym=False, verbose=0 \
                  ):

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
        self.T = float(T)

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

        self.thermal = Thermal
        self.spin_sym = spin_sym
        self.verb = verbose

        #Create Hermitian list here and store
        #only spinful but is spin_sym one may think of smaller one
        #maybe with a variable nspin being 1 or 2 so that [::nspin] always stride properly
        self.H_list,self.tH_list=Hermitian_list(nbath)

        if(self.verb>1):
            print('initial R matirx =')
            print(self.R)
            print('initial Lambda matirx =')
            print(self.Lambda)
            print('initial D matirx =')
            print(self.D)
            print('initial Lambda_c matirx =')
            print(self.Lambda_c)
        print("##### END OF FRAGMENT INITIALIZATION #####")

    def solve_impurity(self, mu, num_eig=1, spin_pen=0.0):
        """
        Solve embedding problem using the solver from Fragment
        """
        h1e = np.zeros((self.ntot,self.ntot), dtype=np.complex128)
        h1e[:self.nimp,:self.nimp] = self.eloc - mu*np.eye(self.nimp)
        h1e[:self.nimp,self.nimp:] = self.D.T
        h1e[self.nimp:,self.nimp:] = -self.Lambda_c
        h1e[self.nimp:,:self.nimp] = self.D.conj()

        if self.solver.type == "CI":
            self.solver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor)

        elif self.solver.type == "FTPS":
            self.solver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)

        elif self.solver.type == "ITensorMPSSolver":
            self.solver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)

        elif self.solver.type == "PySCFCCSD":
            self.solver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)

        elif self.solver.type == "Block2NSZ":
            self.solver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)

        else:
            raise ValueError("only Full ED, CI, and HCI are supported")

        if(self.solver.thermal):
            self.solver.solve_Hemb(num_eig=num_eig, verbose=self.verb , beta=1/self.T)
        else:
            self.solver.solve_Hemb(num_eig=num_eig, verbose=self.verb )

        self.denMat = self.solver.calc_density_matrix()
        self.nfill  = np.trace( self.denMat[:self.nimp,:self.nimp] )
        self.E2loc  = self.solver.compute_E2loc()

    def update_self_energy(self, mix=0.1, move_pen=1e-6):
        '''
        This function update the self-energy parameters Lambda and R
        '''
        cdagf = self.denMat[:self.nimp,self.nimp:]
        fdagf = self.denMat[self.nimp:,self.nimp:]
        self.Delta_aim = np.eye(self.nbath) - fdagf

        if(self.thermal):
            L_new, R_new = update_self_energy_thermal_penalty(self.Lambda, self.R, self.Lambda_c, self.D,
                                                              fdagf, cdagf,
                                                              beta=1/self.T, alpha=move_pen, method="dF")
        else:
            R_new = np.transpose( cdagf.dot( funcMat(self.Delta_aim, denR)) )
            L_new = calc_Lambda( R_new, self.Lambda_c, self.Delta_aim, self.D, self.H_list )

        if(self.spin_sym):
            R_new=np.kron( R_new[::2,::2], np.eye(2) )
            L_new=np.kron( L_new[::2,::2], np.eye(2) )

        self.R = (1-mix)*R_new.copy() + mix*self.R
        self.Lambda = (1-mix)*L_new.copy() + mix*self.Lambda

        return self.R, self.Lambda

    def update_hybridization(self, move_pen=1e-6):
        '''
        This function update the hybridization parameters Lambda_c and D
        '''
        if(self.thermal):
            Lc_new, D_new = update_hybridization_thermal_penalty(self.Lambda_c, self.D, self.Lambda, self.R,
                                                                 self.Delta_qp, self.ERD.T,
                                                                 beta=1/self.T, alpha=move_pen, method="dF")
        else:
            D_new = np.dot(funcMat(self.Delta_qp, denR),np.transpose(self.ERD))
            Lc_new = calc_Lambda_c(self.R, self.Lambda, self.Delta_qp, self.D, self.H_list)

        if(self.spin_sym):
            D_new=np.kron( D_new[::2,::2], np.eye(2) )
            Lc_new=np.kron( Lc_new[::2,::2], np.eye(2) )

        self.D = D_new.copy()
        self.Lambda_c = Lc_new.copy()

        return self.D, self.Lambda_c

    def compute_Z(self, mu=0.0, z0=0.0, h=1e-8):
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