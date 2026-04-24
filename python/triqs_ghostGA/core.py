import numpy as np
from utility.utilities import Hermitian_list

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
        self.Bgh = nbath/nimp
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

        self.Thermal = Thermal
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


class Lattice():
    '''
    Class for the lattice part to solve the quasiparticle problem
    '''
    def __init__(self, \
                 ek_list: np.ndarray, wk_list: np.ndarray, T: float, \
                 verbose=0 \
                  ):
        if not isinstance(ek_list, np.ndarray): raise TypeError(f"ek_list must be ndarray, got {type(ek_list)}")
        if not isinstance(wk_list, np.ndarray): raise TypeError(f"wk_list must be ndarray, got {type(wk_list)}")
        if not isinstance(T, float): raise TypeError(f"T must be int, got {type(T)}")
        if not isinstance(verbose, int): raise TypeError(f"verbose must be int, got {type(verbose)}")

        if ek_list.ndim != 3 or ek_list.shape[1] != ek_list.shape[2]:
            raise ValueError(f"ek_list must be (A,B,B), got {ek_list.shape}")
        if wk_list.ndim != 1 or wk_list.shape[0] != ek_list.shape[0]:
            raise ValueError(f"wk_list must be (A,) matching ek_list first dim, got {wk_list.shape}")

        self.eks = ek_list.copy()
        self.wks = wk_list.copy()
        self.T = T
        self.verb = verbose


        print("##### END OF LATTICE INITIALIZATION #####")