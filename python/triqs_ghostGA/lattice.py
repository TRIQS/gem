import numpy as np
from scipy.linalg import block_diag
from .fragment import Fragment
from .utility.utilities import calc_nf, calc_Fermi


class Lattice():
    '''
    Class for the lattice part to solve the quasiparticle problem
    '''
    def __init__(self, T: float,
                 ek_list: np.ndarray, wk_list: np.ndarray = None,
                 verbose=0
                  ):
        if not isinstance(T, float): raise TypeError(f"T must be int, got {type(T)}")
        if not isinstance(verbose, int): raise TypeError(f"verbose must be int, got {type(verbose)}")
        if not isinstance(ek_list, np.ndarray): raise TypeError(f"ek_list must be ndarray, got {type(ek_list)}")
        if(wk_list is None):
            wk_list=np.ones( ek_list.shape[0] )/ek_list.shape[0]
        if not isinstance(wk_list, np.ndarray): raise TypeError(f"wk_list must be ndarray, got {type(wk_list)}")

        if ek_list.ndim != 3 or ek_list.shape[1] != ek_list.shape[2]:
            raise ValueError(f"ek_list must be (A,B,B), got {ek_list.shape}")
        if wk_list.ndim != 1 or wk_list.shape[0] != ek_list.shape[0]:
            raise ValueError(f"wk_list must be (A,) matching ek_list first dim, got {wk_list.shape}")

        self.eks  = ek_list.copy()
        self.wks  = wk_list.copy()
        self.T    = T
        self.verb = verbose

        print("##### END OF LATTICE INITIALIZATION #####")

    def solve_qp(self, Fragments_list):
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")

        nimp_tot = sum(F.nimp for F in Fragments_list)
        nbath_tot = sum(F.nbath for F in Fragments_list)

        if self.eks.shape[1] != nimp_tot or self.eks.shape[2] != nimp_tot:
            raise ValueError(f"ek_list second and third dimensions must be {nimp_tot}, got {self.eks.shape}")

        self.Rtot = block_diag(*[F.R for F in Fragments_list])
        self.Ltot = block_diag(*[F.Lambda for F in Fragments_list])

        self.Delta_p_tot = np.zeros( (nbath_tot,nbath_tot), dtype=complex )
        self.ERD_tot   = np.zeros( (nimp_tot ,nbath_tot), dtype=complex )
        for ek,wk in zip(self.eks, self.wks):
            Hk_qp = self.Rtot @ ek @ self.Rtot.T.conj() + self.Ltot
            Dk = calc_nf(Hk_qp,self.T).T
            self.Delta_p_tot += wk*Dk
            self.ERD_tot   += wk*(ek @ self.Rtot.T.conj() @ Dk)

        imp_stride=0
        bath_stride=0
        for F in Fragments_list:
            F.Delta_qp = self.Delta_p_tot[ bath_stride:bath_stride+F.nbath , bath_stride:bath_stride+F.nbath ]
            F.ERD = self.ERD_tot[ imp_stride:imp_stride+F.nimp , bath_stride:bath_stride+F.nbath ]
            imp_stride  += F.nimp
            bath_stride += F.nbath

        return self.Delta_p_tot, self.ERD_tot

    def fit_mu(self, n_target, Fragments_list, mu0=0.0, mode='qp'):
        #Check that mu0 is float and mode is str
        # and n_target is float and comprehended between
        match mode.lower():
            case( 'qp' | 'quasiparticle' ):
                self.fit_mu_quasiparticle( n_target, Fragments_list, mu0=0.0 )
            case( 'imp' | 'impurity' | 'frag' | 'fragment' ):
                self.fit_mu_fragment( n_target, Fragments_list, mu0=0.0 )

    def fit_mu_quasiparticle(self, n_target, Fragments_list, mu0=0.0):
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")

        nimp_tot = sum(F.nimp for F in Fragments_list)
        nbath_tot = sum(F.nbath for F in Fragments_list)

        if self.eks.shape[1] != nimp_tot or self.eks.shape[2] != nimp_tot:
            raise ValueError(f"ek_list second and third dimensions must be {nimp_tot}, got {self.eks.shape}")

        self.Rtot = block_diag(*[F.R for F in Fragments_list])
        self.Ltot = block_diag(*[F.Lambda for F in Fragments_list])

        #Check if can be jitted
        def qp_density( mu, T, Lqp, Rqp, ek_qp, wk_qp):
            dens=0.0
            for ek,wk in zip(ek_qp, wk_qp):
                Hk_qp = Rqp @ ( ek -mu )@ Rqp.T.conj() + Lqp
                ekvals = np.linalg.eigvals(Hk_qp)
                dens += np.sum(calc_Fermi(ekvals/T))*wk
            return dens

        nqp_target = 0.5*(nbath_tot-nimp_tot) + n_target
        try:
            #DO: try root_finder which mu=mu_target such that qp_density(mu ...)==nqp_target starting from a guess mu=mu0

            for F in Fragments_list:
                F.Lambda += F.R @ mu_target*np.eye(F.nimp) @ F.R.T.conj()
        except:
            mu_target=None


        return mu_target



    def fit_mu_fragment(self, n_target, Fragments_list, nsteps=5, dmu0=1e-2, ntol=1e-4, mu0=0.0, spin_pen=0.0):
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")

        nfill_new = sum(F.nfill for F in Fragments_list)
        dmu=dmu0*np.sign(nfill_old-n_target)
        mu_old=mu0
        mu_new=mu_old+dmu
        # solve_impurity(self, mu, num_eig=1, spin_pen=0.0) for all fragments and update nfill_new saving nfill_old
        # then produce a linear interpolation to find the mu that would realize nfill_new=n_target and keep until convergence
        # is reached ( ntol < abs(nfill_new-n_target) ) or the maximum number od steps (nsteps) is reached 

        return mu_new

