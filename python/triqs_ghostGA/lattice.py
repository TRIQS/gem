import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import brentq, bisect
from .fragment import Fragment
from .utility.utilities import calc_nf, calc_Fermi
from .utility.delta_fit import build_H
from numba import jit


class Lattice():
    '''
    Class for the lattice part to solve the quasiparticle problem
    '''
    def __init__(self,
                 ek_list: np.ndarray, wk_list: np.ndarray = None,
                 verbose=0
                  ):
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
        self.verb = verbose

        print("##### END OF LATTICE INITIALIZATION #####")

    def solve_qp(self, Fragments_list, T=0.0):
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")

        nimp_tot = sum(F.nimp for F in Fragments_list)
        nbath_tot = sum(F.nbath for F in Fragments_list)

        if self.eks.shape[1] != nimp_tot or self.eks.shape[2] != nimp_tot:
            raise ValueError(f"ek_list second and third dimensions must be {nimp_tot}, got {self.eks.shape}")
        if(T<0.0): raise ValueError("Temperature T must be non-negative")
        Tuse=np.maximum(1e-3,T)
        self.Rtot = block_diag(*[F.R for F in Fragments_list])
        self.Ltot = block_diag(*[F.Lambda for F in Fragments_list])

        self.Delta_p_tot = np.zeros( (nbath_tot,nbath_tot), dtype=complex )
        self.ERD_tot   = np.zeros( (nimp_tot ,nbath_tot), dtype=complex )
        for ek,wk in zip(self.eks, self.wks):
            Hk_qp = self.Rtot @ ek @ self.Rtot.T.conj() + self.Ltot
            Dk = calc_nf(Hk_qp,Tuse).T
            self.Delta_p_tot += wk*Dk
            self.ERD_tot   += wk*(ek @ self.Rtot.T.conj() @ Dk.T)

        imp_stride=0
        bath_stride=0
        for F in Fragments_list:
            F.Delta_qp = self.Delta_p_tot[ bath_stride:bath_stride+F.nbath , bath_stride:bath_stride+F.nbath ]
            F.ERD = self.ERD_tot[ imp_stride:imp_stride+F.nimp , bath_stride:bath_stride+F.nbath ]
            imp_stride  += F.nimp
            bath_stride += F.nbath

        return self.Delta_p_tot, self.ERD_tot
    
    def compute_Gloc(self, w_list, Fragments_list, eps=1e-2):
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")

        nimp_tot = sum(F.nimp for F in Fragments_list)
        nbath_tot = sum(F.nbath for F in Fragments_list)

        if self.eks.shape[1] != nimp_tot or self.eks.shape[2] != nimp_tot:
            raise ValueError(f"ek_list second and third dimensions must be {nimp_tot}, got {self.eks.shape}")

        self.Rtot = block_diag(*[F.R for F in Fragments_list])
        self.Ltot = block_diag(*[F.Lambda for F in Fragments_list])


        @jit(nopython=True)
        def compute_Gloc_at_w( w, Rtot, Ltot, eks, wks, eps):
            # Implementation for computing Gloc at a specific frequency w
            Gloc_w = np.zeros( (eks.shape[1], eks.shape[1]), dtype=np.complex128 )
            for ek,wk in zip(eks, wks):
                Hk_qp = Rtot @ ek @ Rtot.T.conj() + Ltot
                Gk_qp = Rtot.T.conj() @ np.linalg.inv( (w+1j*eps)*np.eye(Hk_qp.shape[0]) - Hk_qp ) @ Rtot
                Gloc_w += wk*Gk_qp
            return Gloc_w

        Gloc = np.zeros( (len(w_list),nimp_tot,nimp_tot), dtype=np.complex128 )
        for i,w in enumerate(w_list):
            Gloc[i,:,:] = compute_Gloc_at_w(w, self.Rtot, self.Ltot, self.eks, self.wks, eps)
        return Gloc
    

    def fit_mu(self, n_target, Fragments_list, T=1e-2, mu_old=0.0, mode='qp', ntol=1e-4):
        m = mode.lower()
        if(T<0.0): raise ValueError("Temperature T must be non-negative")
        if m in ('qp', 'quasiparticle'):
            return self.fit_mu_quasiparticle( n_target, Fragments_list, T=T, mu_old=mu_old )
        elif m in ('imp', 'impurity', 'frag', 'fragment'):
            return self.fit_mu_fragment( n_target, Fragments_list, T=T, mu_old=mu_old, ntol=ntol )

    def fit_mu_quasiparticle(self, n_target, Fragments_list, T=1e-2, mu_old=0.0):
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")
        if T < 0.0: raise ValueError("Temperature T must be non-negative")
        Tuse=np.maximum(1e-3,T)
        nimp_tot = sum(F.nimp for F in Fragments_list)
        nbath_tot = sum(F.nbath for F in Fragments_list)

        if self.eks.shape[1] != nimp_tot or self.eks.shape[2] != nimp_tot:
            raise ValueError(f"ek_list second and third dimensions must be {nimp_tot}, got {self.eks.shape}")

        self.Rtot = block_diag(*[F.R for F in Fragments_list])
        self.Ltot = block_diag(*[F.Lambda for F in Fragments_list])

        #Check if can be jitted
        def qp_density( mu, T, Lqp, Rqp, ek_qp, wk_qp):
            dens=0.0
            Lmu = Lqp - mu* Rqp @ np.eye(Rqp.shape[1]) @Rqp.T.conj()
            for ek,wk in zip(ek_qp, wk_qp):
                Hk_qp = Rqp @ ek @ Rqp.T.conj() + Lmu
                ekvals = np.linalg.eigvalsh(Hk_qp)
                dens += np.sum(calc_Fermi(ekvals/T))*wk
            return dens/np.sum(wk_qp)
        print('start_qp_dens:',qp_density( 0.0, Tuse, self.Ltot, self.Rtot, self.eks, self.wks))

        nqp_target = 0.5*(nbath_tot-nimp_tot) + n_target
        try:
            def residual(mu):
                return qp_density(mu, Tuse, self.Ltot, self.Rtot, self.eks, self.wks) - nqp_target

            a, b = -10.0, 10.0
            for _ in range(200):
                if residual(a) * residual(b) < 0:
                    break
                a -= 10.0
                b += 10.0
            dmu_target =  bisect(f=residual, a=a, b=b, xtol=1e-5)
            mu_target  = mu_old + dmu_target
            for F in Fragments_list:
                F.Lambda -= F.R @ ( dmu_target * np.eye(F.nimp)) @ F.R.T.conj()
        except:
            mu_target=None

        return mu_target



    def fit_mu_fragment(self, n_target, Fragments_list, T=1e-2, nsteps=10, dmu0=1e-2, ntol=1e-4, mu_old=0.0, spin_pen=0.0):
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")
        if T < 0.0: raise ValueError("Temperature T must be non-negative")
        Tuse=np.maximum(1e-3,T)
        nfill_old = sum(F.nfill for F in Fragments_list)
        dmu = dmu0 * np.sign(nfill_old - n_target)
        mu_o = mu_old
        mu_n = mu_o + dmu
        for _ in range(nsteps):
            for F in Fragments_list:
                F.solve_impurity(mu_n, num_eig=1, T=Tuse, spin_pen=spin_pen)
            nfill_new = sum(F.nfill for F in Fragments_list)

            if abs(nfill_new - n_target) < ntol:
                break

            dnfill = nfill_new - nfill_old
            if abs(dnfill) > 1e-14:
                mu_interp = mu_o + (mu_n - mu_o) * (n_target - nfill_old) / dnfill
            else:
                mu_interp = mu_n + dmu

            mu_o, nfill_old = mu_n, nfill_new
            mu_n = mu_interp

        return mu_n
    
    def compute_ekin(self, Fragments_list, T):
        """ Compute kinetic energy from the quasiparticle part
        """
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")
        if(T<0.0): raise ValueError("Temperature T must be non-negative")
        nimp_tot = sum(F.nimp for F in Fragments_list)
        nbath_tot = sum(F.nbath for F in Fragments_list)

        if self.eks.shape[1] != nimp_tot or self.eks.shape[2] != nimp_tot:
            raise ValueError(f"ek_list second and third dimensions must be {nimp_tot}, got {self.eks.shape}")

        self.Rtot = block_diag(*[F.R for F in Fragments_list])
        self.Ltot = block_diag(*[F.Lambda for F in Fragments_list])

        ekin = 0.0
        Tuse=np.maximum(1e-3,T)
        for ek,wk in zip(self.eks, self.wks):
            Hk_qp = self.Rtot @ ek @ self.Rtot.T.conj() + self.Ltot
            Dk = calc_nf(Hk_qp,Tuse).T
            ekin += wk*np.sum( ( np.dot(self.Rtot, np.dot(ek, self.Rtot.T.conj() ) ) ) * Dk.T )
        return ekin

    def compute_functional(self, Fragments_list , T=1e-2):
        """ Compute the value of the finite temperature functional
        """
        if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
            raise TypeError(f"Fragments_list must be a list of Fragment objects")

        #Embedding part of the functional
        Omega_imps = 0.0
        Omega_qp = 0.0
        Omega_mix = 0.0
        if(T<0.0): raise ValueError("Temperature T must be non-negative")
        # If T=0.0, use a small T to compute the functional
        Tuse=np.maximum(1e-3,T)     
        for F in Fragments_list:
            if F.solver is None:
                raise ValueError("Fragment solver is not set")
            else:
                Omega_imps += -Tuse*np.log( F.solver.Zpart/np.exp(F.solver.gs_ene/Tuse) )
            # if T is small (only Gs and non degenerate) then  Omega_imps = F.solver.gs_ene
        #Quasiparticle part of the functional
        for ek,wk in zip(self.eks, self.wks):
            Hk_qp = self.Rtot @ ek @ self.Rtot.T.conj() + self.Ltot
            ekvals = np.linalg.eigvalsh(Hk_qp)
            Omega_qp += np.sum(np.log(1+np.exp(-ekvals/Tuse) ) )*wk
        Omega_qp *= -Tuse
        #Mixed part of the functional
        for F in Fragments_list:
            H_mix = build_H(F.Lambda, F.Lambda_c, F.D, F.R)
            emix_vals=np.linalg.eigvals(H_mix)
            Omega_mix += np.sum(np.log(1+np.exp(-emix_vals/Tuse) ) )
        Omega_mix *= Tuse

        Omega_tot = Omega_qp+Omega_imps+Omega_mix
        return Omega_tot # , Omega_qp, Omega_imps, Omega_mix