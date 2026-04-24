import numpy as np
from scipy.linalg import block_diag
from .core import Fragment, Lattice
from triqs_ghostGA.utility.utilities import calc_nf

def solve_qp(self, Fragments_list):
    if not isinstance(Fragments_list, list) or not all(isinstance(F, Fragment) for F in Fragments_list):
        raise TypeError(f"Fragments_list must be a list of Fragment objects")

    nimp_tot = sum(F.nimp for F in Fragments_list)
    nbath_tot = sum(F.nbath for F in Fragments_list)

    if self.eks.shape[1] != nimp_tot or self.eks.shape[2] != nimp_tot:
        raise ValueError(f"ek_list second and third dimensions must be {nimp_tot}, got {self.eks.shape}")

    self.Rtot = block_diag(*[F.R for F in Fragments_list])
    self.Ltot = block_diag(*[F.Lambda for F in Fragments_list])



    self.Delta_p_tot = np.zeros( (nbath_tot,nbath_tot) )
    self.ERD_tot   = np.zeros( (nimp_tot ,nbath_tot) )
    for ek,wk in zip(self.eks, self.wks):
        Hk_qp = self.Rtot @ ek @ self.Rtot.T.conj() + self.Ltot
        Dk = calc_nf(Hk_qp,self.T).T
        self.Delta_p_tot += wk*Dk
        self.ERD_tot   += wk*(ek @ self.Rtot.T.conj() @ Dk)

    imp_stride=0
    bath_stride=0
    for F in Fragments_list:
        F.Delta_p = self.Delta_p_tot[ bath_stride:bath_stride+F.nbath , bath_stride:bath_stride+F.nbath ]
        F.ERD = self.ERD_tot[ imp_stride:imp_stride+F.nimp , bath_stride:bath_stride+F.nbath ]
        imp_stride  += F.nimp
        bath_stride += F.nbath
    return self.Delta_p_tot, self.ERD_tot