# Author: Samuele Giuli
# 2026

import numpy as np
from triqs_ghostGA.utils_grisb import calc_rhoks
from triqs_ghostGA.delta_fit   import new_self_energy, new_hybridization

#DO WE WANT GLOBAL OR WE TRUST THE USER TO PASS THE OLD ONES TOO?
_lambda_old=None
_R_old=None

def check_convergence(Lambda_list, R_list, tol=1e-5):
    converged=True
    global _lambda_old
    global _R_old
    if( (not _lambda_old is None) and \
        (not _R_old is None) ):
        for ifrag in range(len(Lambda_list)):
            Lfg, U = np.linalg.eigh(Lambda_list[ifrag])
            Lfg_o, U_o = np.linalg.eigh(_lambda_old[ifrag])
            converged = converged and ( not any(abs(Lfg-Lfg_o)>tol))

            Rfg = U.T.conj() @ R_list[ifrag]
            Rfg_o = U.T.conj() @ _R_old[ifrag]
            converged = converged and (not any(abs(Rfg-Rfg_o)>tol ))

    _lambda_old = Lambda_list.copy()
    _R_old = R_list.copy()
    return converged

#TO UNDERSAND HOW THINGS GET PASSED HERE
#LATER CAN DEAL WITH PARALLELIZATION
def update_hybridization( Lambda_list, R_list, old_Lambda_c_list,old_D_hyb_list, F11_list,F12D_list, something_obe ):
    '''
    This function should take a list fo Lambdas (Lambda_list),
    a list of R (R_list) and the obe object containing the Hk in the basis
    the fragments. N.B. to check that the ordering is correct!
    '''

    # B_list and Dim_list?
    # B_list is the B to be used for each correlated space
    # Dim_list is the number of electronics elevels in each correlated space
    #
    # Make sure we are always passing spinfull Lambdas and R

    BNloc = 0
    slice_list=[]
    Nloc=0
    Bslice_list=[]
    R_all = np.zeros( np.sum(Dim_list*B_list) , np.sum(Dim_list) )
    L_all = np.zeros( np.sum(Dim_list*B_list) , np.sum(Dim_list*B_list) )
    for ifrag in range(Nfrag):
        Dim   = Dim_list[ifrag]
        BDim  = B_list[ifrag]*Dim
        R_all[BNloc:BNloc+BDim,Nloc:Nloc+Dim] = R_list[ifrag]
        L_all[BNloc:BNloc+BDim,BNloc:BNloc+BDim] = L_list[ifrag]
        Nloc  += Dim
        BNloc += BDim
    Hk_qp = np.zeros( (BNloc,BNloc) )
    Delta_qp_klist = []
    #suppose we have an object Hk_list = [Nk][Nfrag+1,Nfrag+1][Dim1,Dim2]
    #Build new Delta_qp and sum_k(eps_k@Rdag@Delta_qp_k)/Nk
    H_qp_klist = []
    Hk_list = obe.Hk
    for ik, Hk in enumerate(Hk_list):
        Hk_qp[...] = 0.0
        Hk_qp = R_all @ Hk @ R_all.T.conj() + L_all
        Delta_qp_klist.append( calc_nf(Hk_qp,T) )
    Delta_qp = sum( Delta_qp_klist)/len( Delta_qp_klist )
    ERTD_qp  = sum( [np.dot( np.dot(Hk_list[x],R_all.conj().T ), Delta_qp_klist[x].T ) for x in range(len(self.rhok_list))] ).T/float(len(self.Delta_qp_klist))
    # Get new Lambda_c and D_hyb
    Lambda_c_new = []
    D_hyb_new = []
    Nloc=0;BNloc=0
    for ifrag in range(Nfrag):
        Dim   =Dim_list[ifrag]
        BDim  =B_list[ifrag]*Dim
        F11_trg  = Delta_qp[BNloc:BNloc+BDim,BNloc:BNloc+BDim]
        F12D_trg = ERTD_qp[Nloc:Nloc+Dim,BNloc:BNloc+BDim].T
        Lambda_c_ifrag, D_hyb_frag, mu_qp = new_hybridization(Lambda_c0=old_Lambda_c_list[ifrag],
                                                       D0=old_D_hyb_list[ifrag],
                                                       Lambda=Lambda_list[ifrag],
                                                       R=R_list[ifrag],
                                                       F11_target=F11_list[ifrag],
                                                       F12D_target=F12D_list[ifrag] #,
                                                       # beta=beta ,
                                                       # method="dF"
        )
        Lambda_c_new.append(Lambda_c_ifrag)
        D_hyb_new.append(D_hyb_ifrag)
        Nloc +=Dim
        BNloc+=BDim
    return Lambda_c_new, D_hyb_new

def update_self_energy( Lambda_c_list,D_hyb_list, old_Lambda_list,old_R_list, F22_list,RTF12_list,  eh_sol ):
    Lambda_c_used = eh_sol.Lambda_c_list
    D_hyb_used    = eh_sol.D_hyb_list
    Delta_bath_list = eh_sol.Delta_bath_list
    Delta_hyb_list  = eh_sol.Delta_hyb_list

    Lambda_new = []
    R_new = []

    for lambda_c, d_hyb, delta_bath, delta_hyb in zip(Lambda_c_used,D_hyb_used,Delta_bath_list,Delta_hyb_list):
        Lambda_ifrag, R_ifrag, mu_qp = new_self_energy(Lambda_0=old_Lambda_list[ifrag],
                                                R0=old_R_list[ifrag],
                                                Lambda_c=Lambda_c_list[ifrag],
                                                D=D_hyb_list[ifrag],
                                                F22_target=F22[ifrag],
                                                RTF12_target=RTF12[ifrag] #,
                                                # beta=beta ,
                                                # method="dF"
        )
        Lambda_new.append(Lambda_ifrag)
        R_new.append(R_ifrag)
    return Lambda_new, R_new
    
