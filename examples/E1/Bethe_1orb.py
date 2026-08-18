######################################################################
# Example 1
# Finite temperature calculations for the half-filled Hubbard model
# on the Bethe lattice. Storing total energy and entropy per site
######################################################################

import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED

# Parameters that determine the physical dimentions 
# 1 orbital with 2 spins, B=3 bath per orbital, total 8
B = 3
nimp = 2
nbath = nimp*B
ntot = nimp+nbath

# Physical Parameters
U = 2.0
Tlist = np.hstack( (np.array([0.0]), np.logspace(np.log10(1e-2), np.log10(1.0), 41)) )
mu = 0.0

# Self-consistency Parameters
itmax = 300
mix = 0.1
tol = 1e-4
spin_pen = 0.0
Tsmearing = 1e-4

# Non-interacting density of states and lattice object
e_list = np.linspace(-1, 1, 5001)
wks = np.sqrt(1 - e_list**2)
wks /= np.sum(wks)
eks = e_list[:, None, None] * np.eye(2, dtype=np.complex128)

lattice = Lattice(eks, wk_list=wks)

# Local Hamiltonian
eloc = np.zeros((nimp, nimp))
eloc[0,0] = -U/2.
eloc[1,1] = -U/2.
# Interaction tensor of the embedded space
Utensor = np.zeros((nimp, nimp, nimp, nimp))
Utensor[0,0,1,1] = U
Utensor[1,1,0,0] = U

# SimpleED solver initialization
edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                    N_sector=None, Sz_sector=None, dtype=np.float64,
                    solver_params={'spin_pen': spin_pen})

# Fragment initialization
Lambda0 = None; R0 = None
fragment = Fragment(nimp, nbath, eloc, Utensor, edsolver, Lambda=Lambda0, R=R0, verbose=2)

def check_convergence(R_new,L_new, R_old,L_old):
    # Only 1 spin and gauge invariant difference
    L_eval_new, UL_new = np.linalg.eigh(L_new[::2,::2])
    L_eval_old, UL_old = np.linalg.eigh(L_old[::2,::2])
    diff_R = np.abs(np.abs(UL_old @ R_old[::2,::2]) - np.abs(UL_new @ R_new[::2,::2])).max()
    diff_Lambda = np.abs(L_eval_new - L_eval_old).max()
    diff = max(diff_R, diff_Lambda)
    return diff

Elist = []
Slist = []
docclist = []

for T in Tlist:
    print('--------------------------------------')
    print(f'GEM loop started with U={U} and T={T}')

    # Self consistency loop
    for it in range(itmax):
        print(f"----- ghost-RISB iteration {it} / {itmax} -----")

        lattice.solve_qp([fragment], T=T, Tsmearing=Tsmearing)

        fragment.update_hybridization(T=T, use_Sz=True)

        fragment.solve_impurity(mu, T=T)

        Lambda_old = fragment.Lambda.copy()
        R_old = fragment.R.copy()

        fragment.update_self_energy(T=T, use_Sz=True)

        Lambda_new = fragment.Lambda
        R_new = fragment.R

        diff = check_convergence(R_new,Lambda_new, R_old,Lambda_old)
        print(f"iteration: {it}  diff={diff}")

        fragment.Lambda = (1 - mix) * Lambda_new + mix * Lambda_old
        fragment.R = (1 - mix) * R_new + mix * R_old


        if (diff < tol and it > 2) or it == itmax - 1:
            print(f"----- Exiting loop with diff={diff} after {it} iterations (max={itmax}) -----")
            break


    docc = fragment.E2loc/U
    docclist.append(docc)
    ekin = lattice.compute_ekin([fragment],T=T,Tsmearing=Tsmearing)
    eimp = fragment.compute_energy()
    etot = (eimp+ekin).real
    # Free-energy functional
    L = lattice.compute_functional([fragment],T=T, Tsmearing=Tsmearing).real
    S = (etot-L)/T if T > 0 else 0.0

    Elist.append(etot)
    Slist.append(S)

    print('--------------------------------------')
    print(f'GEM loop ended with U={U} and T={T}')
    print(f'returning docc={docc}, Etot={etot} and S={S}')
    print('--------------------------------------')


np.savetxt('Tlist.dat',Tlist)
np.savetxt('Elist.dat',Elist)
np.savetxt('Slist.dat',Slist)
np.savetxt('docclist.dat',docclist)
