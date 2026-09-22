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
B, nimp = 3, 2
nbath = nimp*B
ntot = nimp+nbath

# Physical Parameters
U, mu = 2.0, 1.0
Tlist = np.hstack((np.array([0]), np.logspace(np.log10(1e-3), np.log10(1), 31)))

# Self-consistency Parameters
itmax, tol, Tsmearing = 100, 1e-5, 1e-3

# Non-interacting density of states and lattice object
# Non-interacting density of states and lattice object for the Bethe lattice
e_list = np.linspace(-1, 1, 5001)
wks = np.sqrt(1 - e_list**2)
wks /= np.sum(wks)
eks = e_list[:, None, None] * np.eye(2, dtype=np.complex128)

lattice = Lattice(eks, wk_list=wks)

# Local Hamiltonian
eloc = np.zeros((nimp, nimp))
# Interaction tensor of the embedded space
Utensor = np.zeros((nimp, nimp, nimp, nimp))
Utensor[0,0,1,1] = U
Utensor[1,1,0,0] = U

# SimpleED solver initialization
edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True, N_sector=None, Sz_sector=None, 
                    dtype=np.float64)

# Fragment initialization
Lambda0 = None; R0 = None
fragment = Fragment(nimp, nbath, eloc, Utensor, edsolver, Lambda=Lambda0, R=R0, 
                    verbose=2)

def check_convergence(R_new,L_new, R_old,L_old):
    # Only 1 spin and gauge invariant difference
    L_eval_new, UL_new = np.linalg.eigh(L_new[::2,::2])
    L_eval_old, UL_old = np.linalg.eigh(L_old[::2,::2])
    diff_R = np.abs(np.abs(UL_old @ R_old[::2,::2]) - np.abs(UL_new @ R_new[::2,::2])).max()
    diff_Lambda = np.abs(L_eval_new - L_eval_old).max()
    diff = max(diff_R, diff_Lambda)
    return diff

Elist, Slist, docclist = [], [], []

for T in Tlist:
    print('--------------------------------------')
    print(f'GEM loop started with U={U} and T={T}')

    # Self consistency loop
    for it in range(itmax):
        print(f'----- ghost-RISB iteration {it} / {itmax} -----')


        lattice.solve_qp([fragment], T=T, Tsmearing=Tsmearing) # Step 1: Solve H_qp
        fragment.update_hybridization(T=T, use_Sz=True) # Step 2: Update Delta
        fragment.solve_impurity(mu, T=T) # Step 3: Solve H_emb
        Lambda_old, R_old = fragment.Lambda.copy(), fragment.R.copy() # Save old Sigma
        fragment.update_self_energy(T=T, use_Sz=True) # Step 4: Update Sigma
        Lambda_new, R_new = fragment.Lambda, fragment.R # Save new Sigma

        diff = check_convergence(R_new, Lambda_new, R_old, Lambda_old)
        print(f'iteration: {it}  diff={diff}')

        if (diff < tol and it > 2) or it == itmax - 1:
            print(f'----- Exiting loop with diff={diff} after '
                  f'{it} iterations (max={itmax}) -----'); break

    docc = fragment.E2loc/U
    docclist.append(docc)
    ekin = lattice.compute_ekin([fragment],T=T,Tsmearing=Tsmearing)
    eimp = fragment.compute_energy()
    etot = (eimp+ekin).real
    # Free-energy functional
    L = lattice.compute_functional([fragment],T=T, Tsmearing=Tsmearing).real
    S = (etot-L)/T if T > 0 else 0.0

    Elist.append(etot); Slist.append(S)

    print('--------------------------------------')
    print(f'GEM loop ended with U={U} and T={T}')
    print(f'returning docc={docc}, Etot={etot} and S={S}')
    print('--------------------------------------')


np.savetxt('Tlist.dat', Tlist); np.savetxt('Elist.dat', Elist)
np.savetxt('Slist.dat', Slist); np.savetxt('docclist.dat', docclist)