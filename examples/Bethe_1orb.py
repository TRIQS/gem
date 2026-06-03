import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
import time

import matplotlib.pyplot as plt

# 1 orbital with 2 spins, 3 bath per orbital, total 8
B = 3
nimp = 2
nbath = nimp*B
ntot = nimp+nbath

# construct ek with semicircular DOS
e_list = np.linspace(-1, 1, 5001)
wks = np.sqrt(1 - e_list**2)
wks /= np.sum(wks)
eks = []
for e in e_list:
    tmp = np.array([[1.0*e]],dtype=np.complex128)
    tmp = np.kron(tmp,np.eye(2))
    eks.append(tmp)
eks = np.array(eks)

lattice = Lattice(eks, wk_list=wks)

itmax = 100
mix = 0.2
tol = 1e-5
spin_pen=1.0
T = 0.0
mu = 0.0

Lambda0 = None
R0 = None
U_list = np.linspace(2.0, 3.0, 11)
Z_list = []
deg_states = []

for iU, U in enumerate(U_list):

    # Hamiltonian of the embedded space
    eloc = np.zeros((nimp, nimp))
    eloc[0,0] = -U/2.
    eloc[1,1] = -U/2.

    # Interaction tensor of the embedded space
    Utensor = np.zeros((nimp, nimp, nimp, nimp))
    Utensor[0,0,1,1] = U
    Utensor[1,1,0,0] = U

    edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                        N_sector=ntot//2, Sz_sector=0, dtype=np.complex128)
    fragment = Fragment(nimp, nbath, eloc, Utensor, edsolver, Lambda=Lambda0, R=R0, verbose=2)

    for it in range(itmax):
        lattice.solve_qp([fragment], T=T)
        fragment.update_hybridization(T=T)

        fragment.impose_spin_SU2_symmetry()

        fragment.solve_impurity(mu, T=T, num_eig=10, spin_pen=spin_pen)

        Lambda_old = fragment.Lambda.copy()
        R_old = fragment.R.copy()

        fragment.update_self_energy(T=T)

        fragment.impose_spin_SU2_symmetry()

        Lambda_new = fragment.Lambda
        R_new = fragment.R

        L_eval_new, UL_new = np.linalg.eigh(Lambda_new[::2,::2])
        L_eval_old, UL_old = np.linalg.eigh(Lambda_old[::2,::2])

        diff_R = np.abs(np.abs(UL_old @ R_old[::2,::2]) - np.abs(UL_new @ R_new[::2,::2])).max()
        diff_Lambda = np.abs(L_eval_new - L_eval_old).max()
        diff = max(diff_R, diff_Lambda)

        fragment.Lambda = (1 - mix) * Lambda_new + mix * Lambda_old
        fragment.R = (1 - mix) * R_new + mix * R_old

        fragment.impose_spin_SU2_symmetry()

        print(f"iteration: {it}  diff={diff}")

        if (diff < tol and it > 2) or it == itmax - 1:
            print(f'Lambda eigvals: {L_eval_new}')
            print(f'Fragment energy: {fragment.E2loc}')
            print(f"----- ghost-RISB converged with diff={diff} -----")
            break

    deg_states.append(fragment.solver.Tstates)

    Z = fragment.compute_Z()
    print(f'Done with U={U} returning Z={np.diag(Z.real)}')
    time.sleep(1)

    Lambda0 = fragment.Lambda.copy()
    R0 = fragment.R.copy()
    Z_list.append(Z[0,0].real)

plt.figure()
plt.plot(U_list, Z_list)
plt.savefig('Z_vs_U_B{B}.png', dpi=100)
plt.show()

plt.figure()
plt.plot(U_list, deg_states)
plt.savefig('deg_vs_U_B{B}.png', dpi=100)
plt.show()



for iter in range(itmax):
    lattice.solve_qp([fragment], T=T)
    fragment.update_hybridization(T=T)
    # fragment.impose_spin_SU2_symmetry()
    fragment.solve_impurity(mu, T=T, num_eig=10, spin_pen=spin_pen)
    fragment.update_self_energy(T=T)
    #fragment.impose_spin_SU2_symmetry()
    # CHECK CONVERGENCE
