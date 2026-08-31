######################################################################
# Example 3
# Hund's physics in the multi-orbital Hubbard-Kanamori model at fixed
# filling n_target on the Bethe lattice.
# Storing the quasiparticle weight Z as a function of U for several
# values of J/U
######################################################################

import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
from gem.utilities import U_matrix_kanamori

import matplotlib.pyplot as plt


B = 3
n_orb = 3
nimp = 2 * n_orb
nbath = nimp * B
ntot = nimp + nbath

# Physical Parameters
U_list = np.linspace(0.1,10.0,100)[:2]
JoverU_list = [0.00,0.10,0.20,0.30]
n_target = n_orb-1
T = 0.0


# Self-consistency Parameters
itmax = 200
mix = 0.1
tol = 1e-4   # tolerance on Lambda,R convergence
ntol = 1e-4  # tolerance on the impurity filling
Tsmearing = 1e-3 # Zero temperature smearing for the QP problem


itmax = 200
# mix is the weight kept from the previous iteration. Refitting mu at every
# iteration makes the loop oscillate between two states unless it is damped
# heavily, hence a value much larger than in the examples at fixed filling.
tol = 1e-4
ntol = 1e-4            # tolerance on the impurity filling
spin_pen = 0.0          # must stay zero: a S^2 penalty would kill Hund's physics
Tsmearing = 1e-3

# Non-interacting density of states and lattice object
e_list = np.linspace(-1, 1, 1001)
wks = np.sqrt(1 - e_list**2)
wks /= np.sum(wks)
eks = e_list[:, None, None] * np.eye(nimp, dtype=np.complex128)
lattice = Lattice(eks, wk_list=wks)


def check_convergence(R_new, L_new, R_old, L_old):
    # Only 1 spin and gauge invariant difference
    L_eval_new, UL_new = np.linalg.eigh(L_new[::2*n_orb, ::2*n_orb])
    L_eval_old, UL_old = np.linalg.eigh(L_old[::2*n_orb, ::2*n_orb])
    diff_R = np.abs(np.abs(UL_old @ R_old[::2*n_orb, ::2*n_orb]) - np.abs(UL_new @ R_new[::2*n_orb, ::2*n_orb])).max()
    diff_Lambda = np.abs(L_eval_new - L_eval_old).max()
    diff = max(diff_R, diff_Lambda)
    return diff


# Zgrid[iJ, iU] = quasiparticle weight
Zgrid = np.zeros((len(JoverU_list), len(U_list)))
ngrid = np.zeros((len(JoverU_list), len(U_list)))

for iJ, JoverU in enumerate(JoverU_list):

    # warm-start: carry the converged Lambda/R and mu from one U to the next
    Lambda0 = None
    R0 = None
    mu=0
    for iU, U in enumerate(U_list):
        J = JoverU * U
        print('--------------------------------------------------------')
        print(f'GEM loop started with U={U:.2f} and J={J:.2f} (J/U={JoverU:.2f})')

        # Local Hamiltonian: the level position is set by mu, fitted to n_target
        eloc = np.zeros((nimp, nimp))
        # Interaction tensor of the embedded space
        Utensor = U_matrix_kanamori(n_orb, U, J) 

        # Fixed sectors for T=0 calculations
        edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                            N_sector=ntot//2, Sz_sector=0, dtype=np.float64)

        # Fragment initialization
        fragment = Fragment(nimp, nbath, eloc, Utensor, edsolver,
                            Lambda=Lambda0, R=R0, verbose=0)

        # Self consistency loop
        for it in range(itmax):
            print(f"----- ghost-RISB iteration {it} / {itmax} -----")

            lattice.solve_qp([fragment], T=T, Tsmearing=Tsmearing)

            fragment.update_hybridization(T=T, use_Sz=True)

            fragment.solve_impurity(mu, T=T)

            #Fit density to n_target
            if abs(fragment.nfill.real - n_target) > ntol:
                mu = lattice.fit_mu(n_target, [fragment], T=T, mu_old=mu,
                                    mode='imp', ntol=1e-5)
                fragment.solve_impurity(mu, T=T)
            nfill=fragment.nfill.real
            Lambda_old = fragment.Lambda.copy(); R_old = fragment.R.copy()

            fragment.update_self_energy(T=T, use_Sz=True)

            # stay in the paramagnetic, orbitally degenerate solution
            fragment.impose_orbital_symmetry()
            fragment.impose_spin_SU2_symmetry()

            Lambda_new = fragment.Lambda; R_new = fragment.R

            diff = check_convergence(R_new, Lambda_new, R_old, Lambda_old)
            print(f"it:{it}  diff={diff:.3e}  mu={mu:.4f}  nfill={nfill:.4f}")

            fragment.Lambda = (1 - mix) * Lambda_new + mix * Lambda_old
            fragment.R = (1 - mix) * R_new + mix * R_old

            if (diff < tol and it > 2) or it == itmax - 1:
                print(f"-- Exiting with diff={diff:.3e} after {it} iterations")
                break

        Z = fragment.compute_Z()
        Zgrid[iJ, iU] = Z.real[0, 0]
        ngrid[iJ, iU] = fragment.nfill.real

        # warm start for the next U
        Lambda0 = fragment.Lambda.copy()
        R0 = fragment.R.copy()

        print('--------------------------------------------------------')
        print(f'GEM loop ended with U={U} and J={J}')
        print(f'returning Z={np.diag(Z.real)}, mu={mu} and n={fragment.nfill.real}')
        print('--------------------------------------------------------')

np.savetxt('B{B}/kanamori_Ulist.dat', U_list)
np.savetxt('B{B}/kanamori_JoverU.dat', np.array(JoverU_list))
np.savetxt('B{B}/kanamori_Zgrid.dat', Zgrid)
np.savetxt('B{B}/kanamori_ngrid.dat', ngrid)

######################################################################
# Z vs U: at quarter filling the atomic charge gap is U-3J, so Hund's
# coupling pushes the Mott transition to much larger U and raises Z at
# fixed U. This is the opposite of what J does at half filling, where
# the gap is U+J: the two faces of Hund's coupling.
######################################################################

plt.figure(figsize=(6, 5))
for iJ, JoverU in enumerate(JoverU_list):
    plt.plot(U_list, Zgrid[iJ], marker='.', label=f'J/U={JoverU:.2f}')
plt.xlabel('U/D')
plt.ylabel('Z')
plt.title(f'Two-orbital Kanamori, n={n_target}, B={B} bath per spin-orbital')
plt.xlim(0, None)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(f'kanamori_Z_vs_U_B{B}.png', dpi=150)
plt.show()
