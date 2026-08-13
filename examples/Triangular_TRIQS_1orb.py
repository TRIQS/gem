# This is an example of a single orbital Hubbard model solved with ghost-GA on the triangular lattice.
# The dispersion is generated with TRIQS's TBLattice class
#
#
#

import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
import time
import matplotlib.pyplot as plt

import numpy as np


# 1 orbital with 2 spins, 3 bath per orbital, total 8
B = 3
nimp = 2
nbath = nimp*B
ntot = nimp+nbath


# Creating tight-binding using TRIQS, extracting the dispersion and creating the gem.Lattice object
from triqs.lattice.tight_binding import TBLattice
# Triangular lattice primitive vectors (2D, embedded in 3D)
# primitive vectors in Cartesian coordinates
a1 = np.array([1.0, 0.0, 0.0])
a2 = np.array([0.5, np.sqrt(3.0) / 2.0, 0.0])
# hopping amplitude
t = 1.0
# Creating the tight-binding model for the triangular lattice with nearest-neighbor hopping
H_t = TBLattice(
    units=[a1, a2],
    hoppings={
        (+1,  0): [[-t]],
        (-1,  0): [[-t]],
        ( 0, +1): [[-t]],
        ( 0, -1): [[-t]],
        (+1, -1): [[-t]],
        (-1, +1): [[-t]],
    }
)

# Build the k-grid
Nk=64
kmesh = H_t.get_kmesh((Nk,Nk,1))
kpts = np.array(list(kmesh.values()))

# List of H(k) matrices on the triangular-lattice BZ
Hk_list = np.array([ np.kron(H_t.fourier(k),np.eye(2)) for k in kpts])
print('Hk_list:',Hk_list.shape)

lattice = Lattice(Hk_list)

itmax = 100
mix = 0.2
tol = 1e-5
spin_pen=1.0
T = 0.0
mu = 0.0
fit_mu = True
n_target = 1.0

Lambda0 = None
R0 = None
U_list = np.linspace(1.0, 15.0, 15)
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
                        N_sector=ntot//2, Sz_sector=0, dtype=np.complex128,
                        solver_params={'num_eig': 10, 'spin_pen': spin_pen})
    fragment = Fragment(nimp, nbath, eloc, Utensor, edsolver, Lambda=Lambda0, R=R0, verbose=2)

    for it in range(itmax):
        lattice.solve_qp([fragment], T=T)
        fragment.update_hybridization(T=T)

        fragment.impose_spin_SU2_symmetry()
        

        fragment.solve_impurity(mu, T=T)

        if( (abs( fragment.nfill-n_target)>1e-3) and fit_mu ):
            print('fit_mu:')
            mu_new = lattice.fit_mu( n_target , [fragment], T=0.0, mu_old=mu, mode='imp', ntol=1e-4)
            print(' --> mu=',mu_new)
            mu=mu_new
        
        
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
            print(f'Fragment density: {fragment.nfill}' )
            print(f"----- ghost-RISB converged with diff={diff} -----")
            break


    Z = fragment.compute_Z()
    print(f'Done with U={U} returning Z={np.diag(Z.real)}')
    time.sleep(1)

    Lambda0 = fragment.Lambda.copy()
    R0 = fragment.R.copy()
    Z_list.append(Z[0,0].real)

#Here we show how to interface GEM with TRIQS by passing the self-energy coming from GEM into a TRQIS Green's function object

#Initialize the mesh


plt.figure()
plt.plot(U_list, Z_list,marker='.')
plt.ylabel('Z')
plt.xlabel('U/t')
plt.xlim(0,None)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(f'trZ_vs_U_B{B}.png', dpi=100)
plt.show()
