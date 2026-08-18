######################################################################
# Example 2
# Finite temperature calculations for the half-filled Hubbard model on
# the square lattice, described with two fragments (A/B sublattices) to
# allow for antiferromagnetic order. The dispersion is generated with
# TRIQS's TBLattice class. One U per run: the temperature scan is stored
# in the group U<U>_B<B> of the output file, so repeated runs at different
# U accumulate in the same file. The critical temperatures and exponents
# are extracted afterwards by fit_Tc.py.
######################################################################

import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
import h5py

# Parameters that determine the physical dimensions
# 1 orbital with 2 spins, B=3 bath per orbital, total 8
B = 3
nimp = 2
nbath = nimp * B
ntot = nimp + nbath

# Physical Parameters
U = 0.8
T_list = np.linspace(0.0, 0.10, 51)
mu = 0.0
t = 0.25
bfield = 1e-2 #small magnetic seed removed after few iterations

# Self-consistency Parameters
itmax = 200
mix = 0.05
tol = 1e-3
move_pen = 1e-8
# smearing used in the quasiparticle problem, needed because T_list starts at 0
Tsmearing = 1e-3
# Output file
h5_file = f'data_Square_1orb_2frag_B{B}_phase.h5'


from triqs.lattice.tight_binding import TBLattice

# Two-site (Neel) unit cell of the square lattice: A at (0,0), B at (1,0),
# primitive vectors along the diagonals so that the two sublattices alternate.
a1 = np.array([1.0,  1.0, 0.0])
a2 = np.array([1.0, -1.0, 0.0])

H_t = TBLattice(
    units=[a1, a2],
    hoppings={
        ( 0,  0): [[0.0, -t], [-t, 0.0]],
        (-1, -1): [[0.0, -t], [0.0, 0.0]],
        ( 1,  1): [[0.0, 0.0], [-t, 0.0]],
        ( 0, -1): [[0.0, -t], [0.0, 0.0]],
        ( 0,  1): [[0.0, 0.0], [-t, 0.0]],
        (-1,  0): [[0.0, -t], [0.0, 0.0]],
        ( 1,  0): [[0.0, 0.0], [-t, 0.0]],
    },
    orbital_positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    orbital_names=['A', 'B'],
)

# Build the k-grid
Nk = 200
kmesh = H_t.get_kmesh((Nk, Nk, 1))
kpts_cart = np.array(list(kmesh.values()))
recip = np.array(kmesh.bz.units)              # rows: reciprocal lattice vectors
kpts = kpts_cart @ np.linalg.inv(recip)

# List of H(k) matrices on the square-lattice BZ, one spin block per sublattice
eks = np.array([np.kron(H_t.fourier(k), np.eye(2)) for k in kpts])
lattice = Lattice(eks)


# Local Hamiltonian
eloc = np.zeros((nimp, nimp))
eloc[0, 0] = -U / 2.
eloc[1, 1] = -U / 2.
# Interaction tensor of the embedded space
Utensor = np.zeros((nimp, nimp, nimp, nimp))
Utensor[0, 0, 1, 1] = U
Utensor[1, 1, 0, 0] = U

# Let the fragment decide a starting value
Lambda_A = None; R_A = None
Lambda_c_A = None; D_A = None
Lambda_B = None; R_B = None
Lambda_c_B = None; D_B = None

#Initalize solvers and fragments for the two sublattices
edsolverA = SimpleED(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
edsolverB = SimpleED(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
fragmentA = Fragment(nimp, nbath, eloc, Utensor, edsolverA, verbose=0,
                        Lambda=Lambda_A, R=R_A, Lambda_c=Lambda_c_A, D=D_A)
fragmentB = Fragment(nimp, nbath, eloc, Utensor, edsolverB, verbose=0,
                        Lambda=Lambda_B, R=R_B, Lambda_c=Lambda_c_B, D=D_B)


# create the file if not present; each U group is appended as it completes
with h5py.File(h5_file, 'a') as _:  # 'a' append, 'w' write and delete previous
    pass

# mag_grid[:,iT] = [ m_A, m_B ] order parameter on sublattice A and B
mag_grid = np.zeros( (2,len(T_list)) )
magA = 0.0; magB = 0.0

nT = len(T_list)
arr_R_A      = np.zeros((nT, nbath, nimp), dtype=np.complex128)
arr_R_B      = np.zeros((nT, nbath, nimp), dtype=np.complex128)
arr_Lambda_A = np.zeros((nT, nbath, nbath), dtype=np.complex128)
arr_Lambda_B = np.zeros((nT, nbath, nbath), dtype=np.complex128)

for iT, T in enumerate(T_list):
    print('--------------------------------------------------------')
    print(f'GEM loop started with U={U} and T={T}')

    # Self consistency loop
    for it in range(itmax):
        print(f"----- ghost-RISB iteration {it} / {itmax} -----")

        lattice.solve_qp([fragmentA, fragmentB], T=T, Tsmearing=Tsmearing)

        fragmentA.update_hybridization(T=T, use_Sz=True, move_pen=move_pen)
        fragmentB.update_hybridization(T=T, use_Sz=True,move_pen=move_pen)

        # seed AFM only at lowest T for the first few iterations
        if iT == 0 and it < 3:
            fragmentA.eloc = eloc + bfield * np.diag([-1, 1])
            fragmentB.eloc = eloc - bfield * np.diag([-1, 1])
        else:
            fragmentA.eloc = eloc.copy()
            fragmentB.eloc = eloc.copy()

        fragmentA.solve_impurity(mu, T=T)
        fragmentB.solve_impurity(mu, T=T)

        Lambda_old_A = fragmentA.Lambda.copy(); R_old_A = fragmentA.R.copy()
        Lambda_old_B = fragmentB.Lambda.copy(); R_old_B = fragmentB.R.copy()

        fragmentA.update_self_energy(T=T, use_Sz=True, move_pen=move_pen)
        fragmentB.update_self_energy(T=T, use_Sz=True, move_pen=move_pen)

        diff_LR = max(
            np.abs(fragmentA.Lambda - Lambda_old_A).max(),
            np.abs(np.abs(fragmentA.R) - np.abs(R_old_A)).max(),
            np.abs(fragmentB.Lambda - Lambda_old_B).max(),
            np.abs(np.abs(fragmentB.R) - np.abs(R_old_B)).max(),
        )

        # Mixing of Lambda and R
        fragmentA.Lambda = (1 - mix) * fragmentA.Lambda + mix * Lambda_old_A
        fragmentA.R = (1 - mix) * fragmentA.R + mix * R_old_A
        fragmentB.Lambda = (1 - mix) * fragmentB.Lambda + mix * Lambda_old_B
        fragmentB.R = (1 - mix) * fragmentB.R + mix * R_old_B

        magA_old = magA
        magB_old = magB

        magA = (fragmentA.denMat[0, 0].real - fragmentA.denMat[1, 1].real)
        magB = (fragmentB.denMat[0, 0].real - fragmentB.denMat[1, 1].real)

        diff_mA = abs(magA - magA_old)
        diff_mB = abs(magB - magB_old)
        diff = max(diff_LR, 10 * diff_mA, 10 * diff_mB)

        print(f'iteration: {it}  diff={diff:.2e} '
                f'diff_mA={diff_mA:.4f} diff_mB={diff_mB:.4f} '
                f'magA={magA:.4f} magB={magB:.4f} '
                f'nfillA={fragmentA.nfill:.4f} nfillB={fragmentB.nfill:.4f} ')

        if (diff < tol and it > 1) or it == itmax - 1:
            print(f"--- Exiting after {it}/{itmax} - diff={diff:.2e}  ---")
            break

    magA = (fragmentA.denMat[0,0].real - fragmentA.denMat[1,1].real)
    magB = (fragmentB.denMat[0,0].real - fragmentB.denMat[1,1].real)
    mag_grid[0,iT] = magA; mag_grid[1,iT] = magB

    arr_R_A[iT]      = fragmentA.R
    arr_R_B[iT]      = fragmentB.R
    arr_Lambda_A[iT] = fragmentA.Lambda
    arr_Lambda_B[iT] = fragmentB.Lambda

    print('--------------------------------------------------------')
    print(f'GEM loop ended with U={U} and T={T}')
    print(f'returning m_A={mag_grid[0,iT]} and m_B={mag_grid[1,iT]}')
    print('--------------------------------------------------------')

# save to output file
with h5py.File(h5_file, 'a') as h5f:
    grp_name = f'U{U:.2f}_B{B}'
    if grp_name in h5f:
        del h5f[grp_name]
    grp = h5f.create_group(grp_name)
    grp.create_dataset('T_list',   data=T_list)
    grp.create_dataset('R_A',      data=arr_R_A)
    grp.create_dataset('R_B',      data=arr_R_B)
    grp.create_dataset('Lambda_A', data=arr_Lambda_A)
    grp.create_dataset('Lambda_B', data=arr_Lambda_B)
    grp.create_dataset('mag_grid',  data=mag_grid)
print(f'Written U={U:.2f} to {h5_file}')

# Post-processing (Tc and beta fits) lives in fit_Tc.py, which scans the
# output file for all the U groups available at this B.
