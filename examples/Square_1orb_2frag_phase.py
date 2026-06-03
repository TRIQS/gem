import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py

# 1 orbital with 2 spins, B bath per orbital
B = 3
nimp = 2
nbath = nimp * B
ntot = nimp + nbath

Nk = 200
Kx = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
Ky = np.linspace(-np.pi, np.pi, Nk, endpoint=False)

t = 0.25

ek_list = []
for ix, kx in enumerate(Kx):
    for iy, ky in enumerate(Ky):
        gamma_k = -t * (1.0
                    + np.exp(-1j * (kx + ky))
                    + np.exp(-1j * ky)
                    + np.exp(-1j * kx))
        Hk_spinless = np.array([[0.0, gamma_k], [gamma_k.conj(), 0.0]], dtype=np.complex128)
        ek_list.append(np.kron(Hk_spinless, np.eye(2)))

eks = np.array(ek_list)
wks = np.ones(len(ek_list)) / len(ek_list)

lattice = Lattice(eks, wk_list=wks)

itmax = 100
mix = 0.01
tol = 2e-4
spin_pen = 0.0
mu = 0.0
bfield = 1e-2

U_list = np.array([0.5])
#U_list = np.array([1.0, 1.5])
T_list = np.linspace(0.02, 0.12, 21)

# mag_grid[iU, iT] = |m_A|, order parameter on sublattice A
mag_grid = np.zeros((len(U_list), len(T_list)))

h5_file = 'Square_1orb_2frag_phase.h5'
# create a fresh file; each U group will be appended as it completes
with h5py.File(h5_file, 'a') as _: # 'a' is append, 'w' is want to delete previous
    pass

L_t0 = np.kron( np.diag([0.0,-0.5,0.5])[:B,:B] , np.eye(2) )
R_t0 = np.kron( np.array([[0.8],[0.2],[0.2]])[:B,:] , np.eye(2) )

for iU, U in enumerate(U_list):
    eloc = np.zeros((nimp, nimp))
    eloc[0, 0] = -U / 2.
    eloc[1, 1] = -U / 2.

    Utensor = np.zeros((nimp, nimp, nimp, nimp))
    Utensor[0, 0, 1, 1] = U
    Utensor[1, 1, 0, 0] = U

    # warm-start: carry converged Lambda/R from one T to the next (low to high T)
    Lambda_A = L_t0.copy()
    R_A = R_t0.copy()
    Lambda_B = -L_t0.copy()
    R_B = R_t0.copy()

    nT = len(T_list)
    arr_denMat_A = np.zeros((nT, nimp, nimp), dtype=np.complex128)
    arr_denMat_B = np.zeros((nT, nimp, nimp), dtype=np.complex128)
    arr_R_A      = np.zeros((nT, nbath, nimp), dtype=np.complex128)
    arr_R_B      = np.zeros((nT, nbath, nimp), dtype=np.complex128)
    arr_Lambda_A = np.zeros((nT, nbath, nbath), dtype=np.complex128)
    arr_Lambda_B = np.zeros((nT, nbath, nbath), dtype=np.complex128)
    arr_Z_A      = np.zeros((nT, nimp, nimp), dtype=np.complex128)
    arr_Z_B      = np.zeros((nT, nimp, nimp), dtype=np.complex128)

    for iT, T in enumerate(T_list):
        print(f'Doing U={U:.2f} - T={T:.2f}')
        edsolverA = SimpleED(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
        edsolverB = SimpleED(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
        fragmentA = Fragment(nimp, nbath, eloc, Utensor, edsolverA, Lambda=Lambda_A, R=R_A, verbose=0)
        fragmentB = Fragment(nimp, nbath, eloc, Utensor, edsolverB, Lambda=Lambda_B, R=R_B, verbose=0)

        for it in range(itmax):
            Dtot, ERDtot = lattice.solve_qp([fragmentA, fragmentB], T=T)

            fragmentA.update_hybridization(T=T,move_pen=1e-8)
            fragmentB.update_hybridization(T=T,move_pen=1e-8)

            # seed AFM only at lowest T for the first few iterations
            if iT == 0 and it < 3:
                fragmentA.eloc = eloc + bfield * np.diag([-1, 1])
                fragmentB.eloc = eloc - bfield * np.diag([-1, 1])
            else:
                fragmentA.eloc = eloc.copy()
                fragmentB.eloc = eloc.copy()

            fragmentA.solve_impurity(mu, T=T, num_eig=10, spin_pen=spin_pen)
            fragmentB.solve_impurity(mu, T=T, num_eig=10, spin_pen=spin_pen)

            Lambda_old_A = fragmentA.Lambda.copy()
            R_old_A = fragmentA.R.copy()
            Lambda_old_B = fragmentB.Lambda.copy()
            R_old_B = fragmentB.R.copy()

            fragmentA.update_self_energy(T=T,move_pen=1e-8)
            fragmentB.update_self_energy(T=T,move_pen=1e-8)

            diff = max(
                np.abs(fragmentA.Lambda - Lambda_old_A).max(),
                np.abs(fragmentA.R - R_old_A).max(),
                np.abs(fragmentB.Lambda - Lambda_old_B).max(),
                np.abs(fragmentB.R - R_old_B).max(),
            )

            fragmentA.Lambda = (1 - mix) * fragmentA.Lambda + mix * Lambda_old_A
            fragmentA.R = (1 - mix) * fragmentA.R + mix * R_old_A
            fragmentB.Lambda = (1 - mix) * fragmentB.Lambda + mix * Lambda_old_B
            fragmentB.R = (1 - mix) * fragmentB.R + mix * R_old_B

            if (diff < tol and it > 2) or it == itmax - 1:
                print(f'U={U:.2f} T={T:.4f} converged at it={it} diff={diff:.2e}')
                break

        # save converged Lambda/R as warm start for next T
        Lambda_A = fragmentA.Lambda.copy()
        R_A = fragmentA.R.copy()
        Lambda_B = fragmentB.Lambda.copy()
        R_B = fragmentB.R.copy()

        if(iT==0):
            L_t0 = Lambda_A.copy()
            R_t0 = R_A.copy()

        dm_A = fragmentA.denMat[:nimp, :nimp].real
        mag_grid[iU, iT] = abs(dm_A[0, 0] - dm_A[1, 1])

        arr_denMat_A[iT] = fragmentA.denMat[:nimp,:nimp]
        arr_denMat_B[iT] = fragmentB.denMat[:nimp,:nimp]
        arr_R_A[iT]      = fragmentA.R
        arr_R_B[iT]      = fragmentB.R
        arr_Lambda_A[iT] = fragmentA.Lambda
        arr_Lambda_B[iT] = fragmentB.Lambda
        arr_Z_A[iT]      = fragmentA.compute_Z()
        arr_Z_B[iT]      = fragmentB.compute_Z()

    # flush this U to disk immediately so partial runs are not lost
    with h5py.File(h5_file, 'a') as h5f:
        grp = h5f.create_group(f'U_{U:.2f}')
        grp.create_dataset('T_list',   data=T_list)
        grp.create_dataset('denMat_A', data=arr_denMat_A)
        grp.create_dataset('denMat_B', data=arr_denMat_B)
        grp.create_dataset('R_A',      data=arr_R_A)
        grp.create_dataset('R_B',      data=arr_R_B)
        grp.create_dataset('Lambda_A', data=arr_Lambda_A)
        grp.create_dataset('Lambda_B', data=arr_Lambda_B)
        grp.create_dataset('Z_A',      data=arr_Z_A)
        grp.create_dataset('Z_B',      data=arr_Z_B)
    print(f'Written U={U:.2f} to {h5_file}')

# --- Plot 1: |m| vs T for each U ---
fig1, ax1 = plt.subplots(figsize=(7, 5))
for iU, U in enumerate(U_list):
    ax1.plot(T_list, mag_grid[iU], marker='o', markersize=3, label=f'U={U:.1f}')
ax1.set_xscale('log')
ax1.set_xlabel('T')
ax1.set_ylabel('|m| = |n↑ - n↓|')
ax1.set_title('AFM order parameter vs temperature')
ax1.legend()
fig1.tight_layout()
fig1.savefig('mag_vs_T.png', dpi=150)
print('Saved mag_vs_T.png')

# --- Plot 2: colormap in (U, T) space ---
fig2, ax2 = plt.subplots(figsize=(7, 5))
# use contourf for a smooth phase-diagram look
UU, TT = np.meshgrid(U_list, T_list, indexing='ij')
levels = np.linspace(0, mag_grid.max(), 51)
cf = ax2.contourf(UU, TT, mag_grid, levels=levels, cmap='RdBu_r')
fig2.colorbar(cf, ax=ax2, label='|m|')
ax2.set_yscale('log')
ax2.set_xlabel('U')
ax2.set_ylabel('T')
ax2.set_title('Phase diagram: AFM order parameter')
fig2.tight_layout()
fig2.savefig('phase_diagram.png', dpi=150)
print('Saved phase_diagram.png')

plt.show()
