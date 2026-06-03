import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
import time
import h5py

import matplotlib.pyplot as plt


# 1 orbital with 2 spins, B bath per orbital
B = 1
nimp = 2
nbath = nimp*B
ntot = nimp+nbath

Nk = 100
Kx = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
Ky = np.linspace(-np.pi, np.pi, Nk, endpoint=False)

# hopping parameter
t=0.25

# construct ek with 2 sites per unit cell
ek_list = []
for ix,kx in enumerate(Kx):
    for iy,ky in enumerate(Ky):
        gamma_k = -t * (1.0
                    + np.exp(-1j * (kx+ky))
                    + np.exp(-1j * ky)
                    + np.exp(-1j * kx))
        Hk_spinless = np.array([[0.0, gamma_k], [gamma_k.conj(), 0.0]], dtype=np.complex128)
        ek_list.append( np.kron( Hk_spinless , np.eye(2)  ) )


eks = np.array(ek_list)
wks = np.ones(len(ek_list))/len(ek_list)

#compute non-interacting kin-energy
ekin0 = 0.0
for ek, wk in zip(eks, wks):
    evals = np.linalg.eigvalsh(ek)
    ekin0 += wk * np.sum(evals[evals < 0.0])
print(f'Non-interacting kinetic energy: {ekin0:.6f}')




lattice = Lattice(eks, wk_list=wks)

itmax = 50
mix = 0.25
tol = 1e-3
spin_pen=1.0
T = 0.0
mu = 0.0
bfield=1e-2
Tqp=1e-3

Lambda0 = np.kron(np.diag([0.0,-0.5,0.5])[:B,:B], np.eye(2)) + 1e-2*np.kron(np.diag([1.0,1.0,1.0])[:B,:B], np.diag([1.0,-1.0]))
R0 =  np.kron(np.ones((B,1)), np.eye(2))/np.sqrt(B)

U_list = np.linspace(0.1, 3.0, 30)

ZA_list=[]
ZB_list=[]
mA_list=[]
mB_list=[]
doccA_list=[]
doccB_list=[]

for iU, U in enumerate(U_list):
    
    # Hamiltonian of the embedded space
    eloc = np.zeros((nimp, nimp))
    eloc[0,0] = -U/2.
    eloc[1,1] = -U/2.

    # Interaction tensor of the embedded space
    Utensor = np.zeros((nimp, nimp, nimp, nimp))
    Utensor[0,0,1,1] = U
    Utensor[1,1,0,0] = U

    edsolverA = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                         N_sector=ntot//2, dtype=np.complex128)
    edsolverB = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                         N_sector=ntot//2, dtype=np.complex128)
    
    fragmentA = Fragment(nimp, nbath, eloc, Utensor, edsolverA, Lambda=Lambda0, R=R0, verbose=2)
    fragmentB = Fragment(nimp, nbath, eloc, Utensor, edsolverB, Lambda=Lambda0, R=R0, verbose=2)

    for it in range(itmax):
        Dtot, ERDtot = lattice.solve_qp([fragmentA, fragmentB],T=T)

        #print('ERD_tot:')
        #print(ERDtot.real)
        #print('sum ERDtot:',np.sum(ERDtot))

        #print('Delta_qp_A',fragmentA.Delta_qp.real)
        #print('Delta_qp_A',fragmentA.Delta_qp.imag)
        #print('Delta_qp_B',fragmentB.Delta_qp.real)
        #print('Delta_qp_B',fragmentB.Delta_qp.imag)

        #print('ERD_A',fragmentA.ERD.real)
        #print('ERD_A',fragmentA.ERD.imag)
        #print('ERD_B',fragmentB.ERD.real)
        #print('ERD_B',fragmentB.ERD.imag)

        fragmentA.update_hybridization(T=T)
        fragmentB.update_hybridization(T=T)

        print('D_A',fragmentA.D.real)
        print('D_B',fragmentB.D.real)

        #fragmentA.impose_spin_SU2_symmetry()
        #fragmentB.impose_spin_SU2_symmetry()

        if(it<1):
            fragmentA.eloc = eloc + bfield*np.diag([-1,1])
            fragmentB.eloc = eloc - bfield*np.diag([-1,1])
        else:
            fragmentA.eloc = eloc.copy()
            fragmentB.eloc = eloc.copy()


        fragmentA.solve_impurity(mu, T=T)
        fragmentB.solve_impurity(mu, T=T)

        Lambda_old_A = fragmentA.Lambda.copy()
        R_old_A = fragmentA.R.copy()
        Lambda_old_B = fragmentB.Lambda.copy()
        R_old_B = fragmentB.R.copy()

        fragmentA.update_self_energy(T=T)
        fragmentB.update_self_energy(T=T)

        diff_L = max(
            np.abs(fragmentA.R - R_old_A).max(),
            np.abs(fragmentB.R - R_old_B).max(),
        )

        diff_R = max(
            np.abs(fragmentA.R - R_old_A).max(),
            np.abs(fragmentB.R - R_old_B).max(),
        )

        diff = max(diff_L, diff_R)

        #fragmentA.impose_spin_SU2_symmetry()
        #fragmentB.impose_spin_SU2_symmetry()
        dm_A = fragmentA.denMat[:nimp, :nimp].real
        dm_B = fragmentB.denMat[:nimp, :nimp].real
        mA = dm_A[0, 0] - dm_A[1, 1]
        mB = dm_B[0, 0] - dm_B[1, 1]
        print(f'U={U:.2f} it={it} diff_R={diff_R:.2e} diff_L={diff_L:.2e}')
        print(f'm_A={mA:.4f} m_B={mB:.4f}')
        fragmentA.Lambda = (1 - mix) * fragmentA.Lambda + mix * Lambda_old_A
        fragmentA.R = (1 - mix) * fragmentA.R + mix * R_old_A
        fragmentB.Lambda = (1 - mix) * fragmentB.Lambda + mix * Lambda_old_B
        fragmentB.R = (1 - mix) * fragmentB.R + mix * R_old_B

        if (diff < tol and it > 4) or it == itmax - 1:
            print(f'Converged at iteration {it} with diff={diff:.2e} and magnetization m_A={mA:.4f} m_B={mB:.4f}')
            time.sleep(1)
            break

    print('convg RA',fragmentA.R)
    print('convg RB',fragmentB.R)

    print('convg LA',fragmentA.Lambda)
    print('convg LB',fragmentB.Lambda)

    #compute Z
    Z_A = fragmentA.compute_Z()[0,0]
    Z_B = fragmentB.compute_Z()[0,0]
    print(f'U={U:.2f} Z_A={Z_A:.3f} Z_B={Z_B:.3f}')
    ZA_list.append(Z_A)
    ZB_list.append(Z_B)

    # magnetization from impurity 1bdm: m = n_up - n_down = denMat[0,0] - denMat[1,1]
    dm_A = fragmentA.denMat[:nimp, :nimp].real
    dm_B = fragmentB.denMat[:nimp, :nimp].real
    mA = dm_A[0, 0] - dm_A[1, 1]
    mB = dm_B[0, 0] - dm_B[1, 1]
    print(f'U={U:.2f} m_A={mA:.4f} m_B={mB:.4f}')
    mA_list.append(mA)
    mB_list.append(mB)

    docc_A = edsolverA.calc_double_occ(0)
    docc_B = edsolverB.calc_double_occ(0)
    print(f'U={U:.2f} docc_A={docc_A:.4f} docc_B={docc_B:.4f}')
    doccA_list.append(docc_A)
    doccB_list.append(docc_B)

with h5py.File(f'Square_GS_B{B}.h5', 'w') as h5f:
    h5f.create_dataset('U_list',  data=U_list)
    h5f.create_dataset('Z_A',     data=np.array(ZA_list))
    h5f.create_dataset('Z_B',     data=np.array(ZB_list))
    h5f.create_dataset('m_A',     data=np.array(mA_list))
    h5f.create_dataset('m_B',     data=np.array(mB_list))
    h5f.create_dataset('docc_A',  data=np.array(doccA_list))
    h5f.create_dataset('docc_B',  data=np.array(doccB_list))
print(f'Saved Square_GS_B{B}.h5')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(U_list, ZA_list, label='Z_A', marker='x')
ax1.plot(U_list, ZB_list, label='Z_B', marker='+')
ax1.set_xlabel('U')
ax1.set_ylabel('Z')
ax1.set_title('Quasiparticle weight')
ax1.legend()

mA_list=np.array(mA_list)
mB_list=np.array(mB_list)

ax2.plot(U_list, mA_list, label='m_A', marker='x')
ax2.plot(U_list, -mB_list, label='-m_B', marker='+')
ax2.set_xlabel('U')
ax2.set_ylabel('m = n↑ - n↓')
ax2.set_title('Magnetization (T=0)')
ax2.legend()

fig.suptitle('Square lattice, 1 orbital, AFM order, 2 fragments')
plt.tight_layout()
plt.show()
