import numpy as np
from triqs_ghostGA.fragment import Fragment
from triqs_ghostGA.lattice import Lattice
from triqs_ghostGA.solvers.simple_ed import SimpleED
import time

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
Tqp=3e-3
mu = 0.0
bfield=1e-2

Lambda0 = np.zeros((2,2),dtype=np.complex128)
R0 =  np.eye(2,dtype=np.complex128)

U_list = np.linspace(0.1, 3.0, 30)

ZA_list=[]
ZB_list=[]
mA_list=[]
mB_list=[]

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
                         N_sector=ntot//2, Sz_sector=0, dtype=np.complex128)
    edsolverB = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                         N_sector=ntot//2, Sz_sector=0, dtype=np.complex128)
    fragmentA = Fragment(nimp, nbath, eloc, Utensor, edsolverA, Lambda=Lambda0, R=R0, verbose=2)
    fragmentB = Fragment(nimp, nbath, eloc, Utensor, edsolverB, Lambda=Lambda0, R=R0, verbose=2)

    for it in range(itmax):
        Dtot, ERDtot = lattice.solve_qp([fragmentA, fragmentB], T=Tqp)

        print('ERD_tot:')
        print(ERDtot.real)
        print('sum ERDtot:',np.sum(ERDtot))

        print('Delta_qp_A',fragmentA.Delta_qp.real)
        print('Delta_qp_A',fragmentA.Delta_qp.imag)
        print('Delta_qp_B',fragmentB.Delta_qp.real)
        print('Delta_qp_B',fragmentB.Delta_qp.imag)

        print('ERD_A',fragmentA.ERD.real)
        print('ERD_A',fragmentA.ERD.imag)
        print('ERD_B',fragmentB.ERD.real)
        print('ERD_B',fragmentB.ERD.imag)

        fragmentA.update_hybridization(T=T)
        fragmentB.update_hybridization(T=T)

        print('D_A',fragmentA.D.real)
        print('D_B',fragmentB.D.real)

        #fragmentA.impose_spin_SU2_symmetry()
        #fragmentB.impose_spin_SU2_symmetry()

        if(it<3):
            fragmentA.eloc = eloc + bfield*np.diag([-1,1])
            fragmentB.eloc = eloc - bfield*np.diag([-1,1])
        else:
            fragmentA.eloc = eloc.copy()
            fragmentB.eloc = eloc.copy()


        fragmentA.solve_impurity(mu, T=T, num_eig=10, spin_pen=spin_pen)
        fragmentB.solve_impurity(mu, T=T, num_eig=10, spin_pen=spin_pen)

        Lambda_old_A = fragmentA.Lambda.copy()
        R_old_A = fragmentA.R.copy()
        Lambda_old_B = fragmentB.Lambda.copy()
        R_old_B = fragmentB.R.copy()

        fragmentA.update_self_energy(T=T)
        fragmentB.update_self_energy(T=T)

        diff = max(
            np.abs(fragmentA.Lambda - Lambda_old_A).max(),
            np.abs(fragmentA.R - R_old_A).max(),
            np.abs(fragmentB.Lambda - Lambda_old_B).max(),
            np.abs(fragmentB.R - R_old_B).max(),
        )
        #fragmentA.impose_spin_SU2_symmetry()
        #fragmentB.impose_spin_SU2_symmetry()
        print(f'U={U:.2f} it={it} diff={diff:.2e}')
        fragmentA.Lambda = (1 - mix) * fragmentA.Lambda + mix * Lambda_old_A
        fragmentA.R = (1 - mix) * fragmentA.R + mix * R_old_A
        fragmentB.Lambda = (1 - mix) * fragmentB.Lambda + mix * Lambda_old_B
        fragmentB.R = (1 - mix) * fragmentB.R + mix * R_old_B

        if (diff < tol and it > 2) or it == itmax - 1:
            print(f'Converged at iteration {it} with diff={diff:.2e}')
            time.sleep(1)
            break

    print('convg RA',fragmentA.R)
    print('convg RB',fragmentB.R)

    print('convg LA',fragmentA.Lambda)
    print('convg LB',fragmentB.Lambda)

    #Compute Gloc and plot
    if(False):
        w_list = np.linspace(-20*t, 20*t, 401)

        Gloc = lattice.compute_Gloc( w_list, [fragmentA, fragmentB])
        Gloc_up = Gloc[:, 0, 0]+Gloc[:, 2, 2]
        Gloc_dw = Gloc[:, 1, 1]+Gloc[:, 3, 3]
        Gloc_A  = Gloc[:, 0, 0] + Gloc[:, 1, 1]  # A sublattice local G (sum over spins)
        Gloc_B  = Gloc[:, 2, 2] + Gloc[:, 3, 3]  # B sublattice local G (sum over spins)
        Gloc_sum = Gloc_up + Gloc_dw

        plt.figure(figsize=(12, 4))
        plt.plot(w_list, -Gloc_up.imag, label='-Im G_loc↑',marker='x')
        plt.plot(w_list, -Gloc_dw.imag, label='-Im G_loc↓',marker='+')
        plt.plot(w_list, -Gloc_sum.imag, label='-Im G_loc↑ + -Im G_loc↓')
        plt.xlabel('ω')
        plt.ylabel('-Im G_loc')
        plt.title(f'U={U:.2f} Square lattice with 1 orbital, 2 fragments')
        plt.legend()
        plt.show()  

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
