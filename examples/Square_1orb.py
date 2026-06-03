import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
import time
from scipy.special import ellipk

import matplotlib.pyplot as plt

# 1 orbital with 2 spins, 3 bath per orbital, total 8
B = 1
nimp = 2
nbath = nimp*B
ntot = nimp+nbath

# construct ek with square lattice dispersion
# solving the elliptic integral
t = 0.25
e_list = np.linspace(-1, 1, 2000)

#check that e=0 is not inside
if( np.any(np.isclose(e_list,0.0)) ):
    raise ValueError('e=0 is inside the list, this is a problem for the elliptic integral')
wks =[]
for ie,e in enumerate(e_list):
    #HERE WE SOLVE THE ELLIPTIC INTEGRAL TO GET THE EXACT DOS OF THE SQUARE LATTICE
    wk = (1.0/(2*np.pi**2*t)*np.real(ellipk(1-e**2)))
    wks.append(wk)
    #It gives all 0, it must be wrong, write the correct one


wks = np.array(wks)
wks /= np.sum(wks)


eks = []
for e in e_list:
    tmp = np.array([[1.0*e]],dtype=np.complex128)
    tmp = np.kron(tmp,np.eye(2))
    eks.append(tmp)
eks = np.array(eks)


#compute non-interacting kin-energy
ekin0 = 0.0
for ek, wk in zip(eks, wks):
    evals = np.linalg.eigvalsh(ek)
    ekin0 += wk * np.sum(evals[evals < 0.0])
print(f'Non-interacting kinetic energy: {ekin0:.6f}')

lattice = Lattice(eks, wk_list=wks)

itmax = 50
mix = 0.2
tol = 1e-5
spin_pen=1.0
T = 0.0
mu = 0.0

Lambda0 = np.eye(2)*0.0
R0 =  np.eye(2)
U_list = np.array([1.5]) # np.linspace(0.1, 2.0, 20)
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

        print('Delta_qp',fragment.Delta_qp.real)
        print('Delta_qp',fragment.Delta_qp.imag)

        print('ERD:',fragment.ERD.real)
        print('ERD:',fragment.ERD.imag)

        fragment.update_hybridization(T=T)

        print('D',fragment.D.real)

        fragment.impose_spin_SU2_symmetry()

        fragment.solve_impurity(mu, T=T, num_eig=10, spin_pen=spin_pen)

        print('Delta_aim:',fragment.Delta_aim.real)
        print('Delta_aim:',fragment.Delta_aim.imag)



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
    print('convg R',fragment.R)
    print('convg L',fragment.Lambda)
    
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
