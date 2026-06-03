#!/usr/bin/env python
import unittest

from gem.gdmft import *
import numpy as np
from gem.solvers.simple_ed import SimpleED
import os
import h5py


class test_temperature(unittest.TestCase):

    def test_gdmft_temperature(self):

        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        B = 3
        nimp = 2
        nbath = nimp * B
        ntot = nimp + nbath
        T_list = np.logspace(-2, 1, 30)

        # construct ek with semicircular DOS
        e_list = np.linspace(-1, 1, 5001)
        wks = np.sqrt(1 - e_list**2)
        wks /= np.sum(wks)
        eks = []
        for e in e_list:
            tmp = np.array([[1.0 * e]], dtype=np.complex128)
            tmp = np.kron(tmp, np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        # initial value for hybridization
        R0 = np.array([[0.2],[0.6],[0.2]]) + np.random.rand(B,1)*0.05
        R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.diag([0.6, 0, -0.6])
        Lambda0 = np.kron(Lambda0, np.eye(2))

        U = 2.0
        eloc = np.zeros((nimp, nimp))
        eloc[0, 0] = -U / 2.
        eloc[1, 1] = -U / 2.

        Utensor = np.zeros((nimp, nimp, nimp, nimp))
        Utensor[0, 0, 1, 1] = U
        Utensor[1, 1, 0, 0] = U

        edsolver = SimpleED(ntot, use_Ntot=False,
                      use_Sz=False, dtype=np.complex128)

        grisb = Gdmft(ntot, nimp, nbath, eks, eloc, Utensor,
                      wks=wks, R=R0, Lambda=Lambda0, edsolver=edsolver)

        T_results = []
        docc_results = []
        etot_results = []
        func_results = []

        for iT, T in enumerate(T_list):
            print(f"\n***** T={T:.4e} ({iT+1}/{len(T_list)}) *****")
            grisb.run(itmax=50, mix=0.05, tol=1e-4, T= T, silence=True)
            T_results.append(T)
            docc_results.append(np.array(grisb.docc).real)
            grisb.compute_energy(beta=1/T)
            etot_results.append(grisb.etot.real)
            func_results.append(grisb.get_functional().real)

        T_results = np.array(T_results)
        docc_results = np.array(docc_results)  # shape: (n_T, n_orb)
        func_results = np.array(func_results)  # shape: (n_T,)
        etot_results = np.array(etot_results)  # shape: (n_T,)
        S_results = (etot_results - func_results) / T_results  # S = ( E - Omega )/T

        output_dir = os.path.dirname(os.path.abspath(__file__))
        hdf5_path = os.path.join(output_dir, "temperature_results.h5")

        # STORE DATA
        #with h5py.File(hdf5_path, "w") as f:
        #    f.create_dataset("T", data=T_results)
        #    f.create_dataset("docc", data=docc_results)
        #    f.create_dataset("etot", data=etot_results)
        #    f.create_dataset("func", data=func_results)
        #    f.create_dataset("S", data=S_results)

        # PLOT ENTROPY AND DOCC
        #import matplotlib
        #matplotlib.use('Agg')
        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #ax.semilogx(T_results, S_results/np.log(2), 'o-', label='Entropy S/log(2)')
        #ax.set_xlabel("Temperature T")  
        #ax.set_ylabel("Entropy S")
        #ax.set_title(f"Entropy vs Temperature (U={U})")
        #ax.grid(True, which='both', alpha=0.4)
        #fig.tight_layout()
        #fig.savefig(os.path.join(output_dir, "entropy_vs_temperature.png"), dpi=150, bbox_inches='tight')
        #plt.close(fig)
        #
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # n_orb = docc_results.shape[1] if docc_results.ndim > 1 else 1
        # if n_orb == 1:
        #     ax.semilogx(T_results, docc_results[:, 0], 'o-', label='docc')
        # else:
        #     for i in range(n_orb):
        #         ax.semilogx(T_results, docc_results[:, i], 'o-', label=f'orb {i}')
        #     ax.legend()
        # ax.set_xlabel("Temperature T")
        # ax.set_ylabel("Double occupancy")
        # ax.set_title(f"Double occupancy vs Temperature (U={U})")
        # ax.grid(True, which='both', alpha=0.4)
        # fig.tight_layout()
        # fig.savefig(os.path.join(output_dir, "docc_vs_temperature.png"), dpi=150, bbox_inches='tight')
        # plt.close(fig)

        with h5py.File(hdf5_path, "r") as f:
            print("Compare docc")
            np.testing.assert_allclose(docc_results, f["docc"][:], atol=1e-2)
            print("Compare etot")
            np.testing.assert_allclose(etot_results, f["etot"][:], atol=1e-2)
            print("Compare func")
            np.testing.assert_allclose(func_results, f["func"][:], atol=1e-2)
            print("Compare S")
            np.testing.assert_allclose(S_results, f["S"][:], atol=1e-2)


if __name__ == '__main__':
    unittest.main()
