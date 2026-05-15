#!/usr/bin/env python

from triqs_ghostGA.grisb import *
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from triqs_ghostGA.solvers.simple_ed import SimpleED
from  triqs_ghostGA.solvers import svd_solver2
import os
import h5py

# from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '..')
from utils import set_gGA, compute_Z
np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf)


nimp, nbath, ntot = 2, 6, 8

nfix = None
tmp_nfix = 1 if nfix is None else nfix

Ks = np.arange(3, 22, 3)
Us = np.arange(0.2, 3.41, 0.8)

dw_Z = 0.0001
itmax = 25

ci_filename = "ci_data.h5"
svd_filename = "svd_data.h5"

fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=200)


for u, U in enumerate(Us):
    print("###########################")
    print(" U = %.1f" % U)

    done = False
    name = 'B%d_U%.1f_nfix%.1f' % (nbath//2, U, tmp_nfix)
    with h5py.File(ci_filename, 'a') as ci_file:
        if name in ci_file:
            done = True
            ci_denMat = ci_file[name]['denMat'][:]
            ci_ene = ci_file[name]['ene'][()]
            ci_Z = ci_file[name]['Z'][()]
            ci_docc = ci_file[name]['docc'][()]

            print("  CI calculation found:")
            print("    ci_denMat:")
            print(ci_denMat)
            print("    ci_ene: %.12f" % ci_ene)
            print("    ci_Z: %.4f" % ci_Z)
            print("    ci_docc: %.4f" % ci_docc)

    if not done:
        print("  CI calculation not found")
        pass

    print(" CI U = %.1f, Z = %.4f" % (U, ci_Z))
    print()


    svd_Ks = []
    svd_drhos = []
    svd_enes = []
    svd_Zs = []
    svd_doccs = []

    for K in Ks:
        if K is None:
            print("  K = None")
        else:
            print("  K = %d" % K)

        # Check if calculation already done
        done = False
        name = 'B%d_U%.1f_nfix%.1f_K%d' % (nbath//2, U, tmp_nfix, K)
        with h5py.File(svd_filename, 'a') as svd_file:
            if name in svd_file:
                done = True
                svd_denMat = svd_file[name]['denMat'][:]
                svd_ene = svd_file[name]['ene'][()]
                svd_Z = svd_file[name]['Z'][()]
                svd_docc = svd_file[name]['docc'][()]

                print("  Calculation found:")
                print("    svd_denMat:")
                print(svd_denMat)
                print("    svd_ene: %.12f" % svd_ene)
                print("    svd_Z: %.4f" % svd_Z)
                print("    svd_docc: %.4f" % svd_docc)

                svd_Ks.append(K)
                svd_drhos.append(np.linalg.norm(svd_denMat - ci_denMat))
                svd_enes.append(svd_ene)
                svd_Zs.append(svd_Z)
                svd_doccs.append(svd_docc)

        print("  SVD U = %.1f, K = %d, Z = %.4f" % (U, K, svd_Z))

    ax[0].plot(svd_Ks, svd_drhos, linewidth=2, marker='x',
               label=r"$U = %.1f$" % U, color="C%d" % u)

    ax[1].plot(svd_Ks, svd_enes, linewidth=2, marker='x',
               label=r"$U = %.1f$" % U, color="C%d" % u)
    ax[1].plot([Ks[-1], 2*Ks[-1]-Ks[-2]], [ci_ene, ci_ene],
               linewidth=1, linestyle='--', color="C%d" % u)

    ax[2].plot(svd_Ks, svd_Zs, linewidth=2, marker='x',
               label=r"$U = %.1f$" % U, color="C%d" % u)
    ax[2].plot([Ks[-1], 2*Ks[-1]-Ks[-2]], [ci_Z, ci_Z],
               linewidth=1, linestyle='--', color="C%d" % u)

    ax[3].plot(svd_Ks, svd_doccs, linewidth=2, marker='x',
               label=r"$U = %.1f$" % U, color="C%d" % u)
    ax[3].plot([Ks[-1], 2*Ks[-1]-Ks[-2]], [ci_docc, ci_docc],
               linewidth=1, linestyle='--', color="C%d" % u)

ax[0].set_yscale("log")
ax[0].set_xlabel(r"Truncation error $K$")
ax[0].set_ylabel(r"$|\rho^{ED} - \rho^{SVD}_K|$")
ax[0].legend(loc="best")

ax[1].set_xlabel(r"Truncation error $K$")
ax[1].set_ylabel(r"$E_{\text{emb GS}}$")
# ax[1].legend(loc="best")

ax[2].set_xlabel(r"Truncation error $K$")
ax[2].set_ylabel(r"$Z(K, U)$")
# ax[2].legend(loc="best")

ax[3].set_xlabel(r"Truncation error $K$")
ax[3].set_ylabel(r"$\langle n_{\uparrow} n_{\downarrow} \rangle$")

plt.tight_layout()
plt.savefig("conv_obsK_B%d_nfix%.1f.png" % (nbath//2, tmp_nfix))
