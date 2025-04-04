#!/usr/bin/env python

from triqs_ghostGA.grisb import *
import numpy as np
from numpy.linalg import norm
from triqs_ghostGA.solvers.ci import CI
from  triqs_ghostGA.solvers import svd_solver2
import os
import h5py

# from matplotlib import pyplot as plt

import sys
from utils import set_gGA, compute_Z
np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf)

from time import time

nimp, nbath, ntot = 2, 6, 8

nfix = None
tmp_nfix = 1 if nfix is None else nfix

Ks = range(3, 22, 3)
Us = np.arange(0.2, 3.41, 0.8)

dw_Z = 0.0001
itmax = 25

ci_filename = "ci_data.h5"
svd_filename = "svd_data.h5"

# fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=200)

ci_R = np.array([[-0.6, -0.3, -0.6]]).T
ci_R = np.kron(ci_R, np.eye(2))
ci_Lambda = np.diag([0.5, 0, -0.5])
ci_Lambda = np.kron(ci_Lambda, np.eye(2))

for u, U in enumerate(Us):
    print("###########################")
    print(" U = %.1f" % U)

    tmp = set_gGA(nimp, nbath, U, R0=ci_R, Lambda0=ci_Lambda)
    (eloc, eks, R0, Lambda0, ci_U, svd_U) = tmp
    R0 = ci_R
    Lambda0 = ci_Lambda

    # Check if calculation already done
    done = False
    name = 'B%d_U%.1f_nfix%.1f' % (nbath//2, U, tmp_nfix)
    with h5py.File(ci_filename, 'a') as ci_file:
        if name in ci_file:
            done = True
            ci_denMat = ci_file[name]['denMat'][:]
            ci_ene = ci_file[name]['ene'][()]
            ci_Z = ci_file[name]['Z'][()]
            ci_docc = ci_file[name]['docc'][()]
            ci_R = ci_file[name]["R"][:]
            ci_Lambda = ci_file[name]["Lambda"][:]

            print("  Calculation found:")
            print("    ci_denMat:")
            print(ci_denMat)
            print("    ci_ene: %.12f" % ci_ene)
            print("    ci_Z: %.4f" % ci_Z)
            print("    ci_docc: %.4f" % ci_docc)

    if not done:
        #####################
        ### Ghost with CI ###
        #####################

        t0 = time()

        edsolver=CI(ntot, use_Ntot=True, use_Sz=False, spin_pen=10)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, ci_U, R=R0,
                      Lambda=Lambda0, edsolver=edsolver)

        if nfix is None:
            grisb.run(mu0=0, itmax=itmax, mix=1, tol=3e-7,
                      beta=500, silence=True, mu_tol=1e-8)
        else:
            grisb.run(mu0=0, nfix=nfix, itmax=itmax, mix=1, tol=3e-7,
                      beta=500, silence=True, mu_tol=1e-8)

        ci_denMat = grisb.denMat
        ci_ene = grisb.edsolver.gs_ene
        ci_Z = compute_Z(grisb, eks, dw_Z)
        ci_docc = grisb.edsolver.calc_double_occ(0)
        ci_R = grisb.R
        ci_Lambda = grisb.Lambda

        with h5py.File(ci_filename, 'a') as ci_file:
            data = {"denMat": ci_denMat,
                    "ene": ci_ene,
                    "Z": ci_Z,
                    "docc": ci_docc,
                    "R": grisb.R,
                    "Lambda": grisb.Lambda}
            print(data)
            ci_file.create_group(name)
            for key, value in data.items():
                ci_file[name].create_dataset(key, data=value)

        print(" TIME: CI run %.3f" % (time() - t0))

    print(" CI U = %.1f, Z = %.4f, docc = %.4f" % (U, ci_Z, ci_docc))
    print()

    R0 = ci_R
    Lambda0 = ci_Lambda

    ######################
    ### Ghost with SVD ###
    ######################

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
                svd_R = svd_file[name]["R"][:]
                svd_Lambda = svd_file[name]["Lambda"][:]

                print("  Calculation found:")
                print("    svd_denMat:")
                print(svd_denMat)
                print("    svd_ene: %.12f" % svd_ene)
                print("    svd_Z: %.4f" % svd_Z)
                print("    svd_docc: %.4f" % svd_docc)

        if not done:

            t0 = time()

            sol=svd_solver2.SVDSolver2(ntot, nimp=nimp, nbath=nbath, K=K)
            sol.load_stuff(path="./B3_solver/")

            print("  TIME: SVD load_stuff %.3f" % (time() - t0))

            grisb = Grisb(ntot, nimp, nbath, eks, eloc, svd_U, R=R0,
                          Lambda=Lambda0, edsolver=sol)
            if nfix is None:
                grisb.run(mu0=0, itmax=itmax, mix=1, tol=1e-8, beta=500,
                          silence=False, mu_tol=1e-8)
            else:
                grisb.run(mu0=0, nfix=nfix, itmax=itmax, mix=1, tol=1e-8, beta=500,
                          silence=False, mu_tol=1e-8)

            svd_denMat = grisb.denMat
            svd_ene = grisb.edsolver.gs_ene
            svd_Z = compute_Z(grisb, eks, dw_Z)
            svd_docc = grisb.edsolver.calc_double_occ(0)

            with h5py.File(svd_filename, 'a') as svd_file:
                data = {"denMat": svd_denMat,
                        "ene": svd_ene,
                        "Z": svd_Z,
                        "docc": svd_docc,
                        "R": grisb.R,
                        "Lambda": grisb.Lambda}
                print(data)
                svd_file.create_group(name)
                for key, value in data.items():
                    svd_file[name].create_dataset(key, data=value)

            print("  TIME: SVD run %.3f" % (time() - t0))


        print("  SVD U = %.1f, K = %d, Z = %.4f, docc = %.4f" % (U, K, svd_Z, svd_docc))

        drho = norm(svd_denMat - ci_denMat)
        if K is None:
            print("  for K = None, drho = %.3e" % (drho))
        else:
            print("  for K = %d, drho = %.3e" % (K, drho))
        print()
