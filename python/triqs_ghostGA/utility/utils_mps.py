# Copyright (c) 2022 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: [Benedikt Kloss] Olivier Gingras and Tsung-Han Lee

import numpy as np
from numpy.linalg import inv
from itertools import product as itp

#DEPRECATE --- not used in current codepath
def setup_MPS(M, Utensor, Norb, Nbath, schedule,tolerances,use_Sz=True,use_Ntot=True,spin_pen=0.0):
    """
    Given the embedded Hamiltonian and parameters, run MPS and return density matrix.
        M               : Embedded Hamiltonian in a matrix of size (Norb+Nbath)x(Norb+Nbath), containing local hamiltonian,
                          hybridization with the bath, and bath degrees of freedom. The bath should be diagonal.
        Norb            : Number of physical orbitals.
        Nbath           : Number of bath degrees of freedom.
        UTensor          : quartic interaction tensor (Triqs convention).
    """


    #assumes that import has happened before (when initializing MPS solver class)
    kwarg_names=["use_Sz","use_Ntot","spin_pen"]
    kwarg_vals=(use_Sz,use_Ntot,spin_pen)
    converged, Eint, Cuu,Cdd=jl.solve(Utensor,M,schedule,tolerances,[kwarg_names,kwarg_vals]
           )
    Cuu = 0.5*(Cuu + Cdd)
    Cdd = Cuu.copy()

    return  Cuu,Cdd, Eint


def rotateBath(M, Norb, Nbath, paramagnetic=True, recouple=True):
    """
    Diagonalizes the Bath part of the M matrix. Also rotates the hybridization.
    This function returns the rotated M matrix, along with the vectors to
    rotate it back.
        M : np.array((Norb*(Nbath+1), Norb*(Nbath+1))) : M matrix describing the
            embedded Hamiltonian.
        Norb : int : Number of orbital degrees of freedom.
        Nbath : int : Number of bath per orbital.
    """
    # Preparing the rotated Embedded Hamiltonian
    if paramagnetic:
        Mav=np.copy(0.5*(M["up"]+M["dn"]))
        M_rot = {"up": np.copy(Mav),
                 "dn": np.copy(Mav)}
        print("Mav")
        print(Mav)
        names = ["up"]
    else:
        M_rot = {"up": np.copy(M["up"]),
                 "dn": np.copy(M["dn"])}
        names = ["up", "dn"]
        raise NotImplementedError("Not sure this is right if not paramagnetic. Due to the random matrix in the folowing decoupled procedure.")

    # Obtained the eigenvectors of the bath sites to rotate the matrix
    v_all = {"up": [], "dn": []}

    for name in names:
        #B = M[name][Norb:, Norb:] # Bath sites
        B = M_rot[name][Norb:, Norb:] # Bath sites

        # Diagonalization of the bath
        assert np.allclose(0.5*(B+B.T.conjugate()),B)
        w, v = np.linalg.eigh(0.5*(B+B.T.conjugate()))

        W = M_rot[name][:Norb, Norb:]
        W_rot = W @ v
        ind = np.argsort(np.amax(np.abs(W_rot[:Norb, :]), axis=0))[::-1]
        W_rot[:, :] = W_rot[:, ind]
        w[:] = w[ind]
        v[:, :] = v[:, ind]

        v_all[name] = np.block([[np.eye(Norb), np.zeros((Norb, Nbath*Norb))],
                                [np.zeros((Norb*Nbath, Norb)), v]])
        if recouple:
            decoupled = 1
            for i in np.arange(Nbath*Norb-1, Norb-1, -1):
                if np.amax(np.abs(W_rot[:Norb, i]), axis=0) < 1e-3:
                     decoupled += 1

            if decoupled > 1:
                print("Recoupling procedure with %s states" % (decoupled-1))
                a = np.random.rand(decoupled, decoupled)
                q, r = np.linalg.qr(a)

                recouple = np.block([[np.eye(Nbath*Norb-decoupled), np.zeros((Nbath*Norb-decoupled, decoupled))],
                                     [np.zeros((decoupled, Nbath*Norb-decoupled)), q]])

                v = v @ recouple

        # Keep the eigenvectors in memory
        v_all[name] = np.block([[np.eye(Norb), np.zeros((Norb, Nbath*Norb))],
                                [np.zeros((Norb*Nbath, Norb)), v]])
        # v_all[name] = np.eye(Norb*(1+Nbath))

        # Rotate the embedded Hamiltonian
        # M_rot[name] = v_all[name].T.conjugate() @ M_rot[name] @ v_all[name]
        M_rot[name] = v_all[name].T.conjugate() @ M_rot[name] @ v_all[name]

        print(v_all[name] @ M_rot[name] @ v_all[name].T.conjugate())
        # try:
        #     assert np.allclose(M_rot[name][Norb:,Norb:],np.diag(w))
        # except:
        #     print(M_rot[name])
        #     print(np.diag(w))
        #     raise

    if paramagnetic:
        M_rot={"up": np.copy(M_rot["up"]), "dn": np.copy(M_rot["up"])}
        v_all={"up": np.copy(v_all["up"]), "dn": np.copy(v_all["up"])}

    np.set_printoptions(precision=8, linewidth=np.inf, threshold=np.inf)
    print("M_rot up")
    print(M_rot["up"])
    return M_rot, v_all

def rotateDensityMatrix(singlePup,singlePdn, v):
    """
    Rotate back the density matrix. Used with rotateBath to minimize the entropy
    in MPS
        singleP : Density matrix obtained by forkTPS.
        Norb : int : Number of orbital degrees of freedom.
        Nbath : int : Number of baths per orbital.
        v : Eigenvectors of the Bath obtained from rotateBath.
    """
    # Extract density matrix for up and rotate back to the original basis,
    # before the bath was diagonalized.
    single_up = singlePup
    single_dn = singlePdn

    v_up = v["up"]
    v_dn = v["dn"]

    single_up = v_up @ single_up @ v_up.T.conjugate()  ##inv is the wrong thing to do here! it's a unitary rotation after all
    # single_up = v_up @ single_up @ v_up.T.conjugate()  ##inv is the wrong thing to do here! it's a unitary rotation after all
    single_dn = v_dn @ single_dn @ v_dn.T.conjugate()

    return np.block([[single_up,np.zeros(single_up.shape)],[np.zeros(single_up.shape),single_dn]])

def rotateToTsungHanConvention(rho_CDC, Norb, Nbath):
    #assert False
    ##FIXME: not checked/implemented yet
    A = list(range(2*Norb*(Nbath+1)))
    B = []
    for a in range(Norb*(Nbath+1)):
        B.append(a)
        B.append(a+Norb*(Nbath+1))
    # B = list(np.arange(0, 2*Norb*(1+Nbath), (1+Nbath)))
    # B = list(np.arange(0, 2*Norb*(1+Nbath), 2))
    # B += list(np.arange(1, 2*Norb*(1+Nbath), 2))
    # for a, b in itp(range(2*Norb), range(Nbath)):
    #     B.append(1+b+a*(1+Nbath))

    rho_CDC[A, :] = rho_CDC[B, :]
    rho_CDC[:, A] = rho_CDC[:, B]
    return rho_CDC

