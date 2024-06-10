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
# Authors: Olivier Gingras and Tsung-Han Lee

import numpy as np
import triqs.utility.mpi as mpi
import forktps as ftps
from forktps.solver_core import Bath, HInt
from itertools import product as itp

def ConstructBath(gfstruc_ , Nbath_, SpinOrbCoup_, hopping_, eps_):
    """
    Construct the bath for the impurity solver.
        gfstruc_ : the structure of the impurity model
        Nbath_ : the number of bath sites for each band or impurity degree of freedom
        SpinOrbCoup_ : the spin-orbital coupling
        hopping_ : np.zeros([size, size, Nbath_], dtype=complex), hopping_[ i_imp, j_bath, ib] : the hopping amptitute from (j_bath orbital of the ib bath) to (i_imp orbital of the impurity)
        eps_ : np.zeros([size, Nbath_])
    """
    bath = Bath(gfstruc_, SpinOrbCoup_)
    for name,size in gfstruc_:
        for iorb in range(size):
            for ib in np.arange(Nbath_):
                indx = [name, iorb]
                eps = eps_[name][iorb, Nbath_ - 1 - ib]
                hop = hopping_[name][:, iorb, Nbath_ - 1 - ib]
                bath.addSite(indx, eps, hop) # note that the first added bath site is placed at the end of the fork
    return bath

def setup_forkTPS(M, Norb, Nbath, gf_struct, int_params, w_grid, maxm, tw,
                  other_params={"dt": 0.1, "time_steps": 1, "sweeps": 10, "prep_napph": 5, "DMRGMethod": "TwoSite"}):
    """
    Given the embedded Hamiltonian and parameters, run ForkTPS and return density matrix.
        M               : Embedded Hamiltonian in a matrix of size (Norb+Nbath)x(Norb+Nbath), containing local hamiltonian,
                          hybridization with the bath, and bath degrees of freedom. The bath should be diagonal.
        Norb            : Number of physical orbitals.
        Nbath           : Number of bath degrees of freedom.
        gf_struct       : Structure of the impurity model.
        int_params      : Parameters of the interacting Hamiltonian.
        w_grid          : Bandwidth of the w-grid.
        maxm            : Maximal bound dimension.
        tw              : Cutoff parameter for the bound dimension.
        other_params    : Other parameters for ForkTPS.
    """

    # Construct the real time ForkTPS solver.
    S = ftps.Solver(gf_struct = gf_struct , nw = w_grid["nw"],
                    wmin=w_grid["window"][0], wmax=w_grid["window"][1])

    # Fix the interacting Hamiltonian
    Hint = HInt(u=int_params["U"], j=int_params["J"],
                up=int_params["Up"], dd=int_params["dd"])

    # Construct the local Hamiltonian and extract from M matrix
    # give the local Hamiltonian the right block structure
    e0 = ftps.solver_core.Hloc(gf_struct)
    e0.Fill("up", M["up"][:Norb, :Norb])
    e0.Fill("dn", M["dn"][:Norb, :Norb])

    # Construct the bath:
    # the bath sites are assigned to an orbital.
    # Since it should be diagonal,
    # there is only one bath orbital, which is (orb, bath).
    eps = {"up": np.zeros((Norb, Nbath)),
           "dn": np.zeros((Norb, Nbath))}
    # Construct the hybridization:
    # now this couples a bath site and an orbital, so (orb, bath)->(orb).
    hopping = {"up": np.zeros((Norb, Norb, Nbath), dtype=complex),
               "dn": np.zeros((Norb, Norb, Nbath), dtype=complex)}
    # Mapping M to the correct shape for ForkTPS:
    for a in range(Norb):
        pos0 = Norb+a*Nbath
        pos1 = Norb+(a+1)*Nbath
        eps["up"][a, :] = np.diag(M["up"][pos0:pos1, pos0:pos1])
        eps["dn"][a, :] = np.diag(M["dn"][pos0:pos1, pos0:pos1])
        hopping["up"][:, a, :] = M["up"][:Norb, pos0:pos1]
        hopping["dn"][:, a, :] = M["dn"][:Norb, pos0:pos1]

    # Assigning in the solver object
    S.b = ConstructBath(gf_struct, Nbath, False, hopping, eps)
    S.e0 = e0

    # Setting up the time-evolution solver
    tevo = ftps.solver.TevoParams(dt = other_params["dt"],
                                  time_steps = other_params["time_steps"])
    # Setting up the DMRG parameters
    dmrg = ftps.solver.DMRGParams(sweeps = other_params["sweeps"],
                                  prep_napph = other_params["prep_napph"],
                                  maxm=maxm, tw=tw,
                                  DMRGMethod=other_params["DMRGMethod"])

    # Solve the impurity model
    S.solve(h_int = Hint,
            tevo = tevo,
            params_partSector = dmrg,
            params_GS = dmrg
           )

    # Extract the single-particle density matrix and reshape it
    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    singleP = S.singleParticleDensity
    print('singleP=')
    print(singleP)
    rho_CDC = singleP[:(Norb*2*(Nbath+1))**2]
    rho_CDC = np.reshape(rho_CDC, (Norb*2*(Nbath+1), Norb*2*(Nbath+1)))
    print(rho_CDC)

    # Return density matrix and interaction energy
    return rho_CDC, S.Ehint

def rotateBath(M, Norb, Nbath):
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
    M_rot = {"up": np.copy(M["up"]),
             "dn": np.copy(M["dn"])}

    # Obtained the eigenvectors of the bath sites to rotate the matrix
    v_all = {"up": [], "dn": []}
    for name in ["up", "dn"]:
        B = M[name][Norb:, Norb:] # Bath sites
        # W = M[name][:Norb, Norb:]
        # Wd = M[name][Norb:, :Norb]

        # Diagonalization of the bath
        w, v = np.linalg.eig(B)
        # Keep the eigenvectors in memory
        v_all[name] = np.block([[np.eye(Norb), np.zeros((Norb, Nbath*Norb))],
                                [np.zeros((Norb*Nbath, Norb)), v]])

        # Rotate the embedded Hamiltonian
        M_rot[name] = np.linalg.inv(v_all[name]) @ M_rot[name] @ v_all[name]
    return M_rot, v_all

def rotateDensityMatrix(singleP, Norb, Nbath, v):
    """
    Rotate back the density matrix. Used with rotateBath to minimize the entropy
    in forkTPS.
        singleP : Density matrix obtained by forkTPS.
        Norb : int : Number of orbital degrees of freedom.
        Nbath : int : Number of baths per orbital.
        v : Eigenvectors of the Bath obtained from rotateBath.
    """
    # ForkTPS writes the density matrix in a basis that mixes spin up and down.
    # These list help convert to separate up and down.
    # list_up = list(range(0, 2*Norb, 2))
    # list_dn = list(range(1, 2*Norb, 2))
    # for a in range(Norb):
    #     list_up += list(range(2*Norb+2*a*Nbath, 2*Norb+(2*a+1)*Nbath))
    #     list_dn += list(range(2*Norb+(2*a+1)*Nbath, 2*Norb+(2*a+2)*Nbath))
    list_up = range(0, Norb*(1+Nbath))
    list_dn = range(Norb*(1+Nbath), 2*Norb*(1+Nbath))

    # Extract density matrix for up and rotate back to the original basis,
    # before the bath was diagonalized.
    single_up = singleP[list_up, :][:, list_up]
    v_up = v["up"]
    single_up = v_up @ single_up @ np.linalg.inv(v_up)

    # Same for down
    single_dn = singleP[list_dn, :][:, list_dn]
    v_dn = v["dn"]
    single_dn = v_dn @ single_dn @ np.linalg.inv(v_dn)

    # Replace in the density matrix
    for a, A in enumerate(list_up):
        for b, B in enumerate(list_up):
            singleP[A, B] = single_up[a, b]
    for a, A in enumerate(list_dn):
        for b, B in enumerate(list_dn):
            singleP[A, B] = single_dn[a, b]
    return singleP

def rotateToTsungHanConvention(rho_CDC, Norb, Nbath):

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

