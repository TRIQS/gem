# Solver based on ITensorMPS, a finite MPS and MPO methods
# based on the Julia version of ITensor ( https://github.com/ITensor/ITensorWebsite.git )
# 
#
# When using this solver please cite the following articles:
# - M. Fishman et al, SciPost Phys. Codebases 4 (2022)

# To be changed once we have the interface with the triqs-itensor solver.

import numpy as np
import os
import sys
import h5py
# from h5 import *
###julia setup
try:
    import juliacall
    from juliacall import Main as jl
    from juliacall import Pkg

    julia_project_dir = os.environ["PYTHON_JULIAPKG_PROJECT"]
    print(julia_project_dir)
    Pkg.activate(julia_project_dir)
    Pkg.instantiate()
    jl.seval("using GGMPSSolver")
    include_str = "include(\"" + julia_project_dir + "/src/driver.jl" + "\")"
except (ImportError, KeyError):
    pass

from itertools import product as itp
import gem
from gem.solvers.utility.utils_mps import setup_MPS, rotateBath, rotateDensityMatrix, rotateToTsungHanConvention

#SAMUELE's COMMENT
# - implement docc
# - prevent thermal calculation with some error
# - is EHint the same as E2loc? then do not use different names

class ITensorMPSSolver(object):
    ''' FTPS solver class'''
    def __init__(self, ntot, nimp, nbath, params={"use_Sz":True,"use_Ntot":True,"spin_pen":0.0}, suff="", rotateBath=True, recouple=True, solver_params=None):
        """Constructor method
        """
        self.type = "ITensorMPSSolver"
        self.solver_params = solver_params if solver_params is not None else {}
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.set_kwargs(self.solver_params.get('kwargs', params))
        self.schedule = []
        self.make_schedule()    #initialize with default
        self.tolerances = []
        self.set_tolerances()
        self.scalartype = np.float_ # if not set elsewhere
        self.scalartype = np.complex_ # if not set elsewhere
        self.paramagnetic = True
        self.suff = suff
        self.rotateBath = rotateBath
        self.recouple = recouple
        self.gs_ene = 0

#MANDATORY FUNCTIONS
    def build_Hemb(self, D, eloc, LAMBDA, V2E): # , spin_pen=0.0):
        # Local Hamiltonian
        #thedtype=np.complex_
        thedtype=self.scalartype
        self.E = {"up": np.zeros((self.nimp//2, self.nimp//2),dtype=thedtype),
                  "dn": np.zeros((self.nimp//2, self.nimp//2),dtype=thedtype)}
        self.E["up"] = eloc[::2,::2]
        self.E["dn"] = eloc[1::2,1::2]

        # Hybridization matrix
        self.W = {"up": np.zeros((self.nimp//2, self.nbath//2),dtype=thedtype),
                  "dn": np.zeros((self.nimp//2, self.nbath//2),dtype=thedtype)}
        self.W["up"][:,:] = D[::2,::2].conj().T
        self.W["dn"][:,:] = D[1::2,1::2].conj().T

        # Bath parameters
        self.B = {"up": np.zeros((self.nbath//2, self.nbath//2),dtype=thedtype),
                  "dn": np.zeros((self.nbath//2, self.nbath//2),dtype=thedtype)}
        self.B["up"][:,:] = -LAMBDA[::2,::2]
        self.B["dn"][:,:] = -LAMBDA[1::2,1::2]

        # Set up the M matrix which has all local Ham, hybridization and bath
        self.M = {"up": np.block([[self.E["up"], self.W["up"]],
                                 [self.W["up"].T.conjugate(), self.B["up"]]]),
                 "dn": np.block([[self.E["dn"], self.W["dn"]],
                                 [self.W["dn"].T.conjugate(), self.B["dn"]]])}

        self.Utensor = V2E

        if self.rotateBath:
            # Rotate the Bath and Hybridization for smaller entropy
            ## We can either assume that this just works out of the box, or assume the bath is diagonal?
            self.M, self.v = rotateBath(self.M, self.nimp//2, self.nbath//self.nimp,paramagnetic=self.paramagnetic, recouple=self.recouple)

        self.M["up"]=0.5*(self.M["up"] + self.M["up"].T.conjugate())
        self.M["dn"]=0.5*(self.M["dn"] + self.M["dn"].T.conjugate())

    def solve_Hemb(self, num_eig=1, verbose=1,tol=1e-8, T=0.0 ):
        # Criteria for the bound dimension of the DMRG, just be converged
        # Set up and run ForkTPS using the useful_func.py
        outfile = "data%s.h5" % self.suff
        self.converged = False
        beta=1/T

        print(self.M)
        ### Run MPS with julia call ###
        self.converged, self.gs, self.EHint, self.singleP_up, self.singleP_dn, self.gs_ene = jl.solve(self.Utensor, self.M, self.schedule,self.tolerances, self.kwargs,outfile=outfile)

        self.singleP_up = np.asarray(self.singleP_up)
        self.singleP_dn = np.asarray(self.singleP_dn)

        print("single particle density matrix up: ")
        print(self.singleP_up)
        print("single particle density matrix dn: ")
        print(self.singleP_dn)
        print("EHint:", self.EHint)

        if self.paramagnetic:
            self.singleP_up = 0.5*(self.singleP_up + self.singleP_dn)   #constrains to paramagnet
            self.singleP_dn = self.singleP_up.copy() #constrains to paramagnet

    def calc_density_matrix(self):
        if self.rotateBath:
            self.singleP = rotateDensityMatrix(self.singleP_up,self.singleP_dn, self.v)
        else:
            zeros = np.zeros(self.singleP_up.shape)
            self.singleP = np.block([[self.singleP_up, zeros],
                                     [zeros, self.singleP_dn]])

        ##Assumes this one is the same now
        self.singleP = rotateToTsungHanConvention(self.singleP, self.nimp//2, self.nbath//self.nimp)
        if self.scalartype==np.float_:
            self.dm = self.singleP.real
        else:
            self.dm = self.singleP
        return self.dm

    def compute_E2loc(self):
        #eone = 2*numpy.einsum('ij,ij',self.h1,self.dm[::2,::2])
        #etwo = self.e0 - eone
        return self.EHint

#AUXILIARY FUNCTIONS
    def add_to_schedule(self,nsweeps=1,maxdim=1024, cutoff=1e-14,noise=0.0,outputlevel=1):
        thesweep=    {
            "nsweeps":nsweeps,
            "maxdim":maxdim,
            "cutoff":cutoff,
            "noise":noise,
            "outputlevel":outputlevel
            }
        self.schedule.append(
        [tuple(thesweep.keys()),tuple(thesweep.values())])
        return thesweep

    def make_schedule(self, input=False):
        assert input==False
        if type(input)==bool and input==False:
            self.add_to_schedule(nsweeps=15,maxdim=32,cutoff=1e-10,noise=1e-5)
            self.add_to_schedule(nsweeps=15,maxdim=64,cutoff=1e-10,noise=1e-6)
            self.add_to_schedule(nsweeps=15,maxdim=128,cutoff=1e-12,noise=1e-6)
            self.add_to_schedule(nsweeps=10,maxdim=256,cutoff=1e-12,noise=1e-6)
            self.add_to_schedule(nsweeps=10,maxdim=512,cutoff=1e-12,noise=1e-7)
            self.add_to_schedule(nsweeps=10,maxdim=1024,cutoff=1e-14,noise=1e-8)
            self.add_to_schedule(nsweeps=10,maxdim=2048,cutoff=1e-14,noise=1e-9)
            self.add_to_schedule(nsweeps=5,maxdim=4096,cutoff=1e-14,noise=1e-10)
            self.add_to_schedule(nsweeps=3,maxdim=8192,cutoff=1e-14,noise=1e-12)
            self.add_to_schedule(nsweeps=2,maxdim=8192,cutoff=1e-14,noise=0.0)
            #setup standard schedule
        else:
            #not implemented yet
            assert False
        return

    def set_kwargs(self,kwargs={"use_Sz":True,"use_Ntot":True,"spin_pen":0.0}):
        self.kwargs=[[tuple(kwargs.keys()),tuple(kwargs.values())]]
        return

    def modify_kwargs(self, key,value):
        #make dict out of key, value pairs
        d=dict(zip(self.kwargs[0][0],self.kwargs[0][1]))
        d[key]=value
        self.set_kwargs(d)
        return

    def set_tolerances(self,tol_names=("E","rho"),tol_vals=(1e-5,5e-3)):
        self.tolerances=[[tol_names,tol_vals]]
        return

    def h5write_gs(self,filename,group_path,name):
        #check that group at group_path exists
        with h5py.File(filename,"a") as f:
            if group_path not in f:
                f.create_group(group_path)
        jl.GGMPSSolver.write_mps_to_file(filename,group_path,name,self.gs)
        return

    def h5read_state(self,filename,group_path,name):
        #check that group at group_path exists
        return jl.GGMPSSolver.read_mps_from_file(filename,group_path,name)

    def inner(self,bra,ket,operator=None):
        """
        operator is expected to have ket indices
        """
        bra=jl.GGMPSSolver.ITensors.replace_siteinds(bra,jl.GGMPSSolver.ITensors.siteinds(ket))
        if operator is not None:
            return jl.GGMPSSolver.ITensors.inner(jl.GGMPSSolver.ITensors.prime(bra),operator,ket)
        else:
            return jl.GGMPSSolver.ITensors.inner(bra,ket)


def write_mps_to_file(state,filename,group_path,name):
    #check that group at group_path exists
    with h5py.File(filename,"a") as f:
        if group_path not in f:
            f.create_group(group_path)
    jl.GGMPSSolver.write_mps_to_file(filename,group_path,name,state)
    return

def read_mps_from_file(filename,group_path,name):
    return jl.GGMPSSolver.read_mps_from_file(filename,group_path,name)
