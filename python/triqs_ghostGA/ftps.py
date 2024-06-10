import numpy as np
import scipy.linalg as lg
from scipy.optimize import minimize

from triqs.gf import *
import forktps as ftps
from forktps.solver import DMRGParams, TevoParams

from forktps.DiscreteBath import *
from forktps.Helpers import getX,MakeGFstruct

# from h5 import *

from itertools import product as itp
import triqs_ghostGA
from triqs_ghostGA.utility.utils_forktps import ConstructBath, setup_forkTPS, rotateBath, rotateDensityMatrix, rotateToTsungHanConvention

class FTPS(object):
    ''' FTPS solver class'''
    def __init__(self, ntot, nimp, nbath, maxM=200):
        """Constructor method
        """
        self.type = "FTPS"
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.maxM = maxM

    def build_Hemb(self, D, H1E, LAMBDA, V2E, spin_pen=0.0):
        # Local Hamiltonian
        self.E = {"up": np.zeros((self.nimp//2, self.nimp//2)),
                  "dn": np.zeros((self.nimp//2, self.nimp//2))}
        self.E["up"] = H1E[::2,::2]
        self.E["dn"] = H1E[1::2,1::2]

        # Hybridization matrix
        self.W = {"up": np.zeros((self.nimp//2, self.nbath//2)),    ##was self.nbath//self.nimp ##but this is not general
                  "dn": np.zeros((self.nimp//2, self.nbath//2))}
        self.W["up"][:,:] = D[::2,::2].conj().T
        self.W["dn"][:,:] = D[1::2,1::2].conj().T

        # Bath parameters
        self.B = {"up": np.zeros((self.nbath//self.nimp, self.nbath//self.nimp)),
                  "dn": np.zeros((self.nbath//self.nimp, self.nbath//self.nimp))}
        self.B["up"][:,:] = -LAMBDA[::2,::2]
        self.B["dn"][:,:] = -LAMBDA[1::2,1::2]

        # Set up the M matrix which has all local Ham, hybridization and bath
        self.M = {"up": np.block([[self.E["up"], self.W["up"]],
                                 [self.W["up"].T, self.B["up"]]]),
                 "dn": np.block([[self.E["dn"], self.W["dn"]],
                                 [self.W["dn"].T, self.B["dn"]]])}
        np.set_printoptions(precision=5, threshold=np.inf, linewidth=np.inf)
        #print('M["up"] before rotating the bath:')
        #print(self.M["up"])
        #print()
        #LAMBDA = B["up"]

        self.U = V2E[0,0,1,1]

        # Rotate the Bath and Hybridization for smaller entropy
        self.M, self.v = rotateBath(self.M, self.nimp//2, self.nbath//self.nimp)
        #print('v=')
        #print(self.v)

        #print("M['up'] after rotating to the basis in which the bath is diagonal:")
        #print(self.M['up'])
        #print()

    def solve_Hemb(self, num_eig=10, verbose=0):
        # Setting up some parameters for ForkTPS
        gfstruct = [("up", self.nimp//2), ("dn", self.nimp//2)] # Structure of the Green's function
        # Interaction parameters for Kanamori
        int_params = {"U": self.U, "J": 0, "Up": 0, "dd": True}

        nw = 501 # Number of real frequencies
        window = [-3., 3.] # Bandwidth of the spectral function
        w_grid = {"nw": 3001, "window": [-3., 3.]} # Resulting grid

        #maxM = 300 # Maximum dimension bond for DMRG

        # Criteria for the bound dimension of the DMRG, just be converged
        tw = 1e-20

        # Set up and run ForkTPS using the useful_func.py
        self.singleP_rot, self.EHint = setup_forkTPS(self.M, self.nimp//2, self.nbath//self.nimp, gfstruct, int_params,
                                                   w_grid, self.maxM, tw)
        #print('self.singleP_rot=')
        #print(self.singleP_rot)
        #print('self.v=')
        #print(self.v)

    def calc_density_matrix(self):
        self.singleP = rotateDensityMatrix(self.singleP_rot, self.nimp//2, self.nbath//self.nimp, self.v)
        #print(self.singleP)
        self.singleP = rotateToTsungHanConvention(self.singleP, self.nimp//2, self.nbath//self.nimp)

        #print('density matrix=')
        #print(self.singleP)
        self.dm = self.singleP
        return self.dm

    def calc_double_occ(self,idx):
        print('warning: double occupancy not implement!')
        return 0.25

    def compute_E2loc(self):
        #eone = 2*numpy.einsum('ij,ij',self.h1,self.dm[::2,::2])
        #etwo = self.e0 - eone
        return self.EHint

