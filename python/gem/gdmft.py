import scipy
import h5py
import numpy as np
import numba
import sys
import time

from .fragment import Fragment
from .lattice import Lattice
from .utility.utilities import calc_nf, calc_Fermi

class Gdmft(object):
    """This is a class representation of a ghost-RISB object (with DMFT-like algorithm).
            
    :param ntot: Total number of orbital.
    :type ntot: int
        
    :param nimp: Total number of imp orbital.
    :type ntot: int
            
    :param nbath: Total number of bath orbital.
    :type ntot: int
            
    :param eks: Momentum distribution.
    :type eks: np.ndarray

    :param eloc: Local one-body Hamtilonian.
    :type eloc: np.ndarray

    :param Utensor: Local two-obdy interaction.
    :type Utensor: np.ndarray

    :param R: R matrix.
    :type R: np.ndarray

    :param Lambda: Lambda matrix.
    :type Lambda: np.ndarray

    :param ed_params: Exact diagonalization solver parameters.
    :type ed_params: dic

    :param Hspin_list: Hermitian matrix basis in each spin block with spin symmetry.
    :type Hspin_list: list

    :param Hfull_list: Full Hermitian matrix basis.
    :type Hfull_list: list

    """
    def __init__(self, ntot, nimp, nbath, eks, eloc, Utensor,
                 wks=None, spin_sym=True, orb_sym=False,
                 soc=False, R=None, Lambda=None, D=None, Lambda_c=None,
                 edsolver=None, suff='',
                 verbose=0,spin_pen=0):
        print("##### INITIALIZATON OF THE GRISB OBJECT (DMFT-like algorithm)#####")
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.eks = eks
        self.wks = wks
        self.eloc = eloc
        self.Utensor = Utensor
        self.soc = soc
        self.spin_sym = spin_sym
        self.orb_sym = orb_sym
        self.gs_wf = None
        self.suff = suff    # Suffixe for file writting when many cpu at same time
        self.verb = verbose
        self.spin_pen = spin_pen


        if edsolver is None:
            raise ValueError("Not edsolver was passed to the GRISB.")

        self.Fragment = Fragment(self.nimp, self.nbath,
                 self.eloc, self.Utensor, edsolver,
                 Lambda=Lambda,R=R,Lambda_c=Lambda_c,D=D,
                 verbose=self.verb
                  )
        self.Lambda=self.Fragment.Lambda
        self.R     =self.Fragment.R
        self.Lambda_c=self.Fragment.Lambda_c
        self.D       =self.Fragment.D

        self.Lattice = Lattice(self.eks, wk_list=self.wks, verbose=self.verb)




    def compute_energy(self,beta=200.,mu=0.0):
        """ Compute total energy, kinetic energy, and potential energy
        """
        #self.ekin = [np.sum(self.R.dot( self.eks[x] ).dot( self.R.conj().T )*self.rhok_list[x].T ) for x in range(len(self.rhok_list))]
        #self.ekin = sum(self.ekin)/float(len(self.rhok_list))
        self.ekin = sum([np.sum( ( np.dot(self.R, np.dot(x, self.R.conj().T )) ) * \
                    calc_nf( np.dot(self.R, np.dot(x, self.R.conj().T) ) + self.Lambda , 1./beta).T )*wk for x,wk in zip(self.Lattice.eks,self.Lattice.wks)] )
        self.epot = self.Fragment.E2loc + np.trace(self.Fragment.eloc.dot(self.Fragment.denMat[:self.nimp,:self.nimp].T))
        self.etot = self.ekin + self.epot - mu*self.Fragment.nfill

# THIS FOR TEMP
    def run(self, mu=0.0, itmax=200, mix=0.5, tol=1e-6, T=1e-3, n_target=None, n_tolerance=1e-3,
            silence=True, spin_pen=0.0, sz_pen=0.0, idx=0, num_eig=2, ed_verbose=0, n_fit_method='qp'):

        print("mu = ", mu)
        self.T = T
        self.diff = 1e20
        self.mu = mu
        for it in range(itmax):

            print(' Doing it:',it,'/',itmax)

            self.Lattice.solve_qp([self.Fragment], T=T)

            self.D, self.Lambda_c = self.Fragment.update_hybridization(T=T)


            
            if not silence:
                if not self.soc:
                    print("Delta_p=")
                    print(self.Fragment.Delta_qp[::2,::2])
                    print("D=")
                    print(self.Fragment.D[::2,::2])
                    print("Lambda_c=")
                    print(self.Fragment.Lambda_c[::2,::2])
                else:
                    print("Delta_p=")
                    print(self.Fragment.Delta_qp[:,:])
                    print("D=")
                    print(self.Fragment.D[:,:])
                    print("Lambda_c=")
                    print(self.Fragment.Lambda_c[:,:])

            # ED solvers
            self.Fragment.solve_impurity(self.mu, T=T, num_eig=1, spin_pen=self.spin_pen)

            #Update R and Update Lambda
            self.nfill = np.trace(self.Fragment.denMat[:self.nimp,:self.nimp])
            print('imp denmat:')
            print(self.Fragment.denMat[:self.nimp,:self.nimp])

            print(" --> n_filling:",self.nfill,' - target:',n_target)
            if( (not n_target is None) and (np.abs(self.nfill - n_target)>n_tolerance) ):
                print('Fitting')
                mu_new = self.Lattice.fit_mu( n_target, [self.Fragment], T=T, mode=n_fit_method, mu_old=self.mu, ntol=n_tolerance )
                if(not mu_new is None):
                    self.mu = mu_new
                    self.Fragment.solve_impurity(self.mu, T=T, num_eig=1, spin_pen=self.spin_pen)

            Lambda_old = self.Fragment.Lambda.copy()
            R_old = self.Fragment.R.copy()

            R_new, Lambda_new = self.Fragment.update_self_energy(T=T)

            L_eval_new, UL_new = np.linalg.eigh(Lambda_new[::2,::2])
            L_eval_old, UL_old = np.linalg.eigh(Lambda_old[::2,::2])

            diff_R = np.abs(np.abs(UL_old@R_old[::2,::2])-np.abs(UL_new@R_new[::2,::2]) ).max()

            diff_Lambda = np.abs(L_eval_new-L_eval_old).max()

            print('Rold:',UL_old@R_old[::2,::2])
            print('Rnew:',UL_new@R_new[::2,::2])
            print('lambda_evals:',L_eval_new)
            print('Rdag@R:', R_new.T.conj() @ R_new )
            print('diff_R:',diff_R)
            print('diff_L:',diff_Lambda)
            print('mu:',self.mu)
            #time.sleep(1)
            self.diff = max(diff_R,diff_Lambda)
            # MIXING
            Lambda_new = (1-mix)*Lambda_new.copy() + mix*Lambda_old
            R_new = (1-mix)*R_new.copy() + mix*R_old
            
            # Update mixed parameters
            self.Fragment.Lambda = Lambda_new
            self.Fragment.R = R_new
            self.Lambda = Lambda_new
            self.R = R_new

            if(self.spin_sym):
                self.Fragment.impose_spin_SU2_symmetry()
                self.Lambda = self.Fragment.Lambda
                self.R = self.Fragment.R
            if(self.orb_sym):
                self.Fragment.impose_orbital_symmetry()
                self.Lambda = self.Fragment.Lambda
                self.R = self.Fragment.R

            convg_n=True
            if not silence:
                #print("R_new=")
                #print(R_new)
                print("R=")
                print(self.R[::2,::2])
                print("Lambda_new=")
                print(Lambda_new[::2,::2])
                print("Lambda=")
                print(self.Lambda[::2,::2])
                print("Delta_p")
                print(self.Fragment.Delta_aim[::2,::2])
                print("density matrix=")
                print(self.Fragment.denMat[::2,::2])
            print("iteration:",it,'diff=',self.diff)

            if (self.diff < tol and it>2) or it == (itmax-1):
                print("--------------------- ghost-RISB converged with diff=%g ---------------------"%(self.diff))
                print("density matrix=")
                print(self.Fragment.denMat)
                self.nfill = np.trace(self.Fragment.denMat[:self.nimp,:self.nimp])
                self.docc = []
                for idx in range(0,self.nimp,2):
                    self.docc.append(self.Fragment.solver.calc_double_occ(idx))
                print("double occupancy=", self.docc)
                print("convg_n",convg_n)
                print('lambda eigvals',np.linalg.eigvalsh(self.Lambda))
                break

    def get_functional(self):
        """ Compute the value of the finite temperature functional
        """
        Omega_tot = self.Lattice.compute_functional( [self.Fragment], T=self.T )
        return Omega_tot
