###############################################
# ghost-RISB/GA algorithm
# Author: Tsung-Han Lee
# Email:  henhans74716@gmail.com
###############################################
import scipy
from scipy.linalg import sqrtm
import h5py
import numpy as np
import numba
from triqs_ghostGA.utility.utils_TH import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, \
     Hermitian_list, get_blocks, funcMat, calc_nf, dF
from triqs_ghostGA.DIIS import *
from triqs_ghostGA.utility.utils_grisb import *
from h5 import *
import sys


class Grisb(object):
    """This is a class representation of a ghost-RISB object.

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

    # :param ed_params: Exact diagonalization solver parameters.
    # :type ed_params: dic
    :param edsolver: Exact diagonalization solver. Should initialize prior.
    :type edsolver: CI, FTPS or ITensorMPSSolver

    :param Hspin_list: Hermitian matrix basis in each spin block with spin symmetry.
    :type Hspin_list: list

    :param Hfull_list: Full Hermitian matrix basis.
    :type Hfull_list: list

    """
    def __init__(self, ntot, nimp, nbath, eks, eloc, Utensor, spin_sym=True, soc=False, R=None, Lambda=None, edsolver=None, suff=''):
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.eks = eks
        self.eloc = eloc
        self.Utensor = Utensor
        self.soc = soc
        self.spin_sym = spin_sym
        self.gs_wf = None
        self.suff = suff    # Suffixe for file writting when many cpu at same time
        # initialize R and Lambda
        if R is None:
            self.R = np.kron(np.ones((nbath//2,nimp//2)), np.eye(2))*0.5
        else:
            if R.shape != (nbath,nimp):
                raise ValueError("R has inconsistent shape. Should be (nbath,nimp)")
            self.R = R
        if Lambda is None:
            #self.Lambda = np.kron(np.random.rand(nbath//2,nbath//2)+1j*np.random.rand(nbath//2,nbath//2), np.eye(2))
            self.Lambda = np.zeros((nbath,nbath))
            self.Lambda[0,0] = 1.0
            self.Lambda[1,1] = 1.0
            self.Lambda[2,2] =-1.0
            self.Lambda[3,3] =-1.0
        else:
            if Lambda.shape != (nbath,nbath):
                raise ValueError("Lambda has inconsistent shape. Should be (nbath,nbath)")
            self.Lambda = Lambda

        if edsolver is None:
            raise ValueError("Not edsolver was passed to the GRISB.")
        else:
            self.edsolver = edsolver

        # initialize single-particle basis
        self.Hspin_list,self.tHspin_list=Hermitian_list(nbath//2)
        self.Hfull_list,self.tHfull_list=Hermitian_list(nbath)
        print('initial R matrix =')
        print(self.R)
        print('initial Lambda matrix =')
        print(self.Lambda)

    def build_h1e(self,mu):
        # TODO: Move to the CI solver?
        h1e = np.zeros((self.ntot,self.ntot), dtype=np.complex128)
        h1e[:self.nimp,:self.nimp] = self.eloc - mu*np.eye(self.nimp)
        h1e[:self.nimp,self.nimp:] = self.D.T
        h1e[self.nimp:,self.nimp:] = -self.Lambda_c
        h1e[self.nimp:,:self.nimp] = self.D.conj()
        return h1e

    def solve_embedding(self, mu, num_eig, ed_verbose, spin_pen, sz_pen=0.0):
        """ Solve embedding problem using a variety of impurity solver
        """
        fh5 = h5py.File('hemb_test%s.h5' % self.suff,'w')
        fh5['eloc'] = self.eloc
        fh5['D'] = self.D
        fh5['Lambda_c'] = self.Lambda_c
        fh5['Utensor'] = self.Utensor
        fh5['mu'] = mu
        fh5.close()
        #print('Lambda_c=')
        #print(self.Lambda_c)
        #print('mu=',mu)
        #print('eloc=')
        #print(self.eloc)
        if self.edsolver.type == "CI":
            h1e = self.build_h1e(mu)
            #print('h1e=')
            #print(h1e)
            #print('spin_pen=',spin_pen)
            self.edsolver.build_Hemb(h1e, self.Utensor, spin_pen=spin_pen, sz_pen=sz_pen)
        elif self.edsolver.type == "FTPS":
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)
        elif self.edsolver.type == "ITensorMPSSolver":
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)
            #self.edsolver.schedule=[]
            #self.edsolver.make_schedule()
            #self.edsolver.set_tolerances(("E","rho"),(1e-5,5e-3))
        elif self.edsolver.type == "PySCFCCSD":
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)
        elif self.edsolver.type == "Block2NSZ":
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)
        else:
            raise ValueError("only Full ED, CI, and HCI are supported")
            # TODO: Replace whole if-clause by edsolver.prolog(self) implemented by
            # Solver(AbstractSolver)
        self.edsolver.solve_Hemb(num_eig=num_eig, verbose=ed_verbose )
        self.denMat = self.edsolver.calc_density_matrix()
        self.E2loc = self.edsolver.compute_E2loc()

    def compute_energy(self,beta=200.,mu=0.0):
        """ Compute total energy, kinetic energy, and potential energy
        """

        ###FIXME: Understand how the partitioning here is meant?
        #self.ekin = [np.sum(self.R.dot( self.eks[x] ).dot( self.R.conj().T )*self.rhok_list[x].T ) for x in range(len(self.rhok_list))]
        #self.ekin = sum(self.ekin)/float(len(self.rhok_list))
        self.ekin = sum([np.sum( ( np.dot(self.R, np.dot(x, self.R.conj().T )) ) * \
                    calc_nf( np.dot(self.R, np.dot(x, self.R.conj().T) ) + self.Lambda , 1./beta).T ) for x in self.eks] )/float(len(self.eks))
        self.epot = self.E2loc + np.trace(self.eloc.dot(self.denMat[:self.nimp,:self.nimp].T))
        self.etot = self.ekin + self.epot - mu*self.nfill

    def save_data(self, mu, filepath=None, save_state=True):
        if filepath is None:
            filepath = "saved_data%s.h5" % self.suff
        from datetime import datetime
        with HDFArchive(filepath, 'a') as A:
            tmp_dict = {
                "eloc": self.eloc,
                "D": self.D,
                "Lambda_c": self.Lambda_c,
                "Utensor": self.Utensor,
                "mu": mu,
            }
            if self.gs_wf is not None:
                tmp_dict['gs_wf'] = self.gs_wf

            timestamp = "%d" % datetime.timestamp(datetime.now())
            A[timestamp] = tmp_dict

    def run(self, mu=0.0, itmax=200, mix=0.5, tol=1e-6, beta=200., silence=True, spin_pen=0.0, sz_pen=0.0, idx=0, num_eig=2, ed_verbose=0, diis=False):
        """ Run ghost-RISB self-consistency

        :param itmax: Maxiumum iteraction for self-consistency.
        :type itmax: int

        :param tol: Tolerence for convergence
        :type tol: float

        :param beta: Inverse temperature (equivalent to smearing temperature).
        :type beta: float

        :param silence: Silence the printing.
        :type silence: bool

        :param spin_pen: Penalty for S2 conservation.
        :type spin_pen: float

        :param silence: Orbital index for computing double occupancy.
        :type idx: int

        """
        print("mu = ", mu)
        self.diff = 1e20
        if diis is True:
            #RDIIS = DIIS(7)
            LDIIS = DIIS(7)
            numNonDIIS = 4
        for it in range(itmax):
            # compute qp density matrix
            self.rhok_list=calc_rhoks(self.R, self.Lambda, self.eks, 1./beta)
            self.Delta_p=calc_Delta_p(self.rhok_list)
            self.D=calc_D(self.R, self.Lambda, self.Delta_p, self.eks, self.rhok_list)
            self.Lambda_c=calc_Lambda_c(self.R, self.Lambda, self.Delta_p, self.D, self.Hfull_list)
            if not silence:
                if not self.soc:
                    print("Delta_p=")
                    print(self.Delta_p[::2,::2])
                    print("D=")
                    print(self.D[::2,::2])
                    print("Lambda_c=")
                    print(self.Lambda_c[::2,::2])
                else:
                    print("Delta_p=")
                    print(self.Delta_p[:,:])
                    print("D=")
                    print(self.D[:,:])
                    print("Lambda_c=")
                    print(self.Lambda_c[:,:])
            # ED solvers
            sys.stdout.flush()

            self.solve_embedding(mu, num_eig, ed_verbose, spin_pen, sz_pen)
            #Update R and Update Lambda
            cdaggerf = self.denMat[:self.nimp,self.nimp:]
            ffdagger = self.denMat[self.nimp:,self.nimp:]
            ffdagger = (np.eye(self.nbath,dtype=np.complex128) - ffdagger).T
            #if not silence:
            print("norm(ffdagger.T-Delta_p)=", np.linalg.norm(ffdagger.T-self.Delta_p))
            self.Delta_p = ffdagger.T
            print("Delta_p new:")
            print(self.Delta_p)
            R_new = np.transpose(cdaggerf.dot(funcMat(self.Delta_p, denR)))
            if not self.soc:
                R_new = np.kron(R_new[::2,::2],np.eye(2))# symmetrize
            R_new = svd_truncate_R(R_new)
            #Lambda_new = find_Lambda(self.Lambda, R_new, ffdagger, self.eks, self.Hspin_list, beta)
            Lambda_new = calc_Lambda(R_new, self.Lambda_c, self.Delta_p, self.D, self.Hfull_list)
            if not self.soc:
                Lambda_new = np.kron(Lambda_new[::2,::2],np.eye(2)) # symmetryize
            diff_R = np.abs(self.R-R_new).max()
            diff_Lambda = np.abs(self.Lambda-Lambda_new).max()
            self.diff = max(diff_R,diff_Lambda)
            if diis and ( it >= numNonDIIS ):
                error = Lambda_new - self.Lambda
                error = np.reshape( error, error.shape[0]*error.shape[1] )
                LDIIS.append( error, Lambda_new )
                #error = R_new - self.R
                #error = np.reshape( error, error.shape[0]*error.shape[1] )
                #RDIIS.append( error, R_new )
                self.R = R_new#RDIIS.Solve()
                self.Lambda = LDIIS.Solve()
            else:
                self.R = (1.-mix)*np.copy(self.R) + mix*R_new
                self.Lambda = (1.-mix)*np.copy(self.Lambda) + mix*Lambda_new
#           Try fix a gague that R is non-zero only on the upper left Experiment!
#            tmp = np.zeros(self.R.shape,dtype=self.R.dtype)
#            tmp[:self.nimp,:self.nimp] = sqrtm(self.R.conj().T.dot(self.R)[:self.nimp,:self.nimp])
#            self.R = tmp
            # check point
            with HDFArchive('checkpoint%s.h5' % self.suff,'a') as fh5:
                fh5['R_%d' % it] = self.R
                fh5['Lambda_%d'% it] = self.Lambda
                fh5['eks'] = self.eks
                fh5['Utensor'] = self.Utensor
                fh5['mu'] = mu

            if not silence:
                print("R_new=")
                print(R_new)
                print("R=")
                print(self.R)
                print("Lambda_new=")
                print(Lambda_new)
                print("Lambda=")
                print(self.Lambda)
                print("ffdagger.T")
                print(ffdagger.T)
                print("density matrix=")
                print(self.denMat[::2,::2])

            # Save information
            self.save_data(mu)

            print("iteration:",it,'diff=',self.diff)
            if self.diff < tol or it == (itmax-1):
                print("--------------------- ghost-RISB converged with diff=%g ---------------------"%(self.diff))
                print("density matrix=")
                print(self.denMat)
                self.nfill = np.trace(self.denMat[:self.nimp,:self.nimp])
                self.docc = []
                for idx in range(0,self.nimp,2):
                    self.docc.append(self.edsolver.calc_double_occ(idx))
                print("double occupancy=", self.docc)
                break
            print("##########")
            print()
            sys.stdout.flush()

    def func_mu(self, mu, *args):
        #self.mu_tmp = mu
        nfix, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis = args
        self.run(mu, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis)
        diff = nfix - self.nfill
        print('nfix-nfill=', diff, 'nfill=',self.nfill)
        return diff

    def run_canonical(self, mu0, nfix, itmax=200, mix=0.5, tol=1e-6, beta=200., silence=True, spin_pen=0.0, sz_pen=0.0, idx=0, num_eig=10, ed_verbose=0, diis=False, mu_tol=1e-2, dmu=0.05):
        print('canonical mu0=',mu0)
        self.run(mu0, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis)
        print('nfix-nfill=', nfix - self.nfill, 'nfill=',self.nfill)
        if np.abs(nfix - self.nfill ) < mu_tol:
            return mu0
        else:
            args = ( nfix, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose , diis)
            #sols = scipy.optimize.root(self.func_mu,x0=mu0,args=args,method='lm',tol=1e-3,options={'eps':1e-5,'factor':0.1})
            #sols = scipy.optimize.root_scalar(self.func_mu,x0=mu0,args=args,method='bisect',bracket=(mu0-0.05,mu0+0.05),xtol=1e-3)
            sols = scipy.optimize.root_scalar(self.func_mu,x0=mu0,x1=mu0+dmu,args=args,method='secant',xtol=mu_tol)
            #sols = scipy.optimize.root_scalar(self.func_mu,args=args,xtol=tol)
            print('root solver for mu converged? ',sols.converged)
            mu = sols.root
            print('mu=',mu)
            return mu

    def compute_Gf_Sig(self, mu, ek_path, oms, eta):
        self.oms = oms
        self.eta = eta
        self.Gf, self.Sig = self._compute_Gf_Sig(mu, ek_path, oms, eta, self.R, self.Lambda, self.eloc, self.nbath, self.nimp)

    @staticmethod
    #@numba.jit(nopython=True)
    def _compute_Gf_Sig(mu, ek_path, oms, eta, R, Lambda, eloc, nbath, nimp):
        Gf = np.zeros((ek_path.shape[0],oms.shape[0],nimp,nimp),dtype=np.complex128)#numba.complex128)
        Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=np.complex128)#numba.complex128)
        for ik, ek in enumerate(ek_path):
            for iom, om in enumerate(oms):
                Gf[ik,iom,:,:] = R.conj().T.dot( np.linalg.inv( (om+1j*eta)*np.eye(nbath)
                                  - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
                if ik == 0:
                    Sig[iom,:,:] = (om + 1j*eta + mu)*np.eye(nimp) - ek - eloc - np.linalg.inv(Gf[ik,iom]) #om + 1j*eta - ek - np.linalg.inv(Gf[ik,iom])
        return Gf, Sig

