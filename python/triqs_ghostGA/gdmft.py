import scipy
from scipy.linalg import sqrtm
from scipy.optimize import bisect
import h5py
import numpy as np
import numba
from triqs_ghostGA.utility.utils_TH import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, \
    Hermitian_list, get_blocks, funcMat, calc_nf, dF
from triqs_ghostGA.DIIS import *
from triqs_ghostGA.utility.delta_fit import * 
from h5 import *
from triqs_ghostGA.utility.utils_grisb import calc_rhoks, calc_Delta_p, calc_D, calc_Lambda_c, calc_Lambda_c_new, calc_Lambda
import sys
import time

def calc_right(R, Lambda, Delta_p, eks, rhoks):
    """ Compute the right matrix for DMFT-like algorithm
    """
    right=[np.dot( np.dot(eks[x], R.conj().T ), rhoks[x].T ) for x in range(len(rhoks))]
    right=sum(right)/float(len(rhoks))
    return right

def calc_denMat0(R, Lambda, D, Lambda_c, beta):
    Hbar = np.zeros((Lambda.shape[0]*2,Lambda.shape[1]*2),dtype=Lambda.dtype)
    Hbar[:Lambda.shape[0],:Lambda.shape[1]] = Lambda
    Hbar[:Lambda.shape[0],Lambda.shape[1]:] = (D.real.dot(R.T)).T
    Hbar[Lambda.shape[0]:,:Lambda.shape[1]] = (D.real.dot(R.T)).conj()
    Hbar[Lambda.shape[0]:,Lambda.shape[1]:] = -Lambda_c.real
    denMat0 = calc_nf(Hbar,1./beta).T
    return denMat0

def cost_function_D_Lamc_dmft(x, *args):
    ''' Cost function for finding D and Lambdac Lambda within DMFT-like algorithm
    '''
    Delta_p, right, E, R, Lambda, Hspin_list, beta = args
    Lambda_c_spin = realHcombination(x[:len(Hspin_list)], Hspin_list)
    D_spin = x[len(Hspin_list):].reshape((Lambda.shape[0]//2,E.shape[0]//2))
    Lambda_c = np.kron(Lambda_c_spin,np.eye(2))
    D = np.kron(D_spin,np.eye(2))
    denMat0 = calc_denMat0(R, Lambda, D, Lambda_c, beta)
    #print(Hbar.shape, D.shape, right.shape)
    diff = np.linalg.norm( ( denMat0[:Lambda.shape[0],:Lambda.shape[0]] - Delta_p ) )
    diff += np.linalg.norm( ( (denMat0[:Lambda.shape[0],Lambda.shape[0]:].dot(D)).T - right ) )
    return diff.real

def find_D_Lambdac_dmft(Lambda, R, E, D0, Lambda_c0, Delta_p, right, Hspin_list, beta):
    """ Find Lambda for given ffdagger
    """
    #success = False
    #while not success:
    D0_spin = D0[::2,::2].real
    Lambda_c0_spin = Lambda_c0[::2,::2].real
    #Lambda0_spin += (fluc + fluc.T)/2
    x = np.hstack((inverse_realHcombination(Lambda_c0_spin, Hspin_list), D0_spin.flatten()))
    args = (Delta_p, right, E, R, Lambda, Hspin_list, beta)
    result = scipy.optimize.minimize( cost_function_D_Lamc_dmft, x, args=args, tol=1e-5, method='BFGS', options={'disp':False, 'eps': 1e-10} )
    if ( result.success==False ):
        print("   Minimize mesage ::",result.message)
    print("   Minimize :: Cost function after convergence =", np.sum(result.fun))#/len(result.fun))
    success = result.success
    print('success=',success)
    Lambda_c = np.kron(realHcombination(result.x[:len(Hspin_list)], Hspin_list),np.eye(2))
    D = np.kron(result.x[len(Hspin_list):].reshape(D0_spin.shape),np.eye(2))
    return D.real, Lambda_c.real # restrict to real for testing

def cost_function_R_Lam_dmft(x, *args):
    ''' Cost function for finding D and Lambdac Lambda within DMFT-like algorithm
    '''
    denMat, E, D, Lambda_c, Hspin_list, beta = args
    Lambda_spin = realHcombination(x[:len(Hspin_list)], Hspin_list)
    R_spin = x[len(Hspin_list):].reshape((D.shape[0]//2,D.shape[1]//2))
    Lambda = np.kron(Lambda_spin,np.eye(2))
    R = np.kron(R_spin,np.eye(2))
    denMat0 = calc_denMat0(R, Lambda, D, Lambda_c, beta)
    #print(denMat0.shape, denMat.shape)
    diff = np.linalg.norm( ( denMat0[Lambda.shape[0]:,Lambda.shape[0]:] - denMat[E.shape[0]:,E.shape[1]:] ) )
    diff += np.linalg.norm( ( R.T.dot(denMat0[:Lambda.shape[0],Lambda.shape[0]:]) - denMat[:E.shape[0],E.shape[1]:] ) )
    return diff.real

def find_R_Lambda_dmft(Lambda0, R0, E, D, Lambda_c, denMat, Hspin_list, beta):
    """ Find Lambda for given ffdagger
    """
    #success = False
    #while not success:
    R0_spin = R0[::2,::2].real
    Lambda0_spin = Lambda0[::2,::2].real
    x = np.hstack((inverse_realHcombination(Lambda0_spin, Hspin_list), R0_spin.flatten()))
    args = (denMat, E, D, Lambda_c, Hspin_list, beta)
    result = scipy.optimize.minimize( cost_function_R_Lam_dmft, x, args=args, tol=1e-5, method='BFGS', options={'disp':False, 'eps': 1e-10} )
    if ( result.success==False ):
        print("   Minimize mesage ::",result.message)
    print("   Minimize :: Cost function after convergence =", np.sum(result.fun))#/len(result.fun))
    success = result.success
    print('success=',success)
    Lambda = np.kron(realHcombination(result.x[:len(Hspin_list)], Hspin_list),np.eye(2))
    R = np.kron(result.x[len(Hspin_list):].reshape(R0_spin.shape),np.eye(2))
    return R.real, Lambda.real # restrict to real for testing


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
    def __init__(self, ntot, nimp, nbath, eks, eloc, Utensor, spin_sym=True, soc=False, R=None, Lambda=None, D=None, Lambda_c=None, edsolver=None, suff=''):
        print("##### INITIALIZATON OF THE GRISB OBJECT (DMFT-like algorithm)#####")
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
            self.R = np.kron(np.ones((nbath//2,nimp//2)), np.eye(2))*(0.5+0j)
        else:
            if R.shape != (nbath,nimp):
                raise ValueError("R has inconsistent shape. Should be (nbath,nimp)")
            self.R = R
        if Lambda is None:
            #self.Lambda = np.kron(np.random.rand(nbath//2,nbath//2)+1j*np.random.rand(nbath//2,nbath//2), np.eye(2))
            self.Lambda = np.zeros((nbath,nbath),dtype=complex)
            self.Lambda[0,0] = 1.0
            self.Lambda[1,1] = 1.0
            self.Lambda[2,2] =-1.0
            self.Lambda[3,3] =-1.0
        else:
            if Lambda.shape != (nbath,nbath):
                raise ValueError("Lambda has inconsistent shape. Should be (nbath,nbath)")
            self.Lambda = Lambda
        # initialize R and Lambda
        if D is None:
            self.D = np.kron(np.ones((nbath//2,nimp//2)), np.eye(2))*(0.5+0j)
            #raise ValueError("Please initialize D")
        else:
            if D.shape != (nbath,nimp):
                raise ValueError("D has inconsistent shape. Should be (nbath,nimp)")
            self.D = D
        if Lambda_c is None:
            #self.Lambda = np.kron(np.random.rand(nbath//2,nbath//2)+1j*np.random.rand(nbath//2,nbath//2), np.eye(2))
            self.Lambda_c = np.zeros((nbath,nbath),dtype=complex)
            self.Lambda_c[0,0] = 1.0
            self.Lambda_c[1,1] = 1.0
            self.Lambda_c[2,2] =-1.0
            self.Lambda_c[3,3] =-1.0
        else:
            if Lambda_c.shape != (nbath,nbath):
                raise ValueError("Lambda has inconsistent shape. Should be (nbath,nbath)")
            self.Lambda_c = Lambda_c

        if edsolver is None:
            raise ValueError("Not edsolver was passed to the GRISB.")
        else:
            self.edsolver = edsolver

        # initialize single-particle basis
        self.Hspin_list,self.tHspin_list=Hermitian_list(nbath//2)
        self.Hfull_list,self.tHfull_list=Hermitian_list(nbath)
        print('initial R matirx =')
        print(self.R)
        print('initial Lambda matirx =')
        print(self.Lambda)
        print('initial D matirx =')
        print(self.D)
        print('initial Lambda_c matirx =')
        print(self.Lambda_c)
        print("##### END OF THE INITIALIZATION #####")

    def build_h1e(self,mu):
        h1e = np.zeros((self.ntot,self.ntot), dtype=np.complex128)
        h1e[:self.nimp,:self.nimp] = self.eloc - mu*np.eye(self.nimp)
        h1e[:self.nimp,self.nimp:] = self.D.T
        h1e[self.nimp:,self.nimp:] = -self.Lambda_c
        h1e[self.nimp:,:self.nimp] = self.D.conj()
        return h1e

    def solve_embedding(self, mu, num_eig, ed_verbose, spin_pen, sz_pen=0.0, beta=None):
        """ Solve embedding problem using a variety of impurity solver
        """
        fh5 = h5py.File('hemb_test.h5','w')
        fh5['eloc'] = self.eloc
        fh5['D'] = self.D
        fh5['Lambda_c'] = self.Lambda_c
        fh5['Utensor'] = self.Utensor
        fh5['mu'] = mu
        fh5.close()

        if self.edsolver.type == "CI":
            h1e = self.build_h1e(mu)
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor)

        elif self.edsolver.type == "FTPS":
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)

        elif self.edsolver.type == "ITensorMPSSolver":
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)

        elif self.edsolver.type == "PySCFCCSD":
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)

        elif self.edsolver.type == "Block2NSZ":
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)

        else:
            raise ValueError("only Full ED, CI, and HCI are supported")
            # TODO: Replace whole if-clause by edsolver.prolog(self) implemented by
            # Solver(AbstractSolver)

        if(self.edsolver.thermal):
            self.edsolver.solve_Hemb(num_eig=num_eig, verbose=ed_verbose , beta=beta)
        else:
            self.edsolver.solve_Hemb(num_eig=num_eig, verbose=ed_verbose )
            
        self.denMat = self.edsolver.calc_density_matrix()
        self.E2loc = self.edsolver.compute_E2loc()

    def compute_energy(self,beta=200.,mu=0.0):
        """ Compute total energy, kinetic energy, and potential energy
        """
        #self.ekin = [np.sum(self.R.dot( self.eks[x] ).dot( self.R.conj().T )*self.rhok_list[x].T ) for x in range(len(self.rhok_list))]
        #self.ekin = sum(self.ekin)/float(len(self.rhok_list))
        self.ekin = sum([np.sum( ( np.dot(self.R, np.dot(x, self.R.conj().T )) ) * \
                    calc_nf( np.dot(self.R, np.dot(x, self.R.conj().T) ) + self.Lambda , 1./beta).T ) for x in self.eks] )/float(len(self.eks))
        self.epot = self.E2loc + np.trace(self.eloc.dot(self.denMat[:self.nimp,:self.nimp].T))
        self.etot = self.ekin + self.epot - mu*self.nfill

    def run_dmft(self, mu=0.0, itmax=200, mix=0.5, tol=1e-6, beta=200., n_target=None, silence=True, spin_pen=0.0, sz_pen=0.0, idx=0, num_eig=2, ed_verbose=0, diis=False, method='minimize'):
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
        for it in range(itmax):
            # experimental: setting R=I
            #self.R = np.zeros((self.nbath,self.nimp),dtype=np.complex128)
            #self.R[:self.nimp,:self.nimp] = np.eye(self.nimp,dtype=np.complex128) # set R to identity
            #print(self.R)
            # compute qp density matrix
            self.rhok_list=calc_rhoks(self.R, self.Lambda, self.eks, 1./beta)
            self.Delta_p=calc_Delta_p(self.rhok_list)
            
            dvals,dvecs = np.linalg.eigh(self.Delta_p)
            print('dqp vals:',dvals)
            self.D=calc_D(self.R, self.Lambda, self.Delta_p, self.eks, self.rhok_list)
            self.Lambda_c=calc_Lambda_c(self.R, self.Lambda, self.Delta_p, self.D, self.Hfull_list)
            lcvals,lcvecs = np.linalg.eigh(self.Lambda_c[::2,::2])
            print('lcvals:',lcvals)
            print(lcvecs.shape)
            lcvals=0.5*(lcvals-lcvals[::-1])
            self.Lambda_c =  np.kron( lcvecs @ np.diag(lcvals) @ lcvecs.T.conj() , np.eye(2) )
            self.D = np.abs(lcvecs.T.conj() @ self.D[::2,::2])
            self.D =  0.5*(self.D+self.D[::-1,::-1])
            self.D = np.kron( lcvecs @ self.D , np.eye(2) )
            time.sleep(1)
            #right = calc_right(self.R, self.Lambda, self.Delta_p, self.eks, self.rhok_list)
            #self.D, self.Lambda_c = find_D_Lambdac_dmft(self.Lambda, self.R, self.eloc, self.D, self.Lambda_c, self.Delta_p, right,self.Hspin_list, beta)
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
            self.solve_embedding(mu, num_eig, ed_verbose, spin_pen, sz_pen, beta=beta)
            #Update R and Update Lambda
            
            self.nfill = np.trace(self.denMat[:self.nimp,:self.nimp])
            print("||||||||||||||||||| n_filling:",self.nfill)
            cdaggerf = self.denMat[:self.nimp,self.nimp:]
            ffdagger = self.denMat[self.nimp:,self.nimp:]
            ffdagger = (np.eye(self.nbath,dtype=np.complex128) - ffdagger).T
            #if not silence:
            #print("norm(ffdagger.T-Delta_p)=", np.linalg.norm(ffdagger.T-self.Delta_p)) # not necessary in the DMFT algorithm
            #self.Delta_p = ffdagger.T # not necessary in the DMFT algorithm
            
            dvals,dvecs = np.linalg.eigh(ffdagger)
            print('dbath vals:',dvals)
            #Lambda_new = calc_Lambda(self.R, self.Lambda_c, self.Delta_p, self.D, self.Hfull_list)
            #if not self.soc:
            #    Lambda_new = np.kron(Lambda_new[::2,::2],np.eye(2)) # symmetryize
            #self.rhok_list=calc_rhoks(self.R, self.Lambda, self.eks, 1./beta)
            #self.Delta_p=calc_Delta_p(self.rhok_list)
            #R_new = np.transpose(cdaggerf.dot(funcMat(self.Delta_p, denR)))
            #if not self.soc:
            #    R_new = np.kron(R_new[::2,::2],np.eye(2))# symmetrize
            #R_new, Lambda_new = find_R_Lambda_dmft(self.Lambda, self.R, self.eloc, self.D, self.Lambda_c,
            #                                       self.denMat, self.Hspin_list, beta)#.real #restrict Lambda to real
            D11_target=self.denMat[self.nimp:,self.nimp:]
            D12_target=self.denMat[:self.nimp,self.nimp:]
            print("D11:",D11_target)
            print("D12:",D12_target)
            print("|||||||||||||||||||||||||||||n_tot:",np.trace(self.denMat))
            if method == 'minimize':
                res, Lambda_new, R_new = solve_F_dF_minimize(beta, self.Lambda_c[::2,::2], self.D[::2,::2], self.Lambda[::2,::2], self.R[::2,::2], D11_target[::2,::2], D12_target[::2,::2])
            elif method == 'root':
                Lambda_new, R_new = new_self_energy(self.Lambda[::2,::2],self.R[::2,::2],self.Lambda_c[::2,::2],self.D[::2,::2], D11_target[::2,::2],D12_target[::2,::2], beta=beta, method="dF")
                
                U,s,Vh = scipy.linalg.svd(R_new)
                if(np.any(np.abs(s)>1)):
                    Lambda_new, R_new = new_self_energy_penalty(self.Lambda[::2,::2],self.R[::2,::2],self.Lambda_c[::2,::2],self.D[::2,::2], D11_target[::2,::2],D12_target[::2,::2], beta=beta)

            L_eval_new, UL_new = np.linalg.eigh(Lambda_new)
            L_eval_old, UL_old = np.linalg.eigh(self.Lambda[::2,::2])
            L_eval_new = 0.5*(L_eval_new - L_eval_new[::-1])
            R_new = np.abs(UL_new.T.conj()@R_new)
            R_new = 0.5*(R_new+R_new[::-1,::-1])
            Lambda_new =  np.diag(L_eval_new)
            diff_R = 0.0 #np.abs(UL_old@self.R[::2,::2]-UL_new@R_new).max()
            R_new = np.kron(R_new, np.eye(2))
            Lambda_new = np.kron(Lambda_new, np.eye(2))
            diff_Lambda = np.abs(L_eval_new-L_eval_old).max()
            print('lambda_evals:',L_eval_new)
            time.sleep(1)
            self.diff = max(diff_R,diff_Lambda)
            # mixing 
            self.R = (1.-mix)*np.copy(self.R) + mix*R_new
            self.Lambda = (1.-mix)*np.copy(self.Lambda) + mix*Lambda_new
            convg_n=True
            if( (not (n_target is None)) and it>1):
                print("CHECK DENSITY")
                if( abs(self.nfill-n_target)>1e-2):
                    convg_n=False
                if( abs(self.nfill-n_target)>1e-3):
                    self.find_mu_qp(n_target,beta)
            if not silence:
                #print("R_new=")
                #print(R_new)
                print("R=")
                print(self.R[::2,::2])
                print("Lambda_new=")
                print(Lambda_new[::2,::2])
                print("Lambda=")
                print(self.Lambda[::2,::2])
                print("ffdagger.T")
                print(ffdagger[::2,::2].T)
                print(np.linalg.eigh(ffdagger[::2,::2].T)[0])
                print("Delta_p")
                print(self.Delta_p[::2,::2])
                print(np.linalg.eigh(self.Delta_p[::2,::2])[0])
                print("density matrix=")
                print(self.denMat[::2,::2])
                denMat0 = calc_denMat0(self.R, self.Lambda, self.D, self.Lambda_c, beta)
                print("density matrix 0=")
                print(denMat0[::2,::2])
            print("iteration:",it,'diff=',self.diff)
            if (self.diff < tol and it>1) or it == (itmax-1):
                print("--------------------- ghost-RISB converged with diff=%g ---------------------"%(self.diff))
                print("density matrix=")
                print(self.denMat)
                self.nfill = np.trace(self.denMat[:self.nimp,:self.nimp])
                self.docc = []
                for idx in range(0,self.nimp,2):
                    self.docc.append(self.edsolver.calc_double_occ(idx))
                print("double occupancy=", self.docc)
                print("convg_n",convg_n)
                print('lambda eigvals',np.linalg.eigvalsh(self.Lambda))
                break

    def func_mu(self, x, *args):
        mu = x
        #self.mu_tmp = mu
        nfix, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis = args
        self.run_dmft(mu, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis)
        diff = nfix - self.nfill 
        print('nfix-nfill=', diff, 'nfill=',self.nfill)
        return diff

    def run_canonical(self, mu0, nfix, itmax=200, mix=0.5, tol=1e-6, beta=200., silence=True, spin_pen=0.0, sz_pen=0.0, idx=0, num_eig=2, ed_verbose=0, diis=False, mu_tol=1e-2, dmu=0.05):
        print('canonical mu0=',mu0)
        self.run_dmft(mu0, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis)
        print('nfix-nfill=', nfix - self.nfill, 'nfill=',self.nfill)
        if np.abs(nfix - self.nfill ) < mu_tol:
            return mu0
        else:
            args = ( nfix, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose , diis)
            #sols = scipy.optimize.root(self.func_mu,x0=mu0,args=args,method='lm',tol=1e-3,options={'eps':1e-5,'factor':0.1})
            # scipy.root handle
            #print(sols.message)
            #print('root solver for mu converged? ',sols.success)
            #mu = sols.x[0]
            sols = scipy.optimize.root_scalar(self.func_mu,x0=mu0,args=args,method='bisect',bracket=(mu0-0.1,mu0+0.1),xtol=1e-3)
            #sols = scipy.optimize.root_scalar(self.func_mu,x0=mu0,x1=mu0+dmu,args=args,method='secant',xtol=mu_tol)
            #sols = scipy.optimize.root_scalar(self.func_mu,args=args,xtol=tol)
            # scipy.root_scalr handle
            print(sols.flag)
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


        
    
    def find_mu_qp(self,n_target,beta):

        def dens_qp(mu, dens_qp_2f,Lambda,R,eks,beta):
            mu_diag =  np.eye(R.shape[-1])*mu
            Lambda_tmp = Lambda + R @ mu_diag @ R.T.conj()
            rhoks = calc_rhoks(R,Lambda_tmp,eks,1/beta)
            Delta = calc_Delta_p(rhoks)
            return np.trace(Delta)-dens_qp_2f
        
        if( n_target<=0.0 or n_target>=self.nimp ): raise ValueError("Wrong n_target")
        dens_qp_2f = (self.nbath-self.nimp)/2.0 + n_target
        eL = np.linalg.eigvalsh(self.Lambda)
        mu_sol = bisect(f=dens_qp,
                        a=np.min(eL)-5.0,
                        b=np.max(eL)+5.0,
                        xtol=1e-6,
                        args=(dens_qp_2f,self.Lambda,self.R,self.eks,beta))
        print('Solution for mu finding is:',mu_sol)
        mu_diag =  np.eye(self.eloc.shape[0])*mu_sol
        self.eloc = self.eloc + mu_diag
        self.Lambda = self.Lambda + self.R @ mu_diag @ self.R.T.conj()
        print( 'dens_qp',dens_qp( 0.0,0.0,self.Lambda,self.R,self.eks,beta))
        return
