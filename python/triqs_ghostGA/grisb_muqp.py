from triqs_ghostGA.grisb import *

def calc_rhoks(R, Lambda, eks, T, mu):
    return [calc_nf( np.dot(R, np.dot(x , R.conj().T ) ) + Lambda - mu*np.eye(Lambda.shape[0]), T).T for x in eks]

def calc_Delta_p(rhok_list):
    return sum(rhok_list)/len(rhok_list)

def calc_D(R, Lambda, Delta_p, eks, rhoks):
    """ Compute D matrix
    """
    Left=[np.dot( np.dot(eks[x], R.conj().T ), rhoks[x].T ) for x in range(len(rhoks))]
    Left=sum(Left)/float(len(rhoks))
    Right=funcMat(Delta_p, denR)
    return np.dot(Right,np.transpose(Left))

def calc_Lambda_c(R, Lambda, Delta_p, D, H_list):
    """ Compute Lambda_c matrix
    """
    no = Lambda.shape[0]
    l=inverse_realHcombination(Lambda,H_list)
    lc=np.copy(l)*0.0
    MM=np.dot(D,np.transpose(R))
    for k in range(len(H_list)):
        AA=Delta_p
        HH=H_list[k].T
        derivative=dF(AA,HH, denRm1, ddenRm1)
        tt=np.trace(np.dot(MM,derivative))
        lc[k]=-l[k]-(tt+np.conjugate(tt)).real
    Lambda_c=realHcombination(lc,H_list)
    return Lambda_c

def calc_Lambda(R, Lambda_c, Delta_p, D, H_list):
    """ Compute Lambda_c matrix
    """
    no = Lambda_c.shape[0]
    lc=inverse_realHcombination(Lambda_c,H_list)
    l=np.copy(lc)*0.0
    MM=np.dot(D,np.transpose(R))
    for k in range(len(H_list)):
        AA=Delta_p
        HH=H_list[k].T
        derivative=dF(AA,HH, denRm1, ddenRm1)
        tt=np.trace(np.dot(MM,derivative))
        l[k]=-lc[k]-(tt+np.conjugate(tt)).real
    Lambda=realHcombination(l,H_list)
    return Lambda

def cost_function(x, *args):
    ''' Cost function for find Lambda
    '''
    R, Lambda, eks, nfix_qp, beta = args
    rhok_list=calc_rhoks(R, Lambda, eks, 1./beta, x)
    Delta_p=calc_Delta_p(rhok_list)
    diff = np.trace(Delta_p) - nfix_qp
    return diff.real

# def find_mu(mu0, R, Lambda, eks, nfix_qp, beta, dmu=0.05, mu_tol=0.001):
#     """ Find Lambda for given ffdagger
#     """
#     print('nfix_qp=',nfix_qp)
#     args = (R, Lambda, eks, nfix_qp, beta)
#     sols = scipy.optimize.root_scalar(cost_function,x0=mu0-dmu,x1=mu0+dmu,args=args,method='secant',xtol=mu_tol)
#     mu = sols.root
#     print('root solver for mu converged? ',sols.converged, 'mu=',mu)
#     return mu

def find_mu(mu0, R, Lambda, eks, nfix_qp, beta, dmu=1.0, mu_tol=0.00001):
    """ Find Lambda for given ffdagger
    """
    print('nfix_qp=',nfix_qp)
    args = (R, Lambda, eks, nfix_qp, beta)
    mu = scipy.optimize.bisect(cost_function,mu0-dmu,mu0+dmu,args=args,xtol=mu_tol)
    print('mu=',mu)
    return mu

class Grisb_muqp(Grisb):
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

    :param ed_params: Exact diagonalization solver parameters.
    :type ed_params: dic

    :param Hspin_list: Hermitian matrix basis in each spin block with spin symmetry.
    :type Hspin_list: list

    :param Hfull_list: Full Hermitian matrix basis.
    :type Hfull_list: list

    """
    def compute_energy(self,beta=200.,mu=0.0):
        """ Compute total energy, kinetic energy, and potential energy
        """
        self.ekin = sum([np.sum( ( np.dot(self.R, np.dot(x, self.R.conj().T )) ) * \
                    calc_nf( np.dot(self.R, np.dot(x, self.R.conj().T) ) + self.Lambda - mu*np.eye(self.Lambda.shape[0]), 1./beta).T ) for x in self.eks] )/float(len(self.eks))
        self.epot = self.E2loc + np.trace(self.eloc.dot(self.denMat[:self.nimp,:self.nimp].T))
        self.etot = self.ekin + self.epot - mu*self.nfill

    def run(self, mu0=0.0, itmax=200, mix=0.5, tol=1e-6, beta=200., silence=True, spin_pen=0.0, idx=0, num_eig=2, ed_verbose=0, diis=False, canonical=False, nfix=None, dmu=0.001, mu_tol=0.001):
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
        print("mu0 = ", mu0)
        self.mu = mu0
        self.diff = 1e20
        if diis is True:
            #RDIIS = DIIS(7)
            LDIIS = DIIS(7)
            numNonDIIS = 4
        for it in range(itmax):
            # compute qp density matrix
            if canonical:
                if it > 4 or self.mu == 0:
                    nfix_qp = (self.nbath - self.nimp)/2 + nfix
                    self.mu = find_mu(self.mu, self.R, self.Lambda, self.eks, nfix_qp, beta, dmu=dmu, mu_tol=mu_tol)
                else:
                    print("Initial chemical potential: ", self.mu)

            self.rhok_list=calc_rhoks(self.R, self.Lambda, self.eks, 1./beta, self.mu)
            self.Delta_p=calc_Delta_p(self.rhok_list)
            self.D=calc_D(self.R, self.Lambda, self.Delta_p, self.eks, self.rhok_list)
            self.Lambda_c=calc_Lambda_c(self.R, self.Lambda, self.Delta_p, self.D, self.Hfull_list)
            if not silence:
                if self.spin_sym:
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
            self.solve_embedding(0.0, num_eig, ed_verbose, spin_pen)
            #Update R and Update Lambda
            cdaggerf = self.denMat[:self.nimp,self.nimp:]
            ffdagger = self.denMat[self.nimp:,self.nimp:]
            ffdagger = (np.eye(self.nbath,dtype=np.complex128) - ffdagger).T
            if not silence:
                print("norm(ffdagger.T-Delta_p)=", np.linalg.norm(ffdagger.T-self.Delta_p))
            self.Delta_p = ffdagger.T
            R_new = np.transpose(cdaggerf.dot(funcMat(self.Delta_p, denR)))
            if self.spin_sym:
                R_new = np.kron(R_new[::2,::2],np.eye(2))# symmetrize
            Lambda_new = calc_Lambda(R_new, self.Lambda_c, self.Delta_p, self.D, self.Hfull_list)
            if self.spin_sym:
                Lambda_new = np.kron(Lambda_new[::2,::2],np.eye(2)) # symmetryize
            diff_R = np.abs(self.R-R_new).max()
            diff_Lambda = np.abs(self.Lambda-Lambda_new).max()
            self.diff = max(diff_R,diff_Lambda)
            if diis and ( it >= numNonDIIS ):
                error = Lambda_new - self.Lambda
                error = np.reshape( error, error.shape[0]*error.shape[1] )
                LDIIS.append( error, Lambda_new )
                self.R = R_new#RDIIS.Solve()
                self.Lambda = LDIIS.Solve()
            else:
                self.R = (1.-mix)*np.copy(self.R) + mix*R_new
                self.Lambda = (1.-mix)*np.copy(self.Lambda) + mix*Lambda_new

            with HDFArchive('checkpoint%s.h5' % self.suff,'a') as fh5:
                fh5['R_%d' % it] = self.R
                fh5['Lambda_%d'% it] = self.Lambda
                fh5['eks'] = self.eks
                fh5['Utensor'] = self.Utensor
                fh5['mu'] = self.mu

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
                Gf[ik,iom,:,:] = R.conj().T.dot( np.linalg.inv( (om+1j*eta + mu)*np.eye(nbath)
                                  - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
                if ik == 0:
                    Sig[iom,:,:] = (om + 1j*eta + mu)*np.eye(nimp) - ek - eloc - np.linalg.inv(Gf[ik,iom]) #om + 1j*eta - ek - np.linalg.inv(Gf[ik,iom])
        return Gf, Sig

