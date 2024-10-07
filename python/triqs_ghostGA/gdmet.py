from triqs_ghostGA.grisb import *

def cost_function(x, *args):
    ''' Cost function for find R and Lambda in gDMET formalism
    '''
    Delta_p, E, D, Lambda_c, Hspin_list, beta = args
    Lambda_spin = realHcombination(x[:len(Hspin_list)], Hspin_list)
    R_spin = x[len(Hspin_list):].reshape((Lambda_c.shape[0]//2,E.shape[0]//2))
    Lambda = np.kron(Lambda_spin,np.eye(2))
    R = np.kron(R_spin,np.eye(2))

    Hbar = np.zeros((Lambda.shape[0]*2,Lambda.shape[1]*2),dtype=Lambda.dtype)
    Hbar[:Lambda.shape[0],:Lambda.shape[1]] = Lambda
    Hbar[:Lambda.shape[0],Lambda.shape[1]:] = (D.real.dot(R.T)).T
    Hbar[Lambda.shape[0]:,:Lambda.shape[1]] = (D.real.dot(R.T)).conj()
    Hbar[Lambda.shape[0]:,Lambda.shape[1]:] = -Lambda_c.real
    denMat0 = calc_nf(Hbar,1./beta).T
    #print(denMat0[Lambda.shape[0]:,:Lambda.shape[1]])
    #print(funcMat(Delta_p, denRm1))
    # below diff is for scipy.optimize.minimize, where we have to return a value.
    diff = np.linalg.norm((denMat0[Lambda.shape[0]:,Lambda.shape[1]:]-(np.eye(Delta_p.shape[0])-Delta_p)))
    diff += np.linalg.norm((R.T.dot(denMat0[Lambda.shape[0]:,:Lambda.shape[1]])-R.T.dot(funcMat(Delta_p, denRm1))))
    #diff+= np.linalg.norm(denMat0[:Lambda.shape[0],:Lambda.shape[1]]-Delta_p) # if we want to minimize also the <f^\dagger f> = Delta_p
    return diff.real
    # below diff is for scipy.optimize.root, where we have to return a np.ndarray.
    #diff1 = denMat0[Lambda.shape[0]:,Lambda.shape[1]:]-(np.eye(Delta_p.shape[0])-Delta_p)
    #diff2 = R.T.dot(denMat0[Lambda.shape[0]:,:Lambda.shape[1]])-R.T.dot(funcMat(Delta_p, denRm1))
    #return np.hstack((inverse_realHcombination( diff1[::2,::2], Hspin_list), diff2[::2,::2].flatten().real))

def find_R_Lambda(Lambda0, R0, E, D, Lambda_c, Delta_p, Hspin_list, beta):
    """ Find R and Lambda for given ffdagger
        NOTE: for testing we assume spin symmetric and real R and Lambda.
    """
    R0_spin = R0[::2,::2].real
    Lambda0_spin = Lambda0[::2,::2].real
    x = np.hstack((inverse_realHcombination(Lambda0_spin, Hspin_list), R0_spin.flatten()))
    args = (Delta_p, E, D, Lambda_c, Hspin_list, beta)
    # below we can choose either scipy.optimize.minimize or scipy.optimize.root
    # different minimization algorithm: 'L-BFGS-B' seems to be more stable
    result = scipy.optimize.minimize( cost_function, x, args=args, tol=1e-5, method='L-BFGS-B', options={'disp':False, 'eps':1e-12, 'maxiter': len(x)*100000} )
    # different root algorithm: 'lm' seems to be more stable
    #result = scipy.optimize.root( cost_function, x, args=args, method='lm', options={'eps':1e-10, 'factor':0.5, 'maxiter': len(x)*1000} )
    #result = scipy.optimize.root( cost_function, x, args=args, method='hybr', options={'eps':1e-12, 'factor':0.1} )
    if ( result.success==False ):
        print("   Minimize mesage ::",result.message)
    print("   Minimize :: Cost function after convergence =", np.sum(result.fun))#/len(result.fun))
    success = result.success
    print('success=',success)
    Lambda = np.kron(realHcombination(result.x[:len(Hspin_list)], Hspin_list),np.eye(2))
    R = np.kron(result.x[len(Hspin_list):].reshape(R0_spin.shape),np.eye(2))
    return R.real, Lambda.real # restrict to real for testing


class Gdmet(Grisb):
    """This is a class representation of a ghost-RISB object with DMET-like algorithm.

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

    def run(self, mu0=0.0, itmax=200, mix=0.5, tol=1e-6, beta=200., silence=True, spin_pen=0.0, sz_pen=0.0, idx=0, num_eig=2, ed_verbose=0, diis=False, nfix=None, dmu=0.1, mu_tol=1e-8, nfix_tol=0.01):
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

        print("########## STARTING THE GHOST-GA LOOP ##########")

        self.mu = mu0
        self.diff = 1e20

        if nfix is not None:
            print("### Calculation performed in the canonical ensemble, starting with mu=", self.mu)

        if diis is True:
            #RDIIS = DIIS(7)
            LDIIS = DIIS(7)
            numNonDIIS = 4

        for it in range(itmax):
            print("### Iteration %d" % it)

            # If number of electron is fixed, recalculate chemical potential to get the right number of electrons
            if nfix is not None:
                # Sometimes you want to start from a given chemical potential
                if it > 1:
                    # Target number of particle in the embedded space
                    # TODO: I don't understand this
                    nfix_qp = (self.nbath - self.nimp)/2 + nfix
                    # Optimize to find chemical potential mu
                    mu_old = self.mu
                    mu_new = find_mu(mu_old, self.R, self.Lambda, self.eks, nfix_qp, beta, dmu=dmu, mu_tol=mu_tol)
                    self.mu = mu_new #mu_old*(1-mix) + mu_new*mix

            # From Lambda and R, calculate Delta_p, D and Lambda_c
            print("# With Lambda and R, compute Delta_p, D and Lambda_c")
            # self.rhok_list = calc_rhoks(self.R, self.Lambda, self.eks, 1./beta, self.mu)
            self.rhok_list = np.real(calc_rhoks(self.R, self.Lambda, self.eks, 1./beta, self.mu))
            # self.Delta_p = calc_Delta_p(self.rhok_list)
            self.Delta_p = np.real(calc_Delta_p(self.rhok_list))
            # self.D = calc_D(self.R, self.Lambda, self.Delta_p, self.eks, self.rhok_list)
            self.D = np.real(calc_D(self.R, self.Lambda, self.Delta_p, self.eks, self.rhok_list))
            # self.Lambda_c = calc_Lambda_c(self.R, self.Lambda, self.Delta_p, self.D, self.Hfull_list)
            self.Lambda_c = np.real(calc_Lambda_c(self.R, self.Lambda, self.Delta_p, self.D, self.Hfull_list))

            # TODO: Nicer print and options for verbose
            if not silence:
                if self.spin_sym:
                    print("Delta_p=")
                    print(self.Delta_p[::2,::2])
                    print("D=")
                    print(self.D[::2,::2])
                    print("Lambda_c=")
                    print(self.Lambda_c[::2,::2])
                    print()
                else:
                    print("Delta_p=")
                    print(self.Delta_p[:,:])
                    print("D=")
                    print(self.D[:,:])
                    print("Lambda_c=")
                    print(self.Lambda_c[:,:])
                    print()
            sys.stdout.flush()

            # With Lambda, R, Delta_p, D and Lambda_c, we have constructed the embedding Hamiltonian.
            # We now solve with the solver passed as an argument.
            print()
            print("# Solving the embedding Hamiltonian:")
            self.solve_embedding(self.mu, num_eig, ed_verbose, spin_pen, sz_pen)

            # Extract relevant quantities from the density matrix, such as Delta_p
            cdaggerf = self.denMat[:self.nimp,self.nimp:]
            ffdagger = self.denMat[self.nimp:,self.nimp:]
            ffdagger = (np.eye(self.nbath,dtype=np.complex128) - ffdagger).T
            if not silence:
                # Compare previous Delta_p with new
                print("norm(ffdagger.T-Delta_p)=", np.linalg.norm(ffdagger.T-self.Delta_p))
                print("new Delta_p =", ffdagger.T)
                print()
            # self.Delta_p = ffdagger.T
            self.Delta_p = np.real(ffdagger.T)

            # TODO: Separate function
            r"""Calculate R matrix, where the element are given by

            .. math:
                R_{b\alpha} = \sum_a \langle \Phi | c^\dagger_\alpha f_a | \Phi \rangle \left[ \Delta ( 1 - \Delta ) \right]^{-1/2}_{ad}
            """
            # R_new = np.transpose(cdaggerf.dot(funcMat(self.Delta_p, denR)))
            R_new = np.real(np.transpose(cdaggerf.dot(funcMat(self.Delta_p, denR))))

            # Spin symmetry for R
            if self.spin_sym:
                R_new = np.kron(R_new[::2,::2],np.eye(2))# symmetrize

            # Calculate the new Lambda from R, Lambda_c, Delta_p and D
            # Lambda_new = calc_Lambda(R_new, self.Lambda_c, self.Delta_p, self.D, self.Hfull_list)
            Lambda_new = np.real(calc_Lambda(R_new, self.Lambda_c, self.Delta_p, self.D, self.Hfull_list))

            # Spin symmetry for Lambda
            if self.spin_sym:
                Lambda_new = np.kron(Lambda_new[::2,::2],np.eye(2)) # symmetryize

            R_new, Lambda_new = find_R_Lambda(Lambda_new, R_new, self.eloc, self.D, self.Lambda_c, self.Delta_p, self.Hspin_list, beta)

            # Calculate difference of R and Lambda from prior iteration and check convergence
            # and apply mixing if required
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

            # Save checkpoint
            # TODO: Improve this
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
                print()

            if nfix is not None:
                # occ = occupation_vs_mu(self.mu, self.R, self.Lambda, self.eks, beta)
                occ = np.trace(self.denMat[:self.nimp, :self.nimp])
                print(self.denMat[:self.nimp, :self.nimp])
                print("# New occupation : ", occ, ", nfix : ", nfix)
                print()

            print("# iteration:",it,'diff=',self.diff)
            print()

            if self.diff < tol or it == (itmax-1):
                if nfix is None or (nfix is not None and (occ - nfix < nfix_tol)):
                    print("--------------------- ghost-RISB converged with diff=%g ---------------------"%(self.diff))
                    print("density matrix=")
                    print(self.denMat)
                    self.nfill = np.trace(self.denMat[:self.nimp,:self.nimp])
                    self.docc = []
                    for idx in range(0,self.nimp,2):
                        self.docc.append(self.edsolver.calc_double_occ(idx))
                    print("double occupancy=", self.docc)
                    break


