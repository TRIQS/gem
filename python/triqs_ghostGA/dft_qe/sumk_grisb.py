from triqs_dft_tools.sumk_dft import *
from triqs_ghostGA.utility.utils_TH import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, \
     Hermitian_list, get_blocks, funcMat, calc_nf, dF, cut_small

class SumkGRISB(SumkDFT):
    '''
    Inherent from SumkDFT for GRISB k-summation
    '''
    def __init__(self, *args, nbaths, **kwargs):
        '''
        Inherent all initial parameters from sumk_dft
        '''
        super().__init__(*args, **kwargs)
        # additional grisb parameters
        self.nbaths = nbaths
        if mpi.is_master_node():
            print('nbaths=', self.nbaths)

        # rotation matrix from Wannier90 convention z^2 xz yz x^2-y^2 xy to z^2 x^2-y^2 xz yz xy convention
        # NOTE: for testing d-shell calculations
        self.u_trans_w90_to_std = np.array([[ 1, 0, 0, 0, 0],
                                            [ 0, 0, 0, 1, 0],
                                            [ 0, 1, 0, 0, 0],
                                            [ 0, 0, 1, 0, 0],
                                            [ 0, 0, 0, 0, 1]], dtype=float)

        # read additional data u_total transformation matrix from bloch to wannier90 orbitals
        if not isinstance(self.hdf_file, str):
            mpi.report("Give a string for the hdf5 filename to read the input!")
        else:
            # additional properties to load
            # soon bz_weights is depraced and replaced by kpt_weights, kpts_basis and kpts will become required to read soon
            additional_things_to_read = ['u_total']
            subgroup_present_additional, self.additional_values_not_read = self.read_input_from_hdf(subgrp=self.dft_data,
                                                                                things_to_read=additional_things_to_read)
        #print(self.hdf_file)
        #print('u_total=')
        #print(self.u_total)
        #print(self.additional_values_not_read)

        #print('number of bath orbital:', nbaths)
        # Generic Hermitian matrix basis for ghostGA
        self.H_list = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.H_list[icrsh][sp] = Hermitian_list(self.nbaths[icrsh])[0]
        # Compute local atomic levels
        self.eff_atomic_levels()
        # non-local part of the hopping matrix
        self.calc_nonlocal_hopping()

    def calc_nonlocal_hopping(self):
        '''
        Compute the non-local part of the hopping term
        '''
        ikarray = np.array(list(range(self.n_k)))
        self.hopping_nloc = np.zeros(self.hopping.shape,dtype=self.hopping.dtype)
        # we need the local potential in the original basis
        self.eloc_orig = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.eloc_orig[icrsh][sp] = cut_small( np.dot( np.dot( self.rot_mat[icrsh], self.Hsumk[icrsh][sp] ),
                                                              self.rot_mat[icrsh].conj().T), tol=1e-8)
                #if self.corr_shells[icrsh]['dim'] == 5:
                #    self.eloc_orig[icrsh][sp] = np.dot( np.dot( self.u_trans_w90_to_std, self.eloc_orig[icrsh][sp] ), self.u_trans_w90_to_std.conj().T)

        if mpi.is_master_node():
            print('eloc_orig=')
            print(self.eloc_orig)
            print('Hsumk=')
            print(self.Hsumk)

        for sp, isp in self.spin_names_to_ind[self.SO].items():
            for ik in mpi.slice_array(ikarray):
                n_orb = self.n_orbitals[ik, isp]
                # u_total wannier90 transformation matrix from bloch to orbital with index [orbital, bloch]
                u = self.u_total[0,ik,:n_orb,:n_orb]
                # rotate to orbital basis
                self.hopping_nloc[ik, isp, :, :] = self.hopping[ik, isp, 0:n_orb, 0:n_orb].copy()
                #self.hopping_nloc[ik, isp, :, :] = np.dot( np.dot( u, self.hopping[ik, isp, 0:n_orb, 0:n_orb]), u.conj().T)
                index = 0
                for icrsh in range(self.n_corr_shells):
                    # local one-body in the original basis: Hsumk has been rotated to local coordinate
                    #eloc_orig = np.dot( np.dot( self.rot_mat[icrsh], self.Hsumk[icrsh][sp] ), self.rot_mat[icrsh].conj().T)
                    #print(eloc_orig)
                    dim = self.corr_shells[icrsh]['dim']
                    #hmat = self.hopping_nloc[ik, isp, index:index+dim,index:index+dim].copy()
                    #self.hopping_nloc[ik, isp, index:index+dim,index:index+dim] = hmat - self.eloc_orig[icrsh][sp]
                    #print(self.hopping_nloc[ik,ind,:,:])
                    projmat = self.proj_mat[ik, isp, icrsh, 0:dim, 0:n_orb]
                    self.hopping_nloc[ik,isp,:,:] -= np.dot( np.dot(projmat.conj().T, self.eloc_orig[icrsh][sp]),projmat)
                    self.hopping_nloc[ik,isp,:,:] = cut_small(self.hopping_nloc[ik,isp,:,:], tol=1e-8)
                    index += dim
                #rotate back to bloch basis
                #self.hopping_nloc[ik, isp, :, :] = np.dot( np.dot( u.conj().T,self.hopping_nloc[ik, isp, 0:n_orb, 0:n_orb]), u)


    def calc_R_Lambda_full(self, R, Lambda, ik, ind, sp):
        '''
        Construct full R and Lambda matrix in the full Wannier90 projection space.
        TODO: this routine will only work for a single-correlated shell. We stil need to generalize
        the code below.
        '''
        n_orb = self.n_orbitals[ik, ind]
        n_qp = np.sum(self.nbaths)
        n_orb_corr = 0
        for icrsh in range(self.n_corr_shells):
            n_orb_corr += self.corr_shells[icrsh]['dim']
        #print(n_orb,n_qp,n_orb_corr)
        R_full = np.zeros((n_qp+n_orb-n_orb_corr,n_orb),dtype=complex)
        n_ncorr = n_orb - n_orb_corr
        R_full[n_qp:,n_orb_corr:] = np.eye(n_ncorr,dtype=complex)
        Lambda_full = np.zeros((n_qp+n_orb-n_orb_corr,n_qp+n_orb-n_orb_corr),dtype=complex)
        indx_phy = 0
        indx_qp = 0
        for icrsh in range(self.n_corr_shells):
            dim_qp = self.nbaths[icrsh]
            dim_phy = self.corr_shells[icrsh]['dim']
            R_full[indx_qp:indx_qp+dim_qp,indx_phy:indx_phy+dim_phy] = R[icrsh][sp]
            Lambda_full[indx_qp:indx_qp+dim_qp,indx_qp:indx_qp+dim_qp] = Lambda[icrsh][sp]
            indx_phy += dim_phy
            indx_qp += dim_qp
        #print(R_full)
        #print(Lambda_full)
        return R_full, Lambda_full

    def calc_rhoks(self, R, Lambda, T, mu=None):
        '''
        density matrix for each momentum.
        '''
        if mu is None:
            mu = self.chemical_potential
        #print('mu=',mu)
        self.rhoks = [{} for icrsh in range(self.n_corr_shells)] # correlated quasiparticle density matrix
        self.rhoks_full = {} # quasiparticle density matrix including correlated and noncorrelated parts
        self.rhoks_phys_bloch = {} # physical density matrix  R^\dagger rho_qp(k) R in the bloch basis
        self.hopping_qp = {} # quasiparticle hopping term without lambda
        n_qp = np.sum(self.nbaths)
        n_orb = self.n_orbitals[0, 0]
        n_orb_corr = 0
        for icrsh in range(self.n_corr_shells):
            n_orb_corr += self.corr_shells[icrsh]['dim']
        assert(np.abs(np.sum(n_orb-self.n_orbitals[:,0]))<1e-8, "assert n_orb are the same for all the k-point and spin")
        ikarray = np.array(list(range(self.n_k)))
        indx_qp = 0
        for icrsh in range(self.n_corr_shells):
            dim_qp = self.nbaths[icrsh]#self.corr_shells[icrsh]['dim']
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.rhoks[icrsh][sp] = np.zeros((self.n_k,Lambda[icrsh][sp].shape[0],
                                           Lambda[icrsh][sp].shape[1]),dtype=complex)
                self.rhoks_full[sp] = np.zeros((self.n_k,n_qp+n_orb-n_orb_corr,n_qp+n_orb-n_orb_corr),dtype=complex)
                self.rhoks_phys_bloch[sp] = np.zeros((self.n_k,self.hopping.shape[2],self.hopping.shape[3]),dtype=complex)
                self.hopping_qp[sp] = np.zeros((self.n_k,n_qp+n_orb-n_orb_corr,n_qp+n_orb-n_orb_corr),dtype=complex)
                ind = self.spin_names_to_ind[self.corr_shells[icrsh]['SO']][sp]
                for ik in mpi.slice_array(ikarray):
                    #print('ik=', ik, 'isp=', isp, 'sp=', sp, self.spin_names_to_ind[self.SO][sp])
                    #print(self.hopping[ik,isp,:,:])
                    R_full, Lambda_full = self.calc_R_Lambda_full(R, Lambda, ik, ind, sp)
                    #print(R_full)
                    #print(Lambda_full)
                    #n_orb = self.n_orbitals[ik, ind]
                    #projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
                    # u_total wannier90 transformation matrix from bloch to orbital with index [orbital, bloch]
                    u = self.u_total[0,ik,:n_orb,:n_orb]
                    MMat = self.hopping_nloc[ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
                    # rotate to orbital basis
                    MMat = np.dot(np.dot(u, MMat), u.conj().T)
                    self.rhoks_full[sp][ik,:,:] = calc_nf(np.dot(R_full, np.dot(MMat, R_full.conj().T ) )
                                                         + Lambda_full - mu*np.eye(Lambda_full.shape[0]) , T ).T
                    self.hopping_qp[sp][ik,:,:] = np.dot(R_full, np.dot(MMat, R_full.conj().T ) ) + Lambda_full
                    self.rhoks_phys_bloch[sp][ik,:,:] = np.dot( np.dot(np.dot( np.dot( u.conj().T, R_full.conj().T),
                                                                         self.rhoks_full[sp][ik,:,:].T), R_full ), u )
                    #print(ik)
                    #assert(np.allclose(np.linalg.eigh(self.hopping[ik,ind,:,:])[0],
                    #                   np.linalg.eigh(np.dot(R_full, np.dot(MMat, R_full.conj().T )) + Lambda_full)[0],atol=1e-5))
                    #assert(np.allclose( np.dot(np.dot(u, self.hopping[ik, ind,:,:]), u.conj().T), 
                    #                    np.dot(R_full, np.dot(MMat, R_full.conj().T )) + Lambda_full, atol=1e-5))
                    #print(self.rhoks_full[sp][ik,:,:])
                    #print(np.dot(np.dot(u.conj().T,calc_nf(self.hopping[ik,ind,:,:],T)),u))
                    #assert(np.allclose(self.rhoks_phys_bloch[sp][ik,:,:],
                    #       #np.allclose(np.dot(np.dot(u.conj().T,self.rhoks_full[sp][ik,:,:].T),u),
                    #       calc_nf(self.hopping[ik,ind,:,:],T),atol=1e-3))
                    #assert(np.allclose(self.rhoks_full[sp][ik,:,:],#self.rhoks_phys_bloch[sp][ik,:,:],
                    #                   np.dot(np.dot(u,calc_nf(self.hopping[ik,ind,:,:],T)),u.conj().T).T,atol=1e-3))
                    self.rhoks[icrsh][sp][ik,:,:] = self.rhoks_full[sp][ik,indx_qp:indx_qp+dim_qp,indx_qp:indx_qp+dim_qp]
            indx_qp += dim_qp

       # mpi reduce:
        for ik in range(self.n_k):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                for icrsh in range(self.n_corr_shells):
                    self.rhoks[icrsh][sp][ik,:,:] = mpi.all_reduce(self.rhoks[icrsh][sp][ik,:,:])
                self.rhoks_full[sp][ik,:,:] = mpi.all_reduce(self.rhoks_full[sp][ik,:,:])


    def calc_Delta(self):
        '''
        The first k-summation for quasiparticle density matrix. NOTE: doesn't work for ghostGA yet.
        '''
        self.Delta = [{} for icrsh in range(self.n_corr_shells)]
        ikarray = np.array(list(range(self.n_k)))
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.Delta[icrsh][sp] = np.zeros((self.rhoks[icrsh][sp].shape[1],
                                                  self.rhoks[icrsh][sp].shape[2]),dtype=complex)
                for ik in mpi.slice_array(ikarray):
                    self.Delta[icrsh][sp][:,:] += self.bz_weights[ik] * self.rhoks[icrsh][sp][ik,:,:]
                self.Delta[icrsh][sp][:,:] = cut_small(self.Delta[icrsh][sp][:,:], tol=1e-8)

        # mpi reduce:
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            for icrsh in range(self.n_corr_shells):
                self.Delta[icrsh][sp][:,:] = mpi.all_reduce(self.Delta[icrsh][sp][:,:])

    def calc_D(self, R, Lambda):
        '''
        The second k-sum for kinetic energy. NOTE: doesn't work fo ghostGA yet.
        '''
        self.D = [{} for icrsh in range(self.n_corr_shells)]
        ikarray = np.array(list(range(self.n_k)))
        indx_qp = 0
        indx_phy = 0
        for icrsh in range(self.n_corr_shells):
            dim_qp = self.nbaths[icrsh]
            dim_phy = self.corr_shells[icrsh]['dim']
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                ind = self.spin_names_to_ind[self.corr_shells[icrsh]['SO']][sp]
                sum_ek_Rdagger_rhoks = np.zeros((dim_phy,dim_qp),dtype=complex)
                for ik in mpi.slice_array(ikarray):
                    R_full, Lambda_full = self.calc_R_Lambda_full(R, Lambda, ik, ind, sp)
                    n_orb = self.n_orbitals[ik, ind]
                    #projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
                    # u_total wannier90 transformation matrix from bloch to orbital with index [orbital, bloch]
                    u = self.u_total[0,ik,:n_orb,:n_orb]
                    MMat = self.hopping_nloc[ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
                    # rotate to orbital basis
                    MMat = np.dot(np.dot(u, MMat), u.conj().T)
                    #MMatproj_nloc = np.dot(np.dot(projmat, MMat), projmat.conjugate().transpose()) - self.Hsumk[icrsh][sp]
                    #sum_ek_Rdagger_rhoks[:,:] += self.bz_weights[ik]*MMatproj_nloc.dot(R[icrsh][sp].conj().T).dot(self.rhoks[icrsh][sp][ik,:,:].T)
                    tmp = self.bz_weights[ik]*np.dot(np.dot(MMat, R_full.conj().T) , self.rhoks_full[sp][ik,:,:].T)
                    sum_ek_Rdagger_rhoks[:,:] += tmp[indx_phy:indx_phy+dim_phy,indx_qp:indx_qp+dim_qp]
                sqrt_Delta=funcMat(self.Delta[icrsh][sp], denR)
                self.D[icrsh][sp] = sqrt_Delta.dot(np.transpose(sum_ek_Rdagger_rhoks))
                self.D[icrsh][sp] = cut_small( self.D[icrsh][sp], tol=1e-8)
            indx_qp += dim_qp
            indx_phy += dim_phy
        # mpi reduce:
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            for icrsh in range(self.n_corr_shells):
                self.D[icrsh][sp][:,:] = mpi.all_reduce(self.D[icrsh][sp])

    def calc_Lambdac(self, R, Lambda):
        '''
        Calculate Lambdac matrix
        '''
        self.Lambdac = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.Lambdac[icrsh][sp] = self.calc_Lambdac_icrsh_isp(R[icrsh][sp], Lambda[icrsh][sp],
                                        self.Delta[icrsh][sp], self.D[icrsh][sp], self.H_list[icrsh][sp])
                self.Lambdac[icrsh][sp] = cut_small(self.Lambdac[icrsh][sp], tol=1e-8)

    def calc_Lambda(self, R, Lambdac):
        '''
        Calculate Lambda matrix
        '''
        Lambda = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                Lambda[icrsh][sp] = self.calc_Lambda_icrsh_isp(R[icrsh][sp], Lambdac[icrsh][sp],
                                        self.Delta[icrsh][sp], self.D[icrsh][sp], self.H_list[icrsh][sp])
                Lambda[icrsh][sp] = cut_small(Lambda[icrsh][sp], tol=1e-8)
        return Lambda

    @staticmethod
    def calc_Lambdac_icrsh_isp(R, Lambda, Delta_p, D, H_list):
        """ Compute Lambda_c matrix for a specific shell icrsh and spin isp
        """
        no = Lambda.shape[0]
        l=inverse_realHcombination(Lambda,H_list)
        lc=numpy.copy(l)*0.0
        MM=numpy.dot(D,numpy.transpose(R))
        for k in range(len(H_list)):
            AA=Delta_p
            HH=H_list[k].T
            derivative=dF(AA,HH, denRm1, ddenRm1)
            tt=numpy.trace(numpy.dot(MM,derivative))
            lc[k]=-l[k]-(tt+numpy.conjugate(tt)).real
        Lambda_c=realHcombination(lc,H_list)
        return Lambda_c

    @staticmethod
    def calc_Lambda_icrsh_isp(R, Lambda_c, Delta_p, D, H_list):
        """ Compute Lambda matrix for a specific shell icrsh and spin isp
        """
        no = Lambda_c.shape[0]
        lc=inverse_realHcombination(Lambda_c,H_list)
        l=numpy.copy(lc)*0.0
        MM=numpy.dot(D,numpy.transpose(R))
        for k in range(len(H_list)):
            AA=Delta_p
            HH=H_list[k].T
            derivative=dF(AA,HH, denRm1, ddenRm1)
            tt=numpy.trace(numpy.dot(MM,derivative))
            l[k]=-lc[k]-(tt+numpy.conjugate(tt)).real
        Lambda=realHcombination(l,H_list)
        return Lambda

    def calc_mu_grisb(self, R, Lambda, precision=0.01, broadening=None, delta=0.5, max_loops=200, method="dichotomy", beta=None):
        r"""
        Searches for the chemical potential that gives the DFT total charge.

        Parameters
        ----------
        precision : float, optional
                    A desired precision of the resulting total charge.
        broadening : float, optional
                     Imaginary shift for the axis along which the real-axis GF is calculated.
                     If not provided, broadening will be set to double of the distance between mesh points in 'mesh'.
                     Only relevant for real-frequency GF.
        max_loops : int, optional
                    Number of dichotomy loops maximally performed.

        method : string, optional
                    Type of optimization used:
                        * dichotomy: usual bisection algorithm from the TRIQS library
                        * newton: newton method, faster convergence but more unstable
                        * brent: finds bounds and proceeds with hyperbolic brent method, a compromise between speed and ensuring convergence
        beta : float, optional, default = broadening
                when using MeshReFreq this determines the temperature for the Fermi function
                smearing when integrating G(w). If not given broadening will be used
                (converted to beta)

        Returns
        -------
        mu : float
             Value of the chemical potential giving the DFT total charge
             within specified precision.

        """
        if beta is None:
            raise NotImplementedError("calc_mu_grisb required beta input.")
        def find_bounds(function, x_init, delta_x, max_loops=1000):
            mpi.report("Finding bounds on chemical potential")
            x = x_init
            # First find the bounds
            y1 = function(x)
            eps = np.sign(y1)
            x1 = x
            x2 = x1
            y2 = y1

            nbre_loop = 0
            # abort the loop after maxiter is reached or when y1 and y2 have different sign
            while (nbre_loop <= max_loops) and (y2*y1) > 0:
                nbre_loop += 1
                x1 = x2
                y1 = y2

                x2 -= eps*delta_x
                y2 = function(x2)

            if nbre_loop > (max_loops):
                raise ValueError("The bounds could not be found")

            # Make sure that x2 > x1
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            mpi.report(f"mu_interval: [  {x1:.4f}  ; {x2:.4f} ]")
            mpi.report(f"delta to target density interval: [ {y1:.4f} ; {y2:.4f} ]")
            return x1, x2

        # previous implementation

        def F_bisection(mu): return self.total_density_grisb(R, Lambda, beta, mu=mu, broadening=broadening).real
        n_qp = np.sum(self.nbaths)
        n_orb = self.n_orbitals[0, 0]
        n_orb_corr = 0
        for icrsh in range(self.n_corr_shells):
            n_orb_corr += self.corr_shells[icrsh]['dim']
        density = self.density_required - self.charge_below + n_qp - n_orb_corr
        mpi.report('density_qp={:.2f}'.format(density))
    
        # using scipy.optimize

        def F_optimize(mu):

            mpi.report("Trying out mu = {}".format(str(mu)))
            calc_dens = self.total_density_grisb(R, Lambda, beta, mu=mu, broadening=broadening).real - density
            mpi.report(f"Target density = {density}; Delta to target = {calc_dens}")
            return calc_dens

        # check for lowercase matching for the method variable
        if method.lower() == "dichotomy":
            mpi.report("\nsumk calc_mu_grisb: Using dichtomy adjustment to find chemical potential\n")
            self.chemical_potential = dichotomy.dichotomy(function=F_bisection,
                                                          x_init=self.chemical_potential, y_value=density,
                                                          precision_on_y=precision, delta_x=delta, max_loops=max_loops,
                                                          x_name="Chemical Potential", y_name="Total Density",
                                                          verbosity=3)[0]
        elif method.lower() == "newton":
            mpi.report("\nsumk calc_mu_grisb: Using Newton method to find chemical potential\n")
            self.chemical_potential = newton(func=F_optimize,
                                             x0=self.chemical_potential,
                                             tol=precision, maxiter=max_loops,
                                             )

        elif method.lower() == "brent":
            mpi.report("\nsumk calc_mu_grisb: Using Brent method to find chemical potential")
            mpi.report("sumk calc_mu_grisb: Finding bounds \n")

            mu_guess_0, mu_guess_1 = find_bounds(function=F_optimize,
                                                 x_init=self.chemical_potential,
                                                 delta_x=delta, max_loops=max_loops,
                                                 )
            mu_guess_1 += 0.01  # scrambles higher lying interval to avoid getting stuck
            mpi.report("\nsumk calc_mu_grisb: Searching root with Brent method\n")
            self.chemical_potential = brenth(f=F_optimize,
                                             a=mu_guess_0,
                                             b=mu_guess_1,
                                             xtol=precision, maxiter=max_loops,
                                             )

        else:
            raise ValueError(
                f"sumk calc_mu_grisb: The selected method: {method}, is not implemented\n",
                """
                    Please check for typos or select one of the following:
                        * dichotomy: usual bisection algorithm from the TRIQS library
                        * newton: newton method, fastest convergence but more unstable
                        * brent: finds bounds and proceeds with hyperbolic brent method, a compromise between speed and ensuring convergence
                    """
            )

        return self.chemical_potential


    def calc_density_correction(self, density_mat_from_emb, observables, E_kin_dft, filename=None, dm_type=None, spinave=False, kpts_to_write=None, broadening=None, beta=None):
        r'''
        Overide the density correction for GRISB
        Calculates the charge density correction and stores it into a file.

        The charge density correction is needed for charge-self-consistent DFT+DMFT calculations.
        It represents a density matrix of the interacting system defined in Bloch basis
        and it is calculated from the sum over Matsubara frequecies of the full GF,

        ..math:: N_{\nu\nu'}(k) = \sum_{i\omega_{n}} G_{\nu\nu'}(k, i\omega_{n})

        The density matrix for every `k`-point is stored into a file.

        Parameters
        ----------
        density_mat_from_emb: dict
                   Embedding density matrix.
        observables: dict
                   Observables.
        E_kin_dft: float
                   kinetic energy from DFT
        filename : string
                   Name of the file to store the charge density correction.
        dm_type : string
                   DFT code to write the density correction for. Options:
                   'vasp', 'wien2k', 'elk' or 'qe'. Needs to be set for 'qe'
        spinave : logical
                   Elk specific and for magnetic calculations in DMFT only.
                   It averages the spin to keep the DFT part non-magnetic.
        kpts_to_write : iterable of int
                   Indices of k points that are written to file. If None (default),
                   all k points are written. Only implemented for dm_type 'vasp'
        broadening : float, optional
                     Imaginary shift for the axis along which the real-axis GF is calculated.
                     If not provided, broadening will be set to double of the distance between mesh points in 'mesh'.
                     Only relevant for real-frequency GF.
        beta : float, optional, default = broadening
                when using MeshReFreq this determines the temperature for the Fermi function
                smearing when integrating G(w). If not given broadening will be used
                (converted to beta)
        Returns
        -------
        (deltaN, dens) : tuple
                         Returns a tuple containing the density matrix `deltaN` and
                         the corresponing total charge `dens`.

        '''
        #automatically set dm_type if required
        if dm_type==None:
            dm_type = self.dft_code

        assert dm_type in ('qe'), "'dm_type' must be 'qe'"
        #default file names
        if filename is None:
            if dm_type == 'vasp':
                filename = 'GAMMA'
            elif dm_type == 'qe':
                filename = self.hdf_file


        assert isinstance(filename, str), ("calc_density_correction: "
                                              "filename has to be a string!")

        assert kpts_to_write is None or dm_type == 'vasp', ('Selecting k-points only'
                                                            +'implemented for vasp')

        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        dens = {sp: 0.0 for sp in spn}
        band_en_correction = 0.0

# Fetch Fermi weights and energy window band indices
        if dm_type in ['vasp','qe']:
            fermi_weights = 0
            band_window = 0
            if mpi.is_master_node():
                with HDFArchive(self.hdf_file,'r') as ar:
                    fermi_weights = ar['dft_misc_input']['dft_fermi_weights']
                    band_window = ar['dft_misc_input']['band_window']
            fermi_weights = mpi.bcast(fermi_weights)
            band_window = mpi.bcast(band_window)

# Convert Fermi weights to a density matrix
            dens_mat_dft = {}
            for sp in spn:
                dens_mat_dft[sp] = [fermi_weights[ik, ntoi[sp], :].astype(complex) for ik in range(self.n_k)]

            #if mpi.is_master_node():
            #    print('dens_mat_dft=')
            #    print(dens_mat_dft['up'][:])

        # Set up deltaN:
        deltaN = {}
        for sp in spn:
            deltaN[sp] = [np.zeros([self.n_orbitals[ik, ntoi[sp]], self.n_orbitals[
                                      ik, ntoi[sp]]], complex) for ik in range(self.n_k)]

        ikarray = np.arange(self.n_k)
        for ik in mpi.slice_array(ikarray):
            #TODO: implement charge correction below using grisb density matrices.
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                n_orb = self.n_orbitals[ik, isp]
                deltaN[sp][ik][:,:] = self.rhoks_phys_bloch[sp][ik,:,:]
                #TODO: the local contribution needs to be replaced using the embedding local density matrix.
                for icrsh in range(self.n_corr_shells):
                    dim = self.corr_shells[icrsh]['dim']
                    #ind = self.spin_names_to_ind[self.corr_shells[icrsh]['SO']][sp]
                    #R_full, Lambda_full = self.calc_R_Lambda_full(observables['R'], observables['Lambda'], ik, ind, sp)
                    projmat = self.proj_mat[ik, isp, icrsh, 0:dim, 0:n_orb]
                    density_mat_from_qp = np.dot( np.dot(observables['R'][icrsh][sp].conj().T ,self.Delta[icrsh][sp] ), observables['R'][icrsh][sp])
                    deltaN[sp][ik][:,:] -= np.dot(np.dot(projmat.conj().T, density_mat_from_qp), projmat)
                    deltaN[sp][ik][:,:] += np.dot(np.dot(projmat.conj().T, density_mat_from_emb[icrsh][sp+'_0']), projmat)
                    #below is used to check U=0 case quasiparticle and embedding density matrix should be identicle
                    #print('icrsh=',icrsh)
                    #print(density_mat_from_qp)
                    #print(density_mat_from_emb[icrsh][sp+'_0'])
                    #assert(np.allclose(density_mat_from_qp,density_mat_from_emb[icrsh][sp+'_0'],atol=1e-5))

                dens[sp] += self.bz_weights[ik] * np.trace(deltaN[sp][ik][:,:])
                if dm_type in ['vasp','qe']:
# In 'vasp'-mode subtract the DFT density matrix
                    nb = self.n_orbitals[ik, ntoi[sp]]
                    diag_inds = np.diag_indices(nb)
                    deltaN[sp][ik][diag_inds] -= dens_mat_dft[sp][ik][:nb]

                    if self.charge_mixing and self.deltaNOld is not None:
                        G2 = np.sum(self.kpts_cart[ik,:]**2)
                        # Kerker mixing
                        mix_fac = self.charge_mixing_alpha * G2 / (G2 + self.charge_mixing_gamma**2)
                        deltaN[sp][ik][diag_inds] = (1.0 - mix_fac) * self.deltaNOld[sp][ik][diag_inds] + mix_fac * deltaN[sp][ik][diag_inds]
                    dens[sp] -= self.bz_weights[ik] * dens_mat_dft[sp][ik].sum().real
                    isp = ntoi[sp]
                    b1, b2 = band_window[isp][ik, :2]
                    nb = b2 - b1 + 1
                    assert nb == self.n_orbitals[ik, ntoi[sp]], "Number of bands is inconsistent at ik = %s"%(ik)
                    # TODO: the band_en_correction needs to be modified
                    #band_en_correction += np.dot(deltaN[sp][ik], self.hopping[ik, isp, :nb, :nb]).trace().real * self.bz_weights[ik]
                    band_en_correction += np.trace(np.dot(self.hopping_qp[sp][ik, :, :], self.rhoks_full[sp][ik,:, :].T))* self.bz_weights[ik]

        # mpi reduce:
        for bname in deltaN:
            for ik in range(self.n_k):
                deltaN[bname][ik] = mpi.all_reduce(deltaN[bname][ik])
            dens[bname] = mpi.all_reduce(dens[bname])
        self.deltaNOld = copy.copy(deltaN)
        mpi.barrier()


        band_en_correction = mpi.all_reduce(band_en_correction)
        band_en_correction = band_en_correction - E_kin_dft

        if mpi.is_master_node():
            print('E_kin_dft=', E_kin_dft)
            #print('rhoks_phys_bloch')
            #print(self.rhoks_phys_bloch[sp][:,:,:].T)
            #for ik in range(self.n_k):
            #    print(deltaN['up'][ik])
            #    print('absmax(deltaN[ik=%d])='%(ik))
            #    print(np.max(np.abs(deltaN['up'][ik])))
            print('absmax(deltaN)=')
            print(np.max(np.abs(deltaN['up'][ik])))
            print('band_en_correction=', band_en_correction)
        #quit()

        # now save to file:
        if dm_type == 'vasp':
            if kpts_to_write is None:
                kpts_to_write = np.arange(self.n_k)
            else:
                assert np.min(kpts_to_write) >= 0 and np.max(kpts_to_write) < self.n_k

            assert self.SP == 0, "Spin-polarized density matrix is not implemented"

            if mpi.is_master_node():
                with open(filename, 'w') as f:
                    f.write(" %i  -1  ! Number of k-points, default number of bands\n"%len(kpts_to_write))
                    for index, ik in enumerate(kpts_to_write):
                        ib1 = band_window[0][ik, 0]
                        ib2 = band_window[0][ik, 1]
                        f.write(" %i  %i  %i\n"%(index + 1, ib1, ib2))
                        for inu in range(self.n_orbitals[ik, 0]):
                            for imu in range(self.n_orbitals[ik, 0]):
                                valre = (deltaN['up'][ik][inu, imu].real + deltaN['down'][ik][inu, imu].real) / 2.0
                                valim = (deltaN['up'][ik][inu, imu].imag + deltaN['down'][ik][inu, imu].imag) / 2.0
                                f.write(" %.14f  %.14f"%(valre, valim))
                            f.write("\n")

        elif dm_type == 'qe':
            if self.SP == 0:
                mpi.report("SUMK calc_density_correction: WARNING! Averaging out spin-polarized correction in the density channel")

            subgrp = 'dft_update'
            delta_N = np.zeros([self.n_k, max(self.n_orbitals[:,0]), max(self.n_orbitals[:,0])], dtype=complex)
            mpi.report(" %i  -1  ! Number of k-points, default number of bands\n"%(self.n_k))
            for ik in range(self.n_k):
                ib1 = band_window[0][ik, 0]
                ib2 = band_window[0][ik, 1]
                for inu in range(self.n_orbitals[ik, 0]):
                    for imu in range(self.n_orbitals[ik, 0]):
                        valre = (deltaN['up'][ik][inu, imu].real + deltaN['down'][ik][inu, imu].real) / 2.0
                        valim = (deltaN['up'][ik][inu, imu].imag + deltaN['down'][ik][inu, imu].imag) / 2.0
                        # write into delta_N
                        delta_N[ik, inu, imu] = valre + 1j*valim
            if mpi.is_master_node():
                with HDFArchive(self.hdf_file, 'a') as ar:
                    if not subgrp in ar:
                        ar.create_group(subgrp)
                    things_to_save = ['delta_N', 'band_en_correction']
                    for it in things_to_save:
                        ar[subgrp][it] = locals()[it]

        else:
            raise NotImplementedError("Unknown density matrix type: '%s'"%(dm_type))

        res = deltaN, dens

        if dm_type in ['vasp', 'qe']:
            res += (band_en_correction,)
        #quit()

        return res


    def total_density_grisb(self, R, Lambda, beta, mu=None, with_Sigma=True, with_dc=True, broadening=None):
        '''
        Compute the total quasiparticle density.
        '''
        self.calc_rhoks(R, Lambda, 1./beta, mu=mu)
        dens = 0.0
        ikarray = np.array(list(range(self.n_k)))
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            for ik in mpi.slice_array(ikarray):
                dens += self.bz_weights[ik] * np.trace(self.rhoks_full[sp][ik,:,:])

        # collect data from mpi:
        dens = mpi.all_reduce(dens)
        mpi.barrier()

        if abs(dens.imag) > 1e-20:
            #print('den=',dens.real)
            mpi.report("Warning: Imaginary part in density will be ignored ({})".format(str(abs(dens.imag))))
        return dens.real

    def lattice_gf_qp(self, R, Lambda, ik, mu=None, broadening=None, mesh=None):
        r"""
        Calculates the Quasiparticle lattice Green function for a given k-point from the DFT Hamiltonian and the self energy.
        Currently only consider a single correlated shell and no ghost orbital.

        Parameters
        ----------
        ik : integer
             k-point index.
        mu : real, optional
             Chemical potential for which the Green's function is to be calculated.
             If not provided, self.chemical_potential is used for mu.
        broadening : real, optional
                     Imaginary shift for the axis along which the real-axis GF is calculated.
                     If not provided, broadening will be set to double of the distance between mesh points in 'mesh'.
        mesh : MeshReFreq or MeshImFreq, optional
                    Mesh to be used if with_Sigma=False. If with Sigma=False and mesh is none then self.mesh is used.

        Returns
        -------
        G_latt_qp : BlockGf
                Quasiparticle Lattice Green's function.

        """
        if mu is None:
            mu = self.chemical_potential
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        if not hasattr(self, "Sigma_imp"):
            with_Sigma = False
        if broadening is None:
            if mesh is None:
                broadening = 0.01
            else:  # broadening = 2 * \Delta omega, where \Delta omega is the spacing of omega points
                broadening = 2.0 * ((mesh.w_max - mesh.w_min) / (len(mesh) - 1))

        # Check if G_latt_qp is present
        set_up_G_latt_qp = False                       # Assume not
        if not hasattr(self, "G_latt_qp" ):
            # Need to create G_latt_(i)w
            set_up_G_latt_qp = True
        else:                                       # Check that existing GF is consistent
            G_latt_qp = self.G_latt_qp
            GFsize = [gf.target_shape[0] for bname, gf in G_latt_qp]
            unchangedsize = all([self.n_orbitals[ik, ntoi[spn[isp]]] == GFsize[
                                isp] for isp in range(self.n_spin_blocks[self.SO])])
            if (not mesh is None) or (not unchangedsize):
                set_up_G_latt_qp = True

        if not mesh is None:
            assert isinstance(mesh, MeshReFreq) or isinstance(mesh, MeshImFreq),  "mesh must be a triqs MeshReFreq or MeshImFreq"
            if isinstance(mesh, MeshImFreq):
                mesh_values = np.linspace(mesh(mesh.first_index()), mesh(mesh.last_index()), len(mesh))
            else:
                mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))
        else:
            mesh = self.mesh
            mesh_values = self.mesh_values

        n_qp = np.sum(self.nbaths)
        n_orb = self.n_orbitals[0, 0]
        n_orb_corr = 0
        for icrsh in range(self.n_corr_shells):
            n_orb_corr += self.corr_shells[icrsh]['dim']

        # Set up G_latt
        if set_up_G_latt_qp:
            #block_structure = [
            #    list(range(self.n_orbitals[ik, ntoi[sp]])) for sp in spn]
            block_structure = [
                list(range(n_qp+n_orb-n_orb_corr)) for sp in spn]
            gf_struct = [(spn[isp], block_structure[isp])
                         for isp in range(self.n_spin_blocks[self.SO])]
            block_ind_list = [block for block, inner in gf_struct]
            if isinstance(mesh, MeshImFreq):
                glist = lambda: [Gf(mesh=mesh, target_shape=[len(inner),len(inner)])
                                 for block, inner in gf_struct]
            else:
                glist = lambda: [Gf(mesh=mesh, target_shape=[len(inner),len(inner)])
                                 for block, inner in gf_struct]
            G_latt_qp = BlockGf(name_list=block_ind_list,
                             block_list=glist(), make_copies=False)
            G_latt_qp.zero()

        #idmat = [np.identity(
        #    self.n_orbitals[ik, ntoi[sp]], complex) for sp in spn]

        #print('mesh:',mesh)
        #print('gf=')
        #print(G_latt_qp['up'].data.shape)
        #quit()

        # fill Glatt
        for ibl, (block, gf) in enumerate(G_latt_qp):
            ind = ntoi[spn[ibl]]
            sp = spn[ibl]
            #n_orb = self.n_orbitals[ik, ind]
            R_full, Lambda_full = self.calc_R_Lambda_full(R, Lambda, ik, ind, sp)
            u = self.u_total[0,ik,:n_orb,:n_orb]
            MMat = self.hopping_nloc[ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
            # rotate to orbital basis
            MMat = np.dot(np.dot(u, MMat), u.conj().T)
            idmat = np.identity( Lambda_full.shape[0],complex)

            if isinstance(mesh, MeshImFreq):
                gf.data[:, :, :] = (idmat * (mesh_values[:, None, None] + mu) #+ self.h_field*(1-2*ibl))
                                    - np.dot(R_full, np.dot(MMat, R_full.conj().T ) )
                                    - Lambda_full )
            else:
                gf.data[:, :, :] = (idmat *
                                    (mesh_values[:, None, None] + mu + 1j*broadening)# + self.h_field*(1-2*ibl)
                                    - np.dot(R_full, np.dot(MMat, R_full.conj().T ) )
                                    - Lambda_full )
            #for icrsh in range(self.n_corr_shells):
            #    dim = self.corr_shells[icrsh]['dim']
            #    MMat = self.hopping[
            #                    ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
            #    projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
            #    #MMatproj_nloc = np.dot(np.dot(projmat, MMat), projmat.conjugate().transpose()) - self.Hsumk[icrsh][sp]
            #    if isinstance(mesh, MeshImFreq):
            #        gf.data[:, :, :] = (idmat[ibl] * (mesh_values[:, None, None] + mu) #+ self.h_field*(1-2*ibl))
            #                            - np.dot(R[icrsh][sp], np.dot(MMatproj_nloc, R[icrsh][sp].conj().T ) )
            #                            - Lambda[icrsh][sp] )
            #    else:
            #        gf.data[:, :, :] = (idmat[ibl] *
            #                            (mesh_values[:, None, None] + mu + 1j*broadening)# + self.h_field*(1-2*ibl)
            #                            - np.dot(R[icrsh][sp], np.dot(MMatproj_nloc, R[icrsh][sp].conj().T ) )
            #                            - Lambda[icrsh][sp] )

        G_latt_qp.invert()
        self.G_latt_qp = G_latt_qp

        return G_latt_qp

    def extract_G_phy(self, R, Lambda, mu=None, broadening=None, mesh=None, show_warnings=True):
        r"""
        Extracts the local downfolded Green function by the Brillouin-zone integration of the lattice Green's function.
        Currently only consider a single correlated shell and no ghost orbital.

        Parameters
        ----------
        mu : real, optional
            Input chemical potential. If not provided the value of self.chemical_potential is used as mu.
        broadening : float, optional
            Imaginary shift for the axis along which the real-axis GF is calculated.
            If not provided, broadening will be set to double of the distance between mesh points in 'mesh'.
            Only relevant for real-frequency GF.
        show_warnings : bool, optional
            Displays warning messages during transformation
            (Only effective if transform_to_solver_blocks = True

        Returns
        -------
        G_loc : list of BlockGf (Green's function) objects
            List of the local Green's functions for all (inequivalent) correlated shells,
            rotated into the corresponding local frames.
            If ``transform_to_solver_blocks`` is True, it will be one per inequivalent correlated shell, else one per
            correlated shell.
        """

        if mu is None:
            mu = self.chemical_potential

        if mesh is None:
            mesh = self.mesh

        # create G_loc to be returned in sumk space for all correlated shells. Trafo to solver block structure done later
        #G_loc = [self.block_structure.create_gf(ish=ish, mesh=mesh, space='sumk') for ish in range(self.n_corr_shells)]
        #print(G_loc[0]['up'].data.shape)
        # create G_loc in the Wannierized space including correlated and non-correlated orbitals
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        # n_orbital doesn't depend on ik when using wannier90
        block_structure = [list(range(self.n_orbitals[0, ntoi[sp]])) for sp in spn]
        gf_struct = [(spn[isp], block_structure[isp])
                     for isp in range(self.n_spin_blocks[self.SO])]
        block_ind_list = [block for block, inner in gf_struct]
        if isinstance(mesh, MeshImFreq):
            glist = lambda: [Gf(mesh=mesh, target_shape=[len(inner),len(inner)])
                             for block, inner in gf_struct]
        else:
            glist = lambda: [Gf(mesh=mesh, target_shape=[len(inner),len(inner)])
                             for block, inner in gf_struct]
        G_loc = BlockGf(name_list=block_ind_list,
                            block_list=glist(), make_copies=False)
        G_loc.zero()


        ikarray = np.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            if isinstance(mesh, MeshImFreq):
                G_latt_qp = self.lattice_gf_qp( R, Lambda, ik=ik, mu=mu)
            else:
                G_latt_qp = self.lattice_gf_qp( R, Lambda, ik=ik, mu=mu, broadening=broadening, mesh=mesh)
            G_latt_qp *= self.bz_weights[ik]

            #n_orb = self.n_orbitals[ik, ind]
            for bname, gf in G_loc:
                #print('bname=',bname)
                #print(G_latt_qp[bname].data.shape)
                ind = ntoi[bname]
                R_full, Lambda_full = self.calc_R_Lambda_full(R, Lambda, ik, ind, bname)
                #sp = spn[ibl]
                gf.data[:,:,:] += np.einsum('ij,kjl,lm->kim', R_full.conj().T, G_latt_qp[bname].data, R_full)
            #for icrsh in range(self.n_corr_shells):
            #    # init temporary storage
            #    for bname, gf in G_loc[icrsh]:
            #        #print(G_latt_qp[bname].data.shape)
            #        gf.data[:,:,:] += np.einsum('ij,kjl,lm->kim', R[icrsh][bname].conj().T, G_latt_qp[bname].data, R[icrsh][bname])

        # Collect data from mpi
        G_loc << mpi.all_reduce(G_loc)
        #for icrsh in range(self.n_corr_shells):
        #    G_loc[icrsh] << mpi.all_reduce(G_loc[icrsh])
        mpi.barrier()

        return G_loc

