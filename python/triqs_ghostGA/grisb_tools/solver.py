import numpy as np
from itertools import product

from triqs.gf import MeshImTime, MeshReTime, MeshReFreq, MeshLegendre, Gf, BlockGf, make_hermitian, Omega, iOmega_n, make_gf_from_fourier, fit_hermitian_tail
from triqs.gf.tools import inverse, make_zero_tail
from triqs.gf.descriptors import Fourier
from triqs.operators import c_dag, c, Operator
import triqs.utility.mpi as mpi
from h5 import HDFArchive

def get_n_orbitals(sum_k):
    """
    determines the number of orbitals within the
    solver block structure.

    Parameters
    ----------
    sum_k : dft_tools sumk object

    Returns
    -------
    n_orb : dict of int
        number of orbitals for up / down as dict for SOC calculation
        without up / down block up holds the number of orbitals
    """
    n_orbitals = [{'up': 0, 'down': 0} for i in range(sum_k.n_inequiv_shells)]
    for icrsh in range(sum_k.n_inequiv_shells):
        for block, n_orb in sum_k.gf_struct_solver[icrsh].items():
            if 'down' in block:
                n_orbitals[icrsh]['down'] += sum_k.gf_struct_solver[icrsh][block]
            else:
                n_orbitals[icrsh]['up'] += sum_k.gf_struct_solver[icrsh][block]

    return n_orbitals

class SolverStructure:

    r'''
    Handles all solid_dmft solver objects and contains TRIQS solver instance.

    Attributes
    ----------

    Methods
    -------
    solve(self, **kwargs)
        solve impurity problem
    '''

    def __init__(self, general_params, solver_params, advanced_params, sum_k, icrsh, h_int, iteration_offset, solver_struct_ftps):
        r'''
        Initialisation of the solver instance with h_int for impurity "icrsh" based on soliDMFT parameters.

        Parameters
        ----------
        general_paramuters: dict
                           general parameters as dict
        solver_params: dict
                           solver-specific parameters as dict
        sum_k: triqs.dft_tools.sumk object
               SumkDFT instance
        icrsh: int
               correlated shell index
        h_int: triqs.operator object
               interaction Hamiltonian of correlated shell
        iteration_offset: int
               number of iterations this run is based on
        '''

        self.general_params = general_params
        self.solver_params = solver_params
        self.advanced_params = advanced_params
        self.sum_k = sum_k
        self.icrsh = icrsh
        self.h_int = h_int
        self.iteration_offset = iteration_offset
        self.solver_struct_ftps = solver_struct_ftps
        self.nimp = self.sum_k.eloc_orig[self.icrsh]['up'].shape[0]
        self.nbath = self.general_params['norb_baths'][self.icrsh]
        # initialize density matrix as zeros
        self.density_matrix = np.zeros((2*(self.nimp+self.nbath),2*(self.nimp+self.nbath)),complex)
        # currently no solver requires random number
        #if solver_params.get("random_seed") is None:
        #    self.random_seed_generator = None

        if self.general_params['solver_type'] == 'fci':
            # sets up necessary GF objects on ImFreq. we are not using it yet.
            self.gf_struct = self.sum_k.gf_struct_solver_list[self.icrsh]
            self._init_ImFreq_objects()
            self._init_ReFreq_hartree()

            # sets up solver
            self.triqs_solver = self._create_fci_solver()
            #self.git_hash = triqs_hubbardI_hash
            #self.version = version

        elif self.general_params['solver_type'] == 'pyscf_dmrg':
            self.gf_struct = self.sum_k.gf_struct_solver_list[self.icrsh]
            self._init_ImFreq_objects()
            self._init_ReFreq_hartree()

            #set up solver
            self.triqs_solver = self._create_pyscf_dmrg_solver()

        elif self.general_params['solver_type'] == 'pyscf_ccsd':
            self.gf_struct = self.sum_k.gf_struct_solver_list[self.icrsh]
            self._init_ImFreq_objects()
            self._init_ReFreq_hartree()

            #set up solver
            self.triqs_solver = self._create_pyscf_ccsd_solver()

        elif self.general_params['solver_type'] == 'block2_dmrg':
            raise NotImplementedError("block2 DMRG solver not implemeted!")
            # sets up necessary GF objects on ImFreq. we are not using it yet.
            #self._init_ImFreq_objects()
            #self._init_ReFreq_objects()
            # sets up solver
            #self.triqs_solver = self._create_block2_solver()
            #self.git_hash = triqs_hartree_fock_hash
            #self.version = version

    # ********************************************************************
    # solver-specific solve() command
    # ********************************************************************

    def solve(self, **kwargs):
        r'''
        solve impurity problem with current solver
        '''

        # No solver requires random number
        #if self.random_seed_generator is None:
        #    random_seed = {}
        #else:
        #    random_seed = { "random_seed": int(self.random_seed_generator(it=kwargs["it"], rank=mpi.rank)) }

        if self.general_params['solver_type'] == 'fci':

            mpi.report('\n Using the full configuration interaction solver.')

            # Solve the impurity problem for icrsh shell
            # construct single particle matrix
            eloc_spinful = np.zeros((2*self.nimp,2*self.nimp),dtype=complex)
            D_spinful = np.zeros((2*self.nbath,2*self.nimp),dtype=complex)
            Lambdac_spinful = np.zeros((2*self.nbath,2*self.nbath),dtype=complex)
            # Sz symmetry assumed
            # put Vdc by hand. There's a minus sign difference from sumk_dft.calc_dc
            #nnom = 1
            #Vdc = -(self.general_params['U'][self.icrsh]+(self.nimp-1)*(self.general_params['U'][self.icrsh]
            #        -2*self.general_params['J'][self.icrsh])+(self.nimp-1)*(self.general_params['U'][self.icrsh]
            #        -3*self.general_params['J'][self.icrsh]))*(nnom-0.5)/(2*self.nimp-1)
            #print('Vdc=', Vdc)
            #print(self.sum_k.dc_imp[self.icrsh]['up'])
            #print(self.sum_k.dc_imp[self.icrsh]['down'])
            eloc_spinful[::2,::2]= self.sum_k.eloc_orig[self.icrsh]['up'] - self.sum_k.dc_imp[self.icrsh]['up']
            eloc_spinful[1::2,1::2]= self.sum_k.eloc_orig[self.icrsh]['down'] - self.sum_k.dc_imp[self.icrsh]['down']
            D_spinful[::2,::2]= self.sum_k.D[self.icrsh]['up']
            D_spinful[1::2,1::2]= self.sum_k.D[self.icrsh]['down']
            Lambdac_spinful[::2,::2]= self.sum_k.Lambdac[self.icrsh]['up']
            Lambdac_spinful[1::2,1::2]= self.sum_k.Lambdac[self.icrsh]['down']
            self.triqs_solver.build_h1e(eloc_spinful, D_spinful, Lambdac_spinful, 0.0)
            #print('h1e_up=')
            #print(self.triqs_solver.h1e[::2,::2])
            #print('h1e_down=')
            #print(self.triqs_solver.h1e[1::2,1::2])
            #print('h_int=')
            #print(self.h_int)
            self.triqs_solver.build_Hemb_for_grisb_cycle(self.h_int, spin_pen=0.05)
            self.triqs_solver.solve_Hemb()
            self.density_matrix = self.triqs_solver.calc_density_matrix()
            #print('density_matrix_up=')
            #print(self.density_matrix[::2,::2])
            #print('density_matrix_down=')
            #print(self.density_matrix[1::2,1::2])
            self.E1loc = self.triqs_solver.compute_E1loc(2*self.nimp)
            self.E2loc = self.triqs_solver.compute_E2loc()
            #quit()

            # call postprocessing
            #self._fci_postprocessing()

        elif self.general_params['solver_type'] == 'pyscf_dmrg':

            mpi.report('\n Using the pyscf dmrg solver.')

            # Solve the impurity problem for icrsh shell
            # construct single particle matrix
            eloc_spinful = np.zeros((2*self.nimp,2*self.nimp),dtype=complex)
            D_spinful = np.zeros((2*self.nbath,2*self.nimp),dtype=complex)
            Lambdac_spinful = np.zeros((2*self.nbath,2*self.nbath),dtype=complex)
            # Sz symmetry assumed
            # put Vdc by hand. There's a minus sign difference from sumk_dft.calc_dc
            #nnom = 1
            #Vdc = -(self.general_params['U'][self.icrsh]+(self.nimp-1)*(self.general_params['U'][self.icrsh]
            #        -2*self.general_params['J'][self.icrsh])+(self.nimp-1)*(self.general_params['U'][self.icrsh]
            #        -3*self.general_params['J'][self.icrsh]))*(nnom-0.5)/(2*self.nimp-1)
            #print('Vdc=', Vdc)
            #print(self.sum_k.dc_imp[self.icrsh]['up'])
            #print(self.sum_k.dc_imp[self.icrsh]['down'])
            eloc_spinful[::2,::2]= self.sum_k.eloc_orig[self.icrsh]['up'] - self.sum_k.dc_imp[self.icrsh]['up']
            eloc_spinful[1::2,1::2]= self.sum_k.eloc_orig[self.icrsh]['down'] - self.sum_k.dc_imp[self.icrsh]['down']
            D_spinful[::2,::2]= self.sum_k.D[self.icrsh]['up']
            D_spinful[1::2,1::2]= self.sum_k.D[self.icrsh]['down']
            Lambdac_spinful[::2,::2]= self.sum_k.Lambdac[self.icrsh]['up']
            Lambdac_spinful[1::2,1::2]= self.sum_k.Lambdac[self.icrsh]['down']

            self.triqs_solver.build_Hemb(D_spinful, eloc_spinful, Lambdac_spinful, self.h_int, spin_pen=0.05)
            self.triqs_solver.solve_Hemb(num_eig=2, verbose=False )
            self.density_matrix = self.triqs_solver.calc_density_matrix()
            self.E1loc = self.triqs_solver.compute_E1loc(2*self.nimp)
            self.E2loc = self.triqs_solver.compute_E2loc()

        elif self.general_params['solver_type'] == 'pyscf_ccsd':

            mpi.report('\n Using the pyscf ccsd solver.')

            # Solve the impurity problem for icrsh shell
            # construct single particle matrix
            eloc_spinful = np.zeros((2*self.nimp,2*self.nimp),dtype=complex)
            D_spinful = np.zeros((2*self.nbath,2*self.nimp),dtype=complex)
            Lambdac_spinful = np.zeros((2*self.nbath,2*self.nbath),dtype=complex)
            # Sz symmetry assumed
            # put Vdc by hand. There's a minus sign difference from sumk_dft.calc_dc
            #nnom = 1
            #Vdc = -(self.general_params['U'][self.icrsh]+(self.nimp-1)*(self.general_params['U'][self.icrsh]
            #        -2*self.general_params['J'][self.icrsh])+(self.nimp-1)*(self.general_params['U'][self.icrsh]
            #        -3*self.general_params['J'][self.icrsh]))*(nnom-0.5)/(2*self.nimp-1)
            #print('Vdc=', Vdc)
            #print(self.sum_k.dc_imp[self.icrsh]['up'])
            #print(self.sum_k.dc_imp[self.icrsh]['down'])
            eloc_spinful[::2,::2]= self.sum_k.eloc_orig[self.icrsh]['up'] - self.sum_k.dc_imp[self.icrsh]['up']
            eloc_spinful[1::2,1::2]= self.sum_k.eloc_orig[self.icrsh]['down'] - self.sum_k.dc_imp[self.icrsh]['down']
            D_spinful[::2,::2]= self.sum_k.D[self.icrsh]['up']
            D_spinful[1::2,1::2]= self.sum_k.D[self.icrsh]['down']
            Lambdac_spinful[::2,::2]= self.sum_k.Lambdac[self.icrsh]['up']
            Lambdac_spinful[1::2,1::2]= self.sum_k.Lambdac[self.icrsh]['down']

            self.triqs_solver.build_Hemb(D_spinful, eloc_spinful, Lambdac_spinful, self.h_int, spin_pen=0.05)
            self.triqs_solver.solve_Hemb(num_eig=2, verbose=False, restrict=self.solver_params['restricted'])
            self.density_matrix = self.triqs_solver.calc_density_matrix()
            self.E1loc = self.triqs_solver.compute_E1loc(2*self.nimp)
            self.E2loc = self.triqs_solver.compute_E2loc()

        elif self.general_params['solver_type'] == 'block2_dmrg':
            raise NotImplementedError("block2 DMRG solver not implemeted!")
            # Solve the impurity problem for icrsh shell
            # *************************************
            # this is done on every node due to very slow bcast of the AtomDiag object as of now
            #self.triqs_solver.solve(h_int=self.h_int, calc_gtau=self.solver_params['measure_G_tau'],
            #                        calc_gw=True, calc_gl=self.solver_params['measure_G_l'],
            #                        calc_dm=self.solver_params['measure_density_matrix'])
            # if density matrix is measured, get this too. Needs to be done here,
            # because solver property 'dm' is not initialized/broadcastable
            #if self.solver_params['measure_density_matrix']:
            #    self.density_matrix = self.triqs_solver.dm
            #    self.h_loc_diagonalization = self.triqs_solver.ad
            # *************************************

            # call postprocessing
            #self._block2_postprocessing()

        return

    # ********************************************************************
    # create solvers objects
    # ********************************************************************

    def _create_fci_solver(self):
        r'''
        Initialize configulration interaction exact-diagonalization solver instance
        '''
        from triqs_ghostGA.solvers.ci import CI
        triqs_solver = CI(2*(self.general_params['norb_baths'][self.icrsh]
                             +self.sum_k.corr_shells[self.icrsh]['dim']), use_Ntot=True, use_Sz=True, dtype=np.complex128)

        return triqs_solver

    def _create_pyscf_dmrg_solver(self):
        r'''
        Initial pyscf dmrgscf solver instance
        '''
        import os
        from pyscf import dmrgscf
        from triqs_ghostGA.pyscf_solvers import Pyscf_dmrg
        dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
        dmrgscf.settings.MPIPREFIX = ''#mpirun -n 1 --bind-to none'

        triqs_solver = Pyscf_dmrg(2*(self.general_params['norb_baths'][self.icrsh]
                            +self.sum_k.corr_shells[self.icrsh]['dim']), 2*self.sum_k.corr_shells[self.icrsh]['dim']
                            , 2*self.general_params['norb_baths'][self.icrsh], self.solver_params["maxM"])

        return triqs_solver


    def _create_pyscf_ccsd_solver(self):
        r'''
        Initial pyscf dmrgscf solver instance
        '''

        from triqs_ghostGA.pyscf_solvers import Pyscf_ccsd

        triqs_solver = Pyscf_ccsd(2*(self.general_params['norb_baths'][self.icrsh]
                            +self.sum_k.corr_shells[self.icrsh]['dim']), 2*self.sum_k.corr_shells[self.icrsh]['dim']
                            , 2*self.general_params['norb_baths'][self.icrsh])

        return triqs_solver

    def _create_block2_solver(self):
        r'''
        Initialize block2 dmrg solver instance
        '''
        from triqs_cthyb.solver import Solver as cthyb_solver
        raise NotImplementedError("block2 DMRG solver not implemeted!")
        return

    #def _make_spin_equal(self, Sigma):
    #
    #    # if not SOC than average up and down
    #    if not self.general_params['magnetic'] and not self.sum_k.SO == 1:
    #        Sigma['up_0'] = 0.5*(Sigma['up_0'] + Sigma['down_0'])
    #        Sigma['down_0'] = Sigma['up_0']
    #
    #    return Sigma

    # ********************************************************************
    # post-processing of solver output
    # ********************************************************************

    def _fci_postprocessing(self):
        r'''
        Organize G_freq, G_time, Sigma_freq and G_l from hubbardI solver
        '''

        # get everything from solver
        #self.Sigma_freq << self.triqs_solver.Sigma_iw
        #self.G0_freq << self.triqs_solver.G0_iw
        #self.G0_Refreq << self.triqs_solver.G0_w
        #self.G_freq << make_hermitian(self.triqs_solver.G_iw)
        #self.G_freq_unsym << self.triqs_solver.G_iw
        #self.sum_k.symm_deg_gf(self.G_freq, ish=self.icrsh)
        #self.G_freq << self.G_freq
        #self.G_Refreq << self.triqs_solver.G_w
        #self.Sigma_Refreq << self.triqs_solver.Sigma_w

        return

    # ********************************************************************
    # initialize Freq and Time objects
    # ********************************************************************

    def _init_ImFreq_objects(self):
        r'''
        Initialize all ImFreq objects
        '''

        # create all ImFreq instances
        self.n_iw = self.general_params['n_iw']
        self.G_freq = self.sum_k.block_structure.create_gf(ish=self.icrsh, gf_function=Gf, space='solver',
                                                           mesh=self.sum_k.mesh)
        # copy
        self.Sigma_freq = self.G_freq.copy()
        self.G0_freq = self.G_freq.copy()
        self.G_freq_unsym = self.G_freq.copy()
        self.Delta_freq = self.G_freq.copy()

        # create all ImTime instances
        self.n_tau = self.general_params['n_tau']
        self.G_time = self.sum_k.block_structure.create_gf(ish=self.icrsh, gf_function=Gf, space='solver',
                                                           mesh=MeshImTime(beta=self.general_params['beta'],
                                                                           S='Fermion', n_tau=self.n_tau)
                                                           )
        # copy
        self.Delta_time = self.G_time.copy()

        # create all Legendre instances
        if (self.general_params['solver_type'] == 'cthyb' and self.solver_params['measure_G_l']
            or self.general_params['solver_type'] == 'cthyb' and  self.general_params['legendre_fit']
            or self.general_params['solver_type'] == 'ctseg' and self.solver_params['measure_gl']
            or self.general_params['solver_type'] == 'ctseg' and  self.general_params['legendre_fit']
            or self.general_params['solver_type'] == 'hubbardI' and self.solver_params['measure_G_l']):

            self.n_l = self.general_params['n_l']
            self.G_l = self.sum_k.block_structure.create_gf(ish=self.icrsh, gf_function=Gf, space='solver',
                                                            mesh=MeshLegendre(beta=self.general_params['beta'],
                                                                              max_n=self.n_l, S='Fermion')
                                                            )
            # move original G_freq to G_freq_orig
            self.G_time_orig = self.G_time.copy()

        if self.general_params['solver_type'] in ['cthyb', 'hubbardI'] and self.solver_params['measure_density_matrix']:
            self.density_matrix = None
            self.h_loc_diagonalization = None

        if self.general_params['solver_type'] in ['cthyb'] and self.general_params['measure_chi'] != 'none':
            self.O_time = None

    def _init_ReFreq_objects(self):
        r'''
        Initialize all ReFreq objects
        '''

        # create all ReFreq instances
        self.n_w = self.general_params['n_w']
        self.G_freq = self.sum_k.block_structure.create_gf(ish=self.icrsh, gf_function=Gf, space='solver',
                                                           mesh=self.sum_k.mesh)
        # copy
        self.Sigma_freq = self.G_freq.copy()
        self.G0_freq = self.G_freq.copy()
        self.Delta_freq = self.G_freq.copy()
        self.G_freq_unsym = self.G_freq.copy()

        # create another Delta_freq for the solver, which uses different spin indices
        n_orb = self.sum_k.corr_shells[self.icrsh]['dim']
        n_orb = n_orb//2 if self.sum_k.corr_shells[self.icrsh]['SO'] else n_orb
        gf = Gf(target_shape = (n_orb, n_orb), mesh=MeshReFreq(n_w=self.n_w, window=self.general_params['w_range']))

        self.Delta_freq_solver = BlockGf(name_list =tuple([block[0] for block in self.gf_struct]), block_list = (gf, gf), make_copies = True)

        # create all ReTime instances
        # FIXME: dummy G_time, since time_steps will be recalculated during run
        #time_steps = int(2 * self.solver_params['time_steps'] * self.solver_params['refine_factor']) if self.solver_params['n_bath'] != 0 else int(2 * self.solver_params['time_steps'])
        time_steps = int(2 * 1 * self.solver_params['refine_factor']) if self.solver_params['n_bath'] != 0 else int(2 * 1)
        self.G_time = self.sum_k.block_structure.create_gf(ish=self.icrsh, gf_function=Gf, space='solver',
                                                           mesh=MeshReTime(n_t=time_steps+1,
                                                           window=[0,time_steps*self.solver_params['dt']])
                                                           )

    def _init_ReFreq_hartree(self):
        r'''
        Initialize all ReFreq objects
        '''

        # create all ReFreq instances
        self.n_w = self.general_params['n_w']
        self.Sigma_Refreq = self.sum_k.block_structure.create_gf(ish=self.icrsh, gf_function=Gf, space='solver',
                                                                 mesh=MeshReFreq(n_w=self.n_w, window=self.general_params['w_range'])
                                                                 )
