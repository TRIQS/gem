import numpy as np
from itertools import product

from triqs.gf import MeshImTime, MeshReTime, MeshReFreq, MeshLegendre, Gf, BlockGf, make_hermitian, Omega, iOmega_n, make_gf_from_fourier, fit_hermitian_tail
from triqs.gf.tools import inverse, make_zero_tail
from triqs.gf.descriptors import Fourier
from triqs.operators import c_dag, c, Operator
import triqs.utility.mpi as mpi
from h5 import HDFArchive

from solid_dmft.dmft_tools.solver import SolverStructure as SStructure

# TODO: Design this class. The solve() method is very different from dmft, so it would make sense to make a ghostGA_SolverStructure class
# in parallel, but also we could just change the solve() method. That would make sense if the same solvers are connected to both classes I believe,
# but at the same time the DMFT class relies heavily on Green's function.

class SolverStructure(SStructure):

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
            # TODO: Do we really need that?
            # self._init_ImFreq_objects()
            # self._init_ReFreq_hartree()

            # sets up solver
            self.triqs_solver = self._create_fci_solver()
            #self.git_hash = triqs_hubbardI_hash
            #self.version = version

        elif self.general_params['solver_type'] == 'pyscf_dmrg':
            self.gf_struct = self.sum_k.gf_struct_solver_list[self.icrsh]
            # self._init_ImFreq_objects()
            # self._init_ReFreq_hartree()

            #set up solver
            self.triqs_solver = self._create_pyscf_dmrg_solver()

        elif self.general_params['solver_type'] == 'pyscf_ccsd':
            self.gf_struct = self.sum_k.gf_struct_solver_list[self.icrsh]
            # self._init_ImFreq_objects()
            # self._init_ReFreq_hartree()

            #set up solver
            self.triqs_solver = self._create_pyscf_ccsd_solver()

        elif self.general_params['solver_type'] == 'block2_dmrg':
            raise NotImplementedError("block2 DMRG solver not implemeted!")

    # ********************************************************************
    # solver-specific solve() command
    # ********************************************************************

    def solve(self, **kwargs):
        r'''
        solve impurity problem with current solver
        '''

        if self.general_params['solver_type'] == 'fci':

            mpi.report('\n Using the full configuration interaction solver.')

        elif self.general_params['solver_type'] == 'pyscf_dmrg':

            mpi.report('\n Using the pyscf dmrg solver.')

        elif self.general_params['solver_type'] == 'pyscf_ccsd':

            mpi.report('\n Using the pyscf ccsd solver.')

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
        from triqs_ghostGA.solvers.pyscf_solvers import Pyscf_dmrg
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

        from triqs_ghostGA.solvers.pyscf_solvers import Pyscf_ccsd

        triqs_solver = Pyscf_ccsd(2*(self.general_params['norb_baths'][self.icrsh]
                            +self.sum_k.corr_shells[self.icrsh]['dim']), 2*self.sum_k.corr_shells[self.icrsh]['dim']
                            , 2*self.general_params['norb_baths'][self.icrsh])

        return triqs_solver

    def _create_block2_solver(self):
        r'''
        Initialize block2 dmrg solver instance
        '''
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
        '''

        return
