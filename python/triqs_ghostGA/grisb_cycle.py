"""
main grisb cycle, grisb step, and helper functions copied from solid_dmft
"""


# system
import os
from copy import deepcopy
from timeit import default_timer as timer
import numpy as np

# triqs
from triqs.operators.util.observables import S_op, N_op
from triqs.version import git_hash as triqs_hash
from triqs.version import version as triqs_version
from h5 import HDFArchive
import triqs.utility.mpi as mpi
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from triqs.gf.tools import inverse
#from triqs_dft_tools.sumk_dft import SumkDFT

# ghostGA
from triqs_ghostGA.sumk_grisb import SumkGRISB
from triqs_ghostGA.utility.utils_TH import funcMat, denR, cut_small
from triqs_ghostGA.grisb_tools.observables import (calc_dft_kin_en, add_grisb_observables, calc_bandcorr_man, write_obs,
                                         add_dft_values_as_zeroth_iteration, write_header_to_file, prep_observables)
from triqs_ghostGA.grisb_tools.solver import SolverStructure
from triqs_ghostGA.grisb_tools import interaction_hamiltonian
from triqs_ghostGA.grisb_tools import results_to_archive
from triqs_ghostGA.grisb_tools import initial_self_energies as initial_sigma
from triqs_ghostGA.grisb_tools import convergence
from triqs_ghostGA.grisb_tools import formatter

# own modules
from solid_dmft.version import solid_dmft_hash
from solid_dmft.version import version as solid_dmft_version
from solid_dmft.dmft_tools import afm_mapping
from solid_dmft.dmft_tools import manipulate_chemical_potential as manipulate_mu
from solid_dmft.dmft_tools import greens_functions_mixer as gf_mixer


def _determine_block_structure(sum_k, general_params, advanced_params):
    """
    Determines block structrure and degenerate deg_shells
    computes first DFT density matrix to determine block structure and changes
    the density matrix according to needs i.e. magnetic calculations, or keep
    off-diag elements

    Parameters
    ----------
    sum_k : SumK Object instances

    Returns
    -------
    sum_k : SumK Object instances
        updated sum_k Object
    """
    mpi.report('\n *** determination of block structure ***')

    # this returns a list of dicts (one entry for each corr shell)
    # the dict contains one entry for up and one for down
    # each entry is a square complex numpy matrix with dim=corr_shell['dim']
    zero_Sigma = [sum_k.block_structure.create_gf(ish=iineq, gf_function=Gf, mesh=sum_k.mesh)
                  for iineq in range(sum_k.n_inequiv_shells)]
    sum_k.put_Sigma(zero_Sigma)
    if general_params['solver_type'] in ['ftps']:
        G_loc_all = sum_k.extract_G_loc(broadening=general_params['eta'], transform_to_solver_blocks=False)
        dens_mat = [G_loc_all[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]
    else:
        dens_mat = sum_k.density_matrix(method='using_gf')

    original_dens_mat = deepcopy(dens_mat)

    # for certain systems it is needed to keep off diag elements
    # this enforces to use the full corr subspace matrix
    solver_struct_ftps = None
    if general_params['enforce_off_diag'] or general_params['solver_type'] in ['ftps', 'hartree']:
        if general_params['solver_type'] in ['ftps']:
            # first round to determine real blockstructure
            mock_sumk = deepcopy(sum_k)
            mock_sumk.analyse_block_structure(dm=dens_mat, threshold=general_params['block_threshold'])
            solver_struct_ftps = [None] * sum_k.n_inequiv_shells
            for icrsh in range(sum_k.n_inequiv_shells):
                solver_struct_ftps[icrsh] = mock_sumk.deg_shells[icrsh]
            mpi.report('Block structure written to "solver_struct_ftps":')
            mpi.report(solver_struct_ftps)

        mpi.report('enforcing off-diagonal elements in block structure finder')
        for dens_mat_per_imp in dens_mat:
            for dens_mat_per_block in dens_mat_per_imp.values():
                dens_mat_per_block += 2 * general_params['block_threshold']

    if not general_params['enforce_off_diag'] and general_params['block_suppress_orbital_symm']:
        mpi.report('removing orbital symmetries in block structure finder')
        for dens_mat_per_imp in dens_mat:
            for dens_mat_per_block in dens_mat_per_imp.values():
                dens_mat_per_block += 2*np.diag(np.arange(dens_mat_per_block.shape[0]))

    mpi.report('using 1-particle density matrix and Hloc (atomic levels) to '
               'determine the block structure')
    sum_k.analyse_block_structure(dm=dens_mat, threshold=general_params['block_threshold'])

    if advanced_params['pick_solver_struct'] != 'none':
        mpi.report('selecting subset of orbital space for gf_struct_solver from input:')
        mpi.report(advanced_params['pick_solver_struct'])
        sum_k.block_structure.pick_gf_struct_solver(advanced_params['pick_solver_struct'])

    # Applies the manual mapping to each inequivalent shell
    if advanced_params['map_solver_struct'] != 'none':
        sum_k.block_structure.map_gf_struct_solver(advanced_params['map_solver_struct'])
        if advanced_params['mapped_solver_struct_degeneracies'] != 'none':
            sum_k.block_structure.deg_shells = advanced_params['mapped_solver_struct_degeneracies']

    # if we want to do a magnetic calculation we need to lift up/down degeneracy
    if general_params['magnetic'] and sum_k.SO == 0:
        mpi.report('magnetic calculation: removing the spin degeneracy from the block structure')

        for icrsh, deg_shells_site in enumerate(sum_k.block_structure.deg_shells):
            deg_shell_mag = []
            # find degenerate orbitals that do not simply connect different spin channels
            for deg_orbs in deg_shells_site:
                for spin in ['up', 'down']:
                    # create a list of all up / down orbitals
                    deg = [orb for orb in deg_orbs if spin in orb]
                    # if longer than one than we have two deg orbitals also in a magnetic calculation
                    if len(deg) > 1:
                        deg_shell_mag.append(deg)
            sum_k.block_structure.deg_shells[icrsh] = deg_shell_mag

    # for SOC we remove all degeneracies
    elif general_params['magnetic'] and sum_k.SO == 1:
        sum_k.block_structure.deg_shells = [[] for icrsh in range(len(sum_k.block_structure.deg_shells))]

    return sum_k, original_dens_mat, solver_struct_ftps


def _calculate_rotation_matrix(general_params, sum_k):
    """
    Applies rotation matrix to make the DMFT calculations easier for the solver.
    Possible are rotations diagonalizing either the local Hamiltonian or the
    density. Diagonalizing the density has not proven really helpful but
    diagonalizing the local Hamiltonian has.
    Note that the interaction Hamiltonian has to be rotated if it is not fully
    orbital-gauge invariant (only the Kanamori fulfills that).
    """

    # Extracts new rotation matrices from density_mat or local Hamiltonian
    if general_params['set_rot'] == 'hloc':
        q_diag = sum_k.eff_atomic_levels()
    elif general_params['set_rot'] == 'den':
        q_diag = sum_k.density_matrix(method='using_gf')
    else:
        raise ValueError('Parameter set_rot set to wrong value.')

    chnl = sum_k.spin_block_names[sum_k.SO][0]

    rot_mat = []
    for icrsh in range(sum_k.n_corr_shells):
        ish = sum_k.corr_to_inequiv[icrsh]
        eigvec = np.array(np.linalg.eigh(np.real(q_diag[ish][chnl]))[1], dtype=complex)
        if sum_k.use_rotations:
            rot_mat.append( np.dot(sum_k.rot_mat[icrsh], eigvec) )
        else:
            rot_mat.append( eigvec )

    sum_k.rot_mat = rot_mat
    # in case sum_k.use_rotations == False before:
    sum_k.use_rotations = True
    # sum_k.eff_atomic_levels() needs to be recomputed if rot_mat were changed
    if hasattr(sum_k, "Hsumk"): delattr(sum_k, "Hsumk")
    mpi.report('Updating rotation matrices using dft {} eigenbasis to maximise sign'.format(general_params['set_rot']))

    # Prints matrices
    mpi.report('\nNew rotation matrices')
    formatter.print_rotation_matrix(sum_k)

    return sum_k


def _chi_setup(sum_k, general_params, solver_params):
    """

    Parameters
    ----------
    sum_k : SumkDFT object
        Sumk object with the information about the correct block structure
    general_paramters: general params dict
    solver_params: solver params dict

    Returns
    -------
    solver_params :  dict
        solver_paramters for the QMC solver
    Op_list : list of one-particle operators to measure per impurity
    """

    if general_params['measure_chi'] == 'SzSz':
        mpi.report('\nSetting up Chi(S_z(tau),S_z(0)) measurement')
    elif general_params['measure_chi'] == 'NN':
        mpi.report('\nSetting up Chi(n(tau),n(0)) measurement')

    Op_list = [None] * sum_k.n_inequiv_shells

    for icrsh in range(sum_k.n_inequiv_shells):
        n_orb = sum_k.corr_shells[icrsh]['dim']
        orb_names = list(range(n_orb))

        if general_params['measure_chi'] == 'SzSz':
            Op_list[icrsh] = S_op('z',
                                  spin_names=sum_k.spin_block_names[sum_k.SO],
                                  n_orb=n_orb,
                                  map_operator_structure=sum_k.sumk_to_solver[icrsh])
        elif general_params['measure_chi'] == 'NN':
            Op_list[icrsh] = N_op(spin_names=sum_k.spin_block_names[sum_k.SO],
                                  n_orb=n_orb,
                                  map_operator_structure=sum_k.sumk_to_solver[icrsh])

    solver_params['measure_O_tau_min_ins'] = general_params['measure_chi_insertions']

    return solver_params, Op_list


def grisb_cycle(general_params, solver_params, advanced_params, dft_params,
               n_iter, dft_irred_kpt_indices=None, dft_energy=None):
    """
    main grisb cycle that works for one shot and CSC equally

    Parameters
    ----------
    general_params : dict
        general parameters as a dict
    solver_params : dict
        solver parameters as a dict
    advanced_params : dict
        advanced parameters as a dict
    observables : dict
        current observable array for calculation
    n_iter : int
        number of iterations to be executed
    dft_irred_kpt_indices: iterable of int
        If given, writes density correction for csc calculations only for
        irreducible kpoints

    Returns
    ---------
    observables : dict
        updated observable array for calculation
    """

    # create Sumk object
    # TODO: use_dft_blocks=True yields inconsistent number of blocks!

    # first we have to determine the mesh
    if general_params['solver_type'] in ['ftps']:
        sumk_mesh = MeshReFreq(window=general_params['w_range'],
                               n_w=general_params['n_w'])
    else:
        sumk_mesh = MeshImFreq(beta=general_params['beta'],
                               S='Fermion',
                               n_iw=general_params['n_iw'])

    sum_k = SumkGRISB(hdf_file=general_params['jobname']+'/'+general_params['seedname']+'.h5',
                      mesh=sumk_mesh, use_dft_blocks=False, h_field=general_params['h_field'],
                      nbaths=general_params['norb_baths'])

    iteration_offset = 0

    # determine chemical potential for bare DFT sum_k object
    if mpi.is_master_node():
        archive = HDFArchive(general_params['jobname']+'/'+general_params['seedname']+'.h5', 'a')
        if 'DMFT_results' not in archive:
            archive.create_group('DMFT_results')
        if 'last_iter' not in archive['DMFT_results']:
            archive['DMFT_results'].create_group('last_iter')
        if 'DMFT_input' not in archive:
            archive.create_group('DMFT_input')
            archive['DMFT_input']['program'] = 'solid_dmft'
            archive['DMFT_input'].create_group('solver')
            archive['DMFT_input'].create_group('version')
            archive['DMFT_input']['version']['triqs_hash'] = triqs_hash
            archive['DMFT_input']['version']['triqs_version'] = triqs_version
            archive['DMFT_input']['version']['solid_dmft_hash'] = solid_dmft_hash
            archive['DMFT_input']['version']['solid_dmft_version'] = solid_dmft_version

        if 'iteration_count' in archive['DMFT_results']:
            iteration_offset = archive['DMFT_results/iteration_count']
            sum_k.chemical_potential = archive['DMFT_results/last_iter/chemical_potential_post']
            print(f'RESTARTING DMFT RUN at iteration {iteration_offset+1} using last self-energy')
        else:
            print('INITIAL DMFT RUN')
        print('#'*80, '\n')
    else:
        archive = None

    # double counting using nominal valence
    #dens_mat = sum_k.density_matrix(method='using_gf')
    ##print(dens_mat)
    #for icrsh in range(sum_k.n_inequiv_shells):
    #    if general_params['h_int_type'][icrsh] == 'kanamori':
    #        sum_k.calc_dc(dens_mat[icrsh], orb=icrsh, U_interact=general_params['U'][icrsh],
    #                      J_hund=general_params['J'][icrsh], use_dc_formula=1)
    #    else:
    #        raise NotImplementedError('Slater-type interaction not implemente for gGA!')
    #print(sum_k.dc_imp)

    iteration_offset = mpi.bcast(iteration_offset)
    sum_k.chemical_potential = mpi.bcast(sum_k.chemical_potential)

    # Incompatabilities for SO coupling
    if sum_k.SO == 1:
        if not general_params['csc'] and general_params['magnetic'] and general_params['afm_order']:
            raise ValueError('AFM order not supported with SO coupling')

    # need to set sigma immediately here, otherwise mesh in unclear for sumK
    # Initializes empty Sigma for calculation of DFT density even if block structure changes later
    #zero_Sigma = [sum_k.block_structure.create_gf(ish=iineq, gf_function=Gf, mesh=sum_k.mesh)
    #              for iineq in range(sum_k.n_inequiv_shells)]
    #sum_k.put_Sigma(zero_Sigma)

    # Initializes chemical potential with mu_initial_guess if this is the first iteration
    if general_params['mu_initial_guess'] != 'none' and iteration_offset == 0:
            sum_k.chemical_potential = general_params['mu_initial_guess']
            mpi.report('\ninitial chemical potential set to {:.3f} eV\n'.format(sum_k.chemical_potential))

    if general_params['solver_type'] in ['ftps']:
        dft_mu = sum_k.calc_mu(precision=general_params['prec_mu'],
                               broadening=general_params['eta'])
    else:
        dft_mu = sum_k.calc_mu(precision=general_params['prec_mu'], method=general_params['calc_mu_method'])
    mpi.report('dft_mu={:2.8f}'.format(dft_mu))

    # calculate E_kin_dft for one shot calculations and CSC (Should be OK)
    #if not general_params['csc'] and general_params['calc_energies']:
    E_kin_dft = calc_dft_kin_en(general_params, sum_k, dft_mu)
    #else:
    #    E_kin_dft = None

    # check for previous broyden data oterhwise initialize it:
    #if mpi.is_master_node() and  general_params['g0_mix_type'] == 'broyden':
    #    if not 'broyler' in archive['DMFT_results']:
    #        archive['DMFT_results']['broyler'] = [{'mu' : [],'V': [], 'dV': [], 'F': [], 'dF': []}
    #                                              for _ in range(sum_k.n_inequiv_shells)]

    # Generates a rotation matrix to change the basis
    if general_params['set_rot'] != 'none':
        # calculate new rotation matrices
        sum_k = _calculate_rotation_matrix(general_params, sum_k)
    # Saves rotation matrix to h5 archive:
    if mpi.is_master_node() and iteration_offset == 0:
        archive['DMFT_input']['rot_mat'] = sum_k.rot_mat
    mpi.barrier()

    # determine block structure for solver
    det_blocks = None
    # load previous block_structure if possible
    if mpi.is_master_node():
        det_blocks = 'block_structure' not in archive['DMFT_input']
    det_blocks = mpi.bcast(det_blocks)

    # Previous rot_mat only not None if the rot_mat changed from load_sigma or previous run
    previous_rot_mat = None
    solver_struct_ftps = None
    # determine block structure for GF and Hyb function
    if det_blocks and not general_params['load_sigma']:
        sum_k, dm, solver_struct_ftps = _determine_block_structure(sum_k, general_params, advanced_params)
    # if load sigma we need to load everything from this h5 archive
    elif general_params['load_sigma']:
        #loading block_struc and rot_mat and deg_shells
        if mpi.is_master_node():
            with HDFArchive(general_params['path_to_sigma'], 'r') as old_calc:
                sum_k.block_structure = old_calc['DMFT_input/block_structure']
                sum_k.deg_shells = old_calc['DMFT_input/deg_shells']
                previous_rot_mat = old_calc['DMFT_input/rot_mat']
                if general_params['solver_type'] in ['ftps']:
                    solver_struct_ftps = old_calc['DMFT_input/solver_struct_ftps']

            if not all(np.allclose(x, y) for x, y in zip(sum_k.rot_mat, previous_rot_mat)):
                print('WARNING: rot_mat in current run is different from loaded_sigma run.')
            else:
                previous_rot_mat = None

        sum_k.block_structure = mpi.bcast(sum_k.block_structure)
        sum_k.deg_shells = mpi.bcast(sum_k.deg_shells)
        previous_rot_mat = mpi.bcast(previous_rot_mat)
        solver_struct_ftps = mpi.bcast(solver_struct_ftps)

        # In a magnetic calculation, no shells are degenerate
        if general_params['magnetic'] and sum_k.SO == 0:
            sum_k.deg_shells = [[] for _ in range(sum_k.n_inequiv_shells)]
        dm = None
    else:
        # Master node checks if rot_mat stayed the same
        if mpi.is_master_node():
            sum_k.block_structure = archive['DMFT_input']['block_structure']
            sum_k.deg_shells = archive['DMFT_input/deg_shells']
            previous_rot_mat = archive['DMFT_input']['rot_mat']
            if not all(np.allclose(x, y) for x, y in zip(sum_k.rot_mat, previous_rot_mat)):
                print('WARNING: rot_mat in current step is different from previous step.')
                archive['DMFT_input']['rot_mat'] = sum_k.rot_mat
            else:
                previous_rot_mat = None
            if general_params['solver_type'] in ['ftps']:
                solver_struct_ftps = archive['DMFT_input/solver_struct_ftps']

        sum_k.block_structure = mpi.bcast(sum_k.block_structure)
        sum_k.deg_shells = mpi.bcast(sum_k.deg_shells)
        previous_rot_mat = mpi.bcast(previous_rot_mat)
        solver_struct_ftps = mpi.bcast(solver_struct_ftps)
        dm = None

    # Compatibility with h5 archives from the triqs2 version
    # Sumk doesn't hold corr_to_inequiv anymore, which is in block_structure now
    if sum_k.block_structure.corr_to_inequiv is None:
        if mpi.is_master_node():
            sum_k.block_structure.corr_to_inequiv = archive['dft_input/corr_to_inequiv']
        sum_k.block_structure = mpi.bcast(sum_k.block_structure)

    # Determination of shell_multiplicity
    shell_multiplicity = [sum_k.corr_to_inequiv.count(icrsh) for icrsh in range(sum_k.n_inequiv_shells)]

    # Initializes new empty Sigma with new blockstructure for calculation of DFT density
    #zero_Sigma = [sum_k.block_structure.create_gf(ish=iineq, gf_function=Gf, mesh=sum_k.mesh)
    #              for iineq in range(sum_k.n_inequiv_shells)]
    #sum_k.put_Sigma(zero_Sigma)

    # print block structure and DFT input quantitites!
    formatter.print_block_sym(sum_k, dm, general_params)

    # extract free lattice greens function
    if general_params['solver_type'] in ['ftps']:
        G_loc_all_dft = sum_k.extract_G_loc(broadening=general_params['eta'], with_Sigma=False, mu=dft_mu)
    else:
        G_loc_all_dft = sum_k.extract_G_loc( mu=dft_mu)
    density_mat_dft = [G_loc_all_dft[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]

    for iineq in range(sum_k.n_inequiv_shells):
        density_shell_dft = G_loc_all_dft[iineq].total_density()
        mpi.report('total density for imp {} from DFT: {:10.6f}'.format(iineq, np.real(density_shell_dft)))

    if general_params['magnetic']:
        sum_k.SP = 1

        if general_params['afm_order']:
            general_params = afm_mapping.determine(general_params, archive, sum_k.n_inequiv_shells)

    # Constructs interaction Hamiltonian and writes it to the h5 archive
    h_int =  interaction_hamiltonian.construct(sum_k, general_params, advanced_params)# we need to store h_int as U_tensor
    if mpi.is_master_node():
        archive['DMFT_input']['h_int'] = h_int

    # If new calculation, writes input parameters and sum_k <-> solver mapping to archive
    if iteration_offset == 0:
        if mpi.is_master_node():
            archive['DMFT_input']['general_params'] = general_params
            archive['DMFT_input']['solver_params'] = solver_params
            archive['DMFT_input']['advanced_params'] = advanced_params

            archive['DMFT_input']['block_structure'] = sum_k.block_structure
            archive['DMFT_input']['deg_shells'] = sum_k.deg_shells
            archive['DMFT_input']['shell_multiplicity'] = shell_multiplicity
            if general_params['solver_type'] in ['ftps']:
                archive['DMFT_input']['solver_struct_ftps'] = solver_struct_ftps

    solvers = [None] * sum_k.n_inequiv_shells
    for icrsh in range(sum_k.n_inequiv_shells):
        # Construct the Solver instances
        solvers[icrsh] = SolverStructure(general_params, solver_params, advanced_params,
                                         sum_k, icrsh, h_int[icrsh],
                                         iteration_offset, solver_struct_ftps)

    # store solver hash to archive
    if mpi.is_master_node():
        if 'version' not in archive['DMFT_input']:
            archive['DMFT_input'].create_group('version')
        archive['DMFT_input']['version']['solver_name'] = general_params['solver_type']
        #archive['DMFT_input']['version']['solver_hash'] = solvers[0].git_hash
        #archive['DMFT_input']['version']['solver_version'] = solvers[0].version
    #print('here. below need to take care of the double counting term')
    #print('density_mat_dft=')
    #print(density_mat_dft)
    #quit()
    # Determines initial Sigma and DC
    sum_k, solvers = initial_sigma.determine_dc_and_initial_sigma(general_params, advanced_params, sum_k,
                                                                  archive, iteration_offset, density_mat_dft, solvers)

    sum_k = manipulate_mu.set_initial_mu(general_params, sum_k, iteration_offset, archive, shell_multiplicity)


    # setup of measurement of chi(SzSz(tau) if requested
    if general_params['measure_chi'] != 'none':
        solver_params, Op_list = _chi_setup(sum_k, general_params, solver_params)
    else:
        Op_list = None

    mpi.report('\n {} GRISB cycles requested. Starting with iteration  {}.\n'.format(n_iter, iteration_offset+1))

    # Prepares observable and conv dicts
    observables = None
    conv_obs = None
    if mpi.is_master_node():
        observables = prep_observables(archive, sum_k)
        conv_obs = convergence.prep_conv_obs(archive, sum_k)
    observables = mpi.bcast(observables)
    conv_obs = mpi.bcast(conv_obs)

    if mpi.is_master_node() and iteration_offset == 0:
        write_header_to_file(general_params, sum_k)
        observables = add_dft_values_as_zeroth_iteration(observables, general_params, dft_mu, dft_energy, sum_k,
                                                         G_loc_all_dft, density_mat_dft, shell_multiplicity)
        # set up observable R and Lambda
        #print(observables['R'])
        #quit()
        for icrsh in range(sum_k.n_inequiv_shells):
            #for spin in sum_k.spin_block_names[sum_k.SO]:
            np.random.seed(1234)
            n_orb = sum_k.corr_shells[icrsh]['dim']
            if general_params['norb_baths'][icrsh] == n_orb:
                observables['R'][icrsh]['up'] = np.eye(general_params['norb_baths'][icrsh],dtype=complex)
                observables['R'][icrsh]['down'] = np.eye(general_params['norb_baths'][icrsh],dtype=complex)
                #observables['Lambda'][icrsh][spin] = np.zeros((general_params['norb_bath'],general_params['norb_bath']),dtype=complex)
                observables['Lambda'][icrsh]['up'] = sum_k.Hsumk[icrsh]['up']
                observables['Lambda'][icrsh]['down'] = sum_k.Hsumk[icrsh]['down']
            else:
                R0 = np.random.rand(general_params['norb_baths'][icrsh],n_orb)*0.9
                observables['R'][icrsh]['up'] = R0
                observables['R'][icrsh]['down'] = R0
                Lambda0 = np.random.rand(general_params['norb_baths'][icrsh],general_params['norb_baths'][icrsh])*2.0
                Lambda0 = (Lambda0 + Lambda0.T)/2 #+ dft_mu*np.eye(Lambda0.shape[0])
                observables['Lambda'][icrsh]['up'] = Lambda0
                observables['Lambda'][icrsh]['down'] = Lambda0

        print('Initial R =')
        print(observables['R'])
        print('Initial Lambda =')
        print(observables['Lambda'])
        write_obs(observables, sum_k, general_params)
        # write convergence file
        convergence.prep_conv_file(general_params, sum_k)
    elif mpi.is_master_node():
        # read R and Lambda from archive
        observables['R'] = archive['DMFT_results/last_iter/R']
        observables['Lambda'] = archive['DMFT_results/last_iter/Lambda']

    observables = mpi.bcast(observables)

    # need to close archive before entering _grisb
    if mpi.is_master_node():
        del archive
        # The line before is useful for debugging unclosed hdf5 archive
        #import h5py
        #print('here')
        #fh5 = h5py.File('nio.h5','r')
        #fh5.close()

    # Initialize the convergence flags to false
    is_converged = False
    if general_params['csc']:
        is_charge_converged = False
        is_energy_converged = False
#        #load previous density and energy correction
#        try:
#            band_en_correction_old = None
#            if mpi.is_master_node():
#                with HDFArchive(sum_k.hdf_file, 'r') as ar:
#                    band_en_correction_old = ar['dft_update']['band_en_correction']
#            band_en_correction_old = mpi.bcast(band_en_correction_old)
#        except:
#            band_en_correction_old = 0.0
#        try:
#            deltaN_old = None
#            if mpi.is_master_node():
#                with HDFArchive(sum_k.hdf_file, 'r') as ar:
#                    deltaN_old = ar['dft_update']['delta_N']
#            deltaN_old = mpi.bcast(deltaN_old)
#        except:
#            #deltaN_old = {}
#            ntoi = sum_k.spin_names_to_ind[sum_k.SO]
#            spn = sum_k.spin_block_names[sum_k.SO]
#            deltaN_old = np.zeros((sum_k.n_k,sum_k.n_orbitals[0, ntoi[spn[0]]],sum_k.n_orbitals[0, ntoi[spn[0]]]),dtype=complex)
            #for sp in spn:
            #    deltaN_old[sp] = [np.zeros([sum_k.n_orbitals[ik, ntoi[sp]], sum_k.n_orbitals[
            #                            ik, ntoi[sp]]], complex) for ik in range(sum_k.n_k)]

    # The not famous GRISB self consistency cycle
    for it in range(iteration_offset + 1, iteration_offset + n_iter + 1):

        # remove h_field when number of iterations is reached
        if sum_k.h_field != 0.0 and general_params['h_field_it'] != 0 and it > general_params['h_field_it']:
            mpi.report('\nRemoving magnetic field now.\n')
            sum_k.h_field = 0.0
            # enforce recomputation of eff_atomic_levels
            delattr(sum_k, 'Hsumk')

        mpi.report('#'*80)
        mpi.report('Running iteration: {} / {}'.format(it, iteration_offset + n_iter))

        (sum_k, solvers,
         observables, is_converged) = _grisb_step(sum_k, solvers, it, general_params,
                                                 solver_params, advanced_params, dft_params,
                                                 #h_int, archive, shell_multiplicity, E_kin_dft,
                                                 h_int, shell_multiplicity, E_kin_dft,
                                                 observables, conv_obs, Op_list, dft_irred_kpt_indices, dft_energy, density_mat_dft,
                                                 is_converged, is_sampling=False)
        if is_converged:
            break

    #load and check charge and energy convergence
#    if general_params['csc']:
#        try:
#            band_en_correction = None
#            if mpi.is_master_node():
#                with HDFArchive(sum_k.hdf_file, 'r') as ar:
#                    band_en_correction = ar['dft_update']['band_en_correction']
#            band_en_correction = mpi.bcast(band_en_correction)
#            print('band_en_correction=',band_en_correction)
#        except:
#            print('the sumk_grisb should output the band_en_correction_old')
#            raise
#        try:
#            deltaN = None
#            if mpi.is_master_node():
#                with HDFArchive(sum_k.hdf_file, 'r') as ar:
#                    deltaN = ar['dft_update']['delta_N']
#                #print(deltaN)
#                #print(deltaN_old)
#            deltaN = mpi.bcast(deltaN)
#        except:
#            print('the sumk_grisb should output the deltaN')
#            raise
#        energy_diff = np.abs(band_en_correction_old -band_en_correction).real
#        charge_diff = np.max(np.abs(deltaN-deltaN_old))
#        mpi.report('########################## charge_diff={:.6f}'.format(charge_diff) +
#                   ' energy_diff={:.6f} #########################'.format(energy_diff))
#        if ( energy_diff < general_params['charge_tol'] and charge_diff < general_params['energy_tol'] ):
#            is_charge_converged = True
#            is_energy_converged = True

    #compute Green's function
    mesh_plot = MeshReFreq(window=general_params['w_range'],
                           n_w=general_params['n_w'])
    #sum_k.lattice_gf_qp(observables['R'], observables['Lambda'], 0, mu=None, broadening=0.05, mesh=mesh_plot)
    Gphy = sum_k.extract_G_phy(observables['R'], observables['Lambda'], mu=None, broadening=0.05, mesh=mesh_plot, show_warnings=True)
    if mpi.is_master_node():
        with HDFArchive(sum_k.hdf_file, 'a') as archive:
            if 'gGA_results' not in archive:
                archive.create_group('gGA_results')
            if 'Gphys' not in archive['gGA_results']:
               archive['gGA_results'].create_group('Gphy')
            archive['gGA_results']['Gphy'] = Gphy

    if is_converged:
        mpi.report('*** Required convergence reached ***')
    else:
        mpi.report('** All requested iterations finished ***')
    mpi.report('#'*80)

    mpi.barrier()

    # close the h5 archive
    if mpi.is_master_node():
        del archive

    if general_params['csc']:
        return (is_charge_converged and is_energy_converged), sum_k
    else:
        return is_converged, sum_k


def _grisb_step(sum_k, solvers, it, general_params,
               solver_params, advanced_params, dft_params,
               #h_int, archive, shell_multiplicity, E_kin_dft,
               h_int, shell_multiplicity, E_kin_dft,
               observables, conv_obs, Op_list, dft_irred_kpt_indices, dft_energy, density_mat_dft,
               is_converged, is_sampling):
    """
    Contains the actual grisb steps when all the preparation is done
    Question: How should I organized R and Lambda? They shouldn't belong to the solver class.
              They should go to the sum_k class
    """
    if mpi.is_master_node():
        archive = HDFArchive(general_params['jobname']+'/'+general_params['seedname']+'.h5', 'a')

    #print('h_int=')
    #print(h_int)
    mpi.report('density_required={:.4f}'.format(sum_k.density_required))
    # compute new chemical potential
    mu = sum_k.calc_mu_grisb(observables['R'], observables['Lambda'], precision=general_params['prec_mu'],
                             method=general_params['calc_mu_method'], beta=general_params['beta'])
    #quit()

    # init local density matrices for observables
    density_tot = 0.0
    density_shell = np.zeros(sum_k.n_inequiv_shells)
    density_mat = [{} for icrsh in range(sum_k.n_inequiv_shells)]#[{}] * sum_k.n_inequiv_shells
    density_mat_unsym = [{} for icrsh in range(sum_k.n_inequiv_shells)]#[{}] * sum_k.n_inequiv_shells
    density_shell_pre = np.zeros(sum_k.n_inequiv_shells)
    density_mat_pre = [{} for icrsh in range(sum_k.n_inequiv_shells)]#[{}] * sum_k.n_inequiv_shells

    mpi.barrier()

    if sum_k.SO:
        printed = ((np.real, 'real'), (np.imag, 'imaginary'))
    else:
        printed = ((np.real, 'real'), )

    # Extracts G local
    #if general_params['solver_type'] in ['ftps']:
    #    G_loc_all = sum_k.extract_G_loc(broadening=general_params['eta'])
    #else:
    #    G_loc_all = sum_k.extract_G_loc()

    # Copies Sigma and G0 before Solver run for mixing later
    #Sigma_freq_previous = [solvers[iineq].Sigma_freq.copy() for iineq in range(sum_k.n_inequiv_shells)]
    #G0_freq_previous = [solvers[iineq].G0_freq.copy() for iineq in range(sum_k.n_inequiv_shells)]
    #if general_params['dc'] and general_params['dc_type'] == 4:
    #    cpa_G_loc = gf_mixer.init_cpa(sum_k, solvers, general_params)

    diff = 0 # Initialize difference between previous R and Lambda as 0
    # looping over inequiv shells and solving for each site seperately
    for icrsh in range(sum_k.n_inequiv_shells):
        # copy the block of G_loc into the corresponding instance of the impurity solver
        # TODO: why do we set solvers.G_freq? Isn't that simply an output of the solver?
        #solvers[icrsh].G_freq << G_loc_all[icrsh]

        density_shell_pre[icrsh] = np.real(np.trace(solvers[icrsh].density_matrix[:2*solvers[icrsh].nimp,:2*solvers[icrsh].nimp]))
        mpi.report('\n *** Correlated Shell type #{:3d} : '.format(icrsh)
                   + 'Estimated total charge of impurity problem = {:.6f}'.format(density_shell_pre[icrsh]))
        R_pre_icrsh = deepcopy(observables['R'][icrsh])
        Lambda_pre_icrsh = deepcopy(observables['Lambda'][icrsh])
        # parse density matrix to spin resolved matrix
        for spin_channel in sorted(sum_k.gf_struct_solver[icrsh].keys()):
            isp = int(spin_channel=='down_%d'%(icrsh))# this needs to be changed in future
            density_mat_pre[icrsh][spin_channel] = solvers[icrsh].density_matrix[isp:2*solvers[icrsh].nimp:2,
                                                               isp:2*solvers[icrsh].nimp:2]
        mpi.report('Estimated density matrix:')
        for key, value in sorted(density_mat_pre[icrsh].items()):
            for func, name in printed:
                mpi.report('{}, {} part'.format(key, name))
                mpi.report(func(value))

        # Compute Delta
        #print(general_params['beta'])
        #for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
        #    print('R_%s='%sp)
        #    print(observables['R'][icrsh][sp])
        #    print('Lambda_%s='%sp)
        #    print(observables['Lambda'][icrsh][sp])
        sum_k.calc_rhoks(observables['R'], observables['Lambda'], 1./general_params['beta'])
        sum_k.calc_Delta()
        #Delta_sym = (sum_k.Delta[icrsh]['up']+sum_k.Delta[icrsh]['down'])/2.# symmetrize
        #for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
        #    sum_k.Delta[icrsh][sp] = cut_small(Delta_sym, tol=1e-8)
        #for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
        #    print('Delta_%s='%sp)
        #    print(sum_k.Delta[icrsh][sp])

        # Compute D
        sum_k.calc_D(observables['R'], observables['Lambda'])
        #for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
        #    print('D_%s='%sp)
        #    print(sum_k.D[icrsh][sp])

        # Compute Lambda_c
        sum_k.calc_Lambdac(observables['R'], observables['Lambda'])
        #for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
        #    print('Lambdac_%s='%sp)
        #    print(sum_k.Lambdac[icrsh][sp])

         # store solver to h5 archive
        if general_params['store_solver'] and mpi.is_master_node():
            archive['DMFT_input/solver'].create_group('it_'+str(it))
            archive['DMFT_input/solver/it_'+str(it)]['S_'+str(icrsh)] = solvers[icrsh].triqs_solver

        # store DMFT input directly in last_iter
        #if mpi.is_master_node():
        #    archive['DMFT_results/last_iter']['G0_freq_{}'.format(icrsh)] = solvers[icrsh].G0_freq

        # setup of measurement of chi(SzSz(tau) if requested
        #if general_params['measure_chi'] != 'none':
        #    solvers[icrsh].solver_params['measure_O_tau'] = (Op_list[icrsh], Op_list[icrsh])

        if (general_params['magnetic'] and general_params['afm_order'] and general_params['afm_mapping'][icrsh][0]):
            # If we do a AFM calculation we can use the init magnetic moments to
            # copy the self energy instead of solving it explicitly
            solvers = afm_mapping.apply(general_params, solver_params, icrsh, sum_k.gf_struct_solver[icrsh], solvers)
        else:
            # Solve the impurity problem for this shell
            mpi.report('\nSolving the impurity problem for shell {} ...'.format(icrsh))
            mpi.barrier()
            start_time = timer()
            solvers[icrsh].solve(it=it)
            mpi.barrier()
            mpi.report('Actual time for solver: {:.2f} s'.format(timer() - start_time))

        # compute new R and Lambda
        cdaggerf = solvers[icrsh].density_matrix[:2*solvers[icrsh].nimp,2*solvers[icrsh].nimp:]
        ffdagger = solvers[icrsh].density_matrix[2*solvers[icrsh].nimp:,2*solvers[icrsh].nimp:]
        ffdagger = (np.eye(2*solvers[icrsh].nbath,dtype=complex) - ffdagger).T
        mpi.report("norm(ffdagger.T-Delta_p)_up= {:.2e}".format(np.linalg.norm(ffdagger[::2,::2].T-sum_k.Delta[icrsh]["up"])) )
        mpi.report("norm(ffdagger.T-Delta_p)_down= {:.2e}".format(np.linalg.norm(ffdagger[1::2,1::2].T-sum_k.Delta[icrsh]["down"])) )
        Delta_spinful = ffdagger.T
        #print("Delta new:")
        #print(Delta_spinful)
        R_new_spinful = np.transpose(cdaggerf.dot(funcMat(Delta_spinful, denR)))
        #print("R_new=")
        #print(R_new_spinful)
        #R_new = svd_truncate_R(R_new)
        # Convert R_spinful and Delta_spinful to R and Lambda data structure
        R_new_icrsh = deepcopy(observables['R'][icrsh])
        Delta = {}
        for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
            isp = int(sp=='down')# this needs to be changed in future
            #print(isp)
            R_new_icrsh[sp] = R_new_spinful[isp::2,isp::2]
            Delta[sp] = Delta_spinful[isp::2,isp::2]
        Lambda_new_icrsh = {}
        for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
            Lambda_new_icrsh[sp] = sum_k.calc_Lambda_icrsh_isp(R_new_icrsh[sp], sum_k.Lambdac[icrsh][sp],
                                        Delta[sp], sum_k.D[icrsh][sp], sum_k.H_list[icrsh][sp])
        #print('R_pre_icrsh=')
        #print(R_pre_icrsh)
        #print('R_new_icrsh=')
        #print(R_new_icrsh)
        #print('Lambda_pre_icrsh=')
        #print(Lambda_pre_icrsh)
        #print('Lambda_new_icrsh=')
        #print(Lambda_new_icrsh)
        # symmetrize over spin
        R_sym = (R_new_icrsh['up']+R_new_icrsh['down'])/2.# symmetrize
        Lambda_sym = (Lambda_new_icrsh['up']+Lambda_new_icrsh['down'])/2. # symmetryize
        for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
            R_new_icrsh[sp] = cut_small(R_sym, tol=1e-8)
            Lambda_new_icrsh[sp] = cut_small(Lambda_sym, tol=1e-8)
        diff_R, diff_Lambda = 0.0, 0.0
        for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
            diff_R = np.abs(R_pre_icrsh[sp]-R_new_icrsh[sp]).max()
            diff_Lambda = np.abs(Lambda_pre_icrsh[sp]-Lambda_new_icrsh[sp]).max()
        diff += max(diff_R,diff_Lambda)

        # some printout of the obtained density matrices and some basic checks from the unsymmetrized solver output
        # parse solver density matrix to spin resolved index
        # parse density matrix to the grisb_cycle style structure
        density_shell[icrsh] = np.real(np.trace( solvers[icrsh].density_matrix[:2*solvers[icrsh].nimp,:2*solvers[icrsh].nimp] ) )
        density_tot += density_shell[icrsh]*shell_multiplicity[icrsh]
        for spin_channel in sorted(sum_k.gf_struct_solver[icrsh].keys()):
            #mpi.report('spinchannel=',spin_channel)
            isp = int(spin_channel=='down_%d'%(icrsh))
            density_mat[icrsh][spin_channel] = solvers[icrsh].density_matrix[isp:2*solvers[icrsh].nimp:2,
                                                           isp:2*solvers[icrsh].nimp:2]
            density_mat_unsym[icrsh][spin_channel] = solvers[icrsh].density_matrix[isp:2*solvers[icrsh].nimp:2,
                                                                 isp:2*solvers[icrsh].nimp:2]
        print('density_mat=')
        print(density_mat)
        formatter.print_local_density(density_shell[icrsh], density_shell_pre[icrsh],
                                      density_mat_unsym[icrsh], sum_k.SO)

        # update solver in h5 archive
        if general_params['store_solver'] and mpi.is_master_node():
            archive['DMFT_input/solver/it_'+str(it)]['S_'+str(icrsh)] = solvers[icrsh].triqs_solver

        # add to cpa_G_time
        #if general_params['dc'] and general_params['dc_type'] == 4:
        #    cpa_G_time << cpa_G_time + general_params['cpa_x'][icrsh] * solvers[icrsh].G_time

        # mixing R and Lambda and update the R and Lambda in the observable class
        for sp, isp in sum_k.spin_names_to_ind[sum_k.SO].items():
            observables['R'][icrsh][sp] = (1.0-general_params['grisb_mix'])*R_pre_icrsh[sp] + general_params['grisb_mix']*R_new_icrsh[sp]
            observables['Lambda'][icrsh][sp] = (1.0-general_params['grisb_mix'])*Lambda_pre_icrsh[sp] + general_params['grisb_mix']*Lambda_new_icrsh[sp]
        #quit()
    diff /= sum_k.n_inequiv_shells # average the difference over shells
    mpi.report('diff= {:.2e}'.format(diff))

    # Done with loop over impurities
#    quit()

    if mpi.is_master_node():
        # Done. Now do post-processing:
        print('\n *** Post-processing the solver output ***')
        print('Total charge of all correlated shells : {:.6f}\n'.format(density_tot))

    # if CPA average Sigma over impurities before mixing
    #if general_params['dc'] and general_params['dc_type'] == 4:
    #    solvers = gf_mixer.mix_cpa(cpa_G0_freq, sum_k.n_inequiv_shells, solvers)
    #solvers = gf_mixer.mix_sigma(general_params, sum_k.n_inequiv_shells, solvers, Sigma_freq_previous)

    # calculate new DC
    # for the hartree solver the DC potential will be formally set to zero as it is already present in the Sigma
    if general_params['dc'] and general_params['dc_grisb']:
        sum_k = initial_sigma.calculate_double_counting(sum_k, density_mat,
                                                        general_params, advanced_params)

    #The hartree solver computes the DC energy internally, set it in sum_k
    #if general_params['solver_type'] == 'hartree':
    #    for icrsh in range(sum_k.n_inequiv_shells):
    #        sum_k.dc_energ[icrsh] = solvers[icrsh].DC_energy

    # doing the dmft loop and set new sigma into sumk
    #sum_k.put_Sigma([solvers[icrsh].Sigma_freq for icrsh in range(sum_k.n_inequiv_shells)])

    # saving previous mu for writing to observables file
    #if it > 1:
    #    previous_mu = sum_k.chemical_potential*(1-general_params['mu_mix_const']) + previous_mu*general_params['mu_mix_const']
    #else:
    previous_mu = sum_k.chemical_potential
    #sum_k = manipulate_mu.update_mu(general_params, sum_k, it, archive)

    # if we do a CSC calculation we need always an updated GAMMA file
    E_bandcorr = 0.0
    deltaN = None
    dens = None
    if general_params['csc']:
        # handling the density correction for fcsc calculations
        assert dft_irred_kpt_indices is None or dft_params['dft_code'] == 'vasp'
        deltaN, dens, E_bandcorr = sum_k.calc_density_correction(density_mat, observables, E_kin_dft, dm_type=dft_params['dft_code'],
                                                                 kpts_to_write=dft_irred_kpt_indices)
    elif general_params['calc_energies']:
        # for a one shot calculation we are using our own method
        #deltaN, dens, E_bandcorr = sum_k.calc_density_correction(density_mat, observables, E_kin_dft, dm_type='qe',#dft_params['dft_code'],
        #                                                         kpts_to_write=dft_irred_kpt_indices)
        E_bandcorr = calc_bandcorr_man(observables['R'], observables['Lambda'], general_params, sum_k, E_kin_dft)

    # Writes results to h5 archive
    if mpi.is_master_node():
        results_to_archive.write(archive, sum_k, general_params, solver_params, solvers, it,
                                 is_sampling, previous_mu, density_mat_pre, density_mat,
                                 observables['R'], observables['Lambda'], deltaN, dens)

    mpi.barrier()

    # calculate observables and write them to file
    if mpi.is_master_node():
        print('\n *** calculation of observables ***')
        observables = add_grisb_observables(observables,
                                           general_params,
                                           solver_params,
                                           dft_energy,
                                           it,
                                           solvers,
                                           h_int,
                                           previous_mu,
                                           sum_k,
                                           density_mat,
                                           shell_multiplicity,
                                           E_bandcorr)

        write_obs(observables, sum_k, general_params)

        # write the new observable array to h5 archive
        archive['DMFT_results']['observables'] = observables

    # Computes convergence quantities and writes them to file
    if mpi.is_master_node():
        conv_obs = convergence.calc_convergence_quantities(sum_k, general_params, conv_obs, observables,
                                                           solvers)#, G0_freq_previous, G_loc_all, Sigma_freq_previous)
        convergence.write_conv(conv_obs, sum_k, general_params)
        archive['DMFT_results']['convergence_obs'] = conv_obs
    conv_obs = mpi.bcast(conv_obs)

    mpi.report('*** iteration finished ***')

    # Checks for convergence
    is_now_converged = convergence.check_convergence(sum_k.n_inequiv_shells, general_params, conv_obs)
    print('is_now_converged=', is_now_converged)
    # use the current simple criterion for one-shot
    if not general_params['csc'] and diff < general_params['grisb_tol']:
        is_converged =True
    elif is_now_converged is None:
        is_converged = False
    else:
        # if convergency criteria was already reached don't overwrite it!
        is_converged = is_converged or is_now_converged
    print('is_converged=', is_converged)

    # Final prints
    formatter.print_summary_observables(observables, sum_k.n_inequiv_shells,
                                        sum_k.spin_block_names[sum_k.SO])
    if general_params['calc_energies']:
        formatter.print_summary_energetics(observables)
    if general_params['magnetic'] and sum_k.SO == 0:
        # if a magnetic calculation is done print out a summary of up/down occ
        formatter.print_summary_magnetic_occ(observables, sum_k.n_inequiv_shells)
    #formatter.print_summary_convergence(conv_obs, general_params, sum_k.n_inequiv_shells)

    print('dft_energy=', dft_energy)
    print('density_mat_dft=')
    print(density_mat_dft)
    #print(sum_k.rhoks_phys_bloch)

    if mpi.is_master_node():
        del archive

    return sum_k, solvers, observables, is_converged
