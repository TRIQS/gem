#!@TRIQS_PYTHON_EXECUTABLE@
"""
copied from solid_dmft for one-shot and CSC grisb implementation
"""

# system
import os
import sys
import shutil
from timeit import default_timer as timer

# triqs
import triqs.utility.mpi as mpi

# own modules
from solid_dmft.read_config_grisb import read_config
from solid_dmft.dmft_cycle import dmft_cycle
from solid_dmft.csc_flow import csc_flow_control


def main(argv=sys.argv):
    """The main function for one-shot and charge-self-consistent calculations"""
    # timing information
    if mpi.is_master_node():
        global_start = timer()

    # reading configuration for calculation
    general_params = None
    solver_params = None
    dft_params = None
    advanced_params = None
    if len(argv) > 1:
        config_file_name = str(argv[1])
    else:
        config_file_name = 'grisb_config.ini'
    if not os.path.isfile(config_file_name):
        raise FileNotFoundError(f'Could not find config file {config_file_name}.')

    if mpi.is_master_node():
        print('Reading the config file ' + config_file_name)
        general_params, solver_params, dft_params, advanced_params = read_config(config_file_name)
        general_params['config_file'] = config_file_name

        print('-'*25 + '\nGeneral parameters:')
        for key, value in general_params.items():
            print('{0: <20} {1: <4}'.format(key, str(value)))
        print('-'*25 + '\nSolver parameters:')
        for key, value in solver_params.items():
            print('{0: <20} {1: <4}'.format(key, str(value)))
        print('-'*25 + '\nDFT parameters:')
        for key, value in dft_params.items():
            print('{0: <20} {1: <4}'.format(key, str(value)))
        print('-'*25 + '\nAdvanced parameters, don\'t change them unless you know what you are doing:')
        for key, value in advanced_params.items():
            print('{0: <20} {1: <4}'.format(key, str(value)))

    general_params = mpi.bcast(general_params)
    solver_params = mpi.bcast(solver_params)
    dft_params = mpi.bcast(dft_params)
    advanced_params = mpi.bcast(advanced_params)

    if general_params['csc']:
        # Start CSC calculation, always in same folder as grisb_config
        general_params['jobname'] = '.'
        csc_flow_control(general_params, solver_params, dft_params, advanced_params)
    else:
        # Sets up one-shot calculation
        mpi.report('', '#'*80)
        mpi.report(f'Using input file {general_params["seedname"]}.h5 '
                   + f'and running in folder {general_params["jobname"]}')

        if mpi.is_master_node():
            # Checks for h5 file
            if not os.path.exists(general_params['seedname']+'.h5'):
                raise FileNotFoundError('Input h5 file not found')

            # Creates output directory if it does not exist
            if not os.path.exists(general_params['jobname']):
                os.makedirs(general_params['jobname'])

            # Copies h5 archive and config file to subfolder if are not there
            for file in (general_params['seedname']+'.h5',
                         general_params['config_file']):
                if not os.path.isfile(general_params['jobname']+'/'+os.path.basename(file)):
                    shutil.copyfile(file, general_params['jobname']+'/'+os.path.basename(file))
        mpi.barrier()

        # Runs grisb_cycle
        grisb_cycle(general_params, solver_params, advanced_params,
                   dft_params, general_params['n_iter_grisb'])

    mpi.barrier()
    if mpi.is_master_node():
        global_end = timer()
        print('-------------------------------')
        print('overall elapsed time: %10.4f seconds'%(global_end-global_start))


if __name__ == '__main__':
    main(sys.argv)
