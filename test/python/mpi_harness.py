'''Helpers for the tests that ctest launches under ``mpirun``.

Not a test module. ``test_mpi_fragment.py`` and ``test_mpi_lattice.py`` run
inside an MPI job and compare the distributed result against the serial one,
computed redundantly on the same rank by building the same object with
``use_mpi=False``.

Two things need care once unittest runs on several ranks at once:

* a failure on one rank only would leave the others waiting at the next
  collective, so :func:`run` turns any failure into ``MPI_Abort``;
* every rank writes to the same terminal, so only rank 0 reports.
'''

import sys
import unittest

try:
    from gem.mpi import MPI, HAS_MPI, resolve_comm
except ImportError:                                  # pragma: no cover
    MPI, HAS_MPI = None, False

COMM = resolve_comm(None) if HAS_MPI else None
RANK = COMM.Get_rank() if HAS_MPI else 0
SIZE = COMM.Get_size() if HAS_MPI else 1

# on one rank the split is the whole range and every reduction is a no-op, so
# the comparison would hold trivially
NEEDS_RANKS = 'needs mpirun with more than one rank'
CAN_RUN = HAS_MPI and SIZE > 1


def run(module):
    '''Entry point for ``python <test_file>.py`` inside an MPI job.'''
    stream = sys.stderr if RANK == 0 else open('/dev/null', 'w')
    result = unittest.main(module=module, exit=False, argv=sys.argv[:1],
                           testRunner=unittest.TextTestRunner(stream=stream,
                                                              verbosity=2)).result
    failed = not result.wasSuccessful()
    # any rank failing must take the job down, or the survivors hang
    if HAS_MPI and SIZE > 1 and COMM.allreduce(int(failed)) > 0:
        if failed:
            print(f'[rank {RANK}] failed', file=sys.stderr, flush=True)
        COMM.Barrier()
        if RANK == 0:
            sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
    sys.exit(1 if failed else 0)
