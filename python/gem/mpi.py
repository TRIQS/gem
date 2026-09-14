###########################################
# Shared MPI plumbing
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
###########################################
'''Communicator resolution and reduction helpers shared by the solvers and the
lattice.

Both parallel layers (the sector distribution in ``SimpleED`` and the k-point
distribution in ``Lattice``) resolve their communicator through
:func:`resolve_comm`, so that a run without mpi4py, or one that opts out with
``use_mpi=False``, takes the same code path with a single-rank stand-in instead
of branching on ``mpi_size``.
'''
import numpy as np

try:
    # importing mpi4py calls MPI_Init; fall back to the serial path if it is
    # not installed or the MPI runtime refuses to start
    from mpi4py import MPI
    HAS_MPI = True
except Exception:
    MPI = None
    HAS_MPI = False


class SerialComm:
    '''Minimal stand-in for an MPI communicator used when mpi4py is absent.

    Every collective is the identity on a single rank, so the callers need no
    ``if self.mpi_size > 1`` branches.
    '''
    rank = 0
    size = 1

    def Get_rank(self):                     return 0
    def Get_size(self):                     return 1
    def allreduce(self, value, op=None):    return value
    def allgather(self, value):             return [value]
    def gather(self, value, root=0):        return [value]
    def bcast(self, value, root=0):         return value
    def Barrier(self):                      pass

    def Allreduce(self, sendbuf, recvbuf, op=None):
        '''Buffer allreduce; in-place on a single rank means doing nothing.'''
        if sendbuf is not None and not (HAS_MPI and sendbuf is MPI.IN_PLACE):
            recvbuf[...] = sendbuf


MPI_SUM = MPI.SUM if HAS_MPI else None


def resolve_comm(comm, use_mpi=True):
    '''Pick the communicator: explicit one, else COMM_WORLD, else serial.'''
    if comm is not None:
        return comm
    if use_mpi and HAS_MPI:
        return MPI.COMM_WORLD
    return SerialComm()


def split_range(n, rank, size):
    '''Split ``range(n)`` into ``size`` contiguous near-equal blocks.

    Returns the ``[start, stop)`` half-open bounds of the block owned by
    ``rank``. The first ``n % size`` blocks get one extra element, and blocks
    are empty when ``size > n``.
    '''
    base, rest = divmod(n, size)
    start = rank * base + min(rank, rest)
    stop  = start + base + (1 if rank < rest else 0)
    return start, stop


def allreduce_array(comm, arr):
    '''Sum ``arr`` over all ranks in place, avoiding the pickling path.

    ``arr`` must be a contiguous numpy array with the same shape and dtype on
    every rank. Returns ``arr`` for convenience.
    '''
    if comm.Get_size() == 1:
        return arr
    comm.Allreduce(MPI.IN_PLACE, arr, op=MPI_SUM)
    return arr


def allreduce_scalar(comm, value):
    '''Sum a python/numpy scalar over all ranks.'''
    if comm.Get_size() == 1:
        return value
    return comm.allreduce(value, op=MPI_SUM)
