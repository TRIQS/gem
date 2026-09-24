#######################################################
# Simple Exact Diagonalization solver
# Author: Tsung-Han Lee, Samuele Giuli, Olivier Gingras
# Email:  henhans74716@gmail.com, samuele.giuli@gmail.com
#######################################################

import warnings
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import numpy as np
from numba import jit
import h5py

from math import factorial, comb
from itertools import combinations

from .gem_solver import gemSolver
from .utilities.simple_ed_matvec import (
    EmbeddingHamiltonian, accum_denmat, accum_docc, build_index_table,
    cid_cj_target, docc_target, find_count, lookup, one_body_terms,
    two_body_target, two_body_terms)

from ..mpi import MPI, MPI_SUM as _MPI_SUM, resolve_comm

# List of what can be passed via solver_params:
# spin_pen : Coupling of (\hat{S})^2 to enforce spin singlet
# sx_pen   : Coupling of (\hat{S}_x)^2 to unfavor magnetization in X direction
# sy_pen   : Coupling of (\hat{S}_y)^2 to unfavor magnetization in Y direction
# sz_pen   : Coupling of (\hat{S}_z)^2 to unfavor magnetization in Z direction
# which    : Parameter of scipy.sparse.linalg
# tol      : Parameter of scipy.sparse.linalg
# num_eig  : Number of eigenvalues to compute. If absent (or None), the ground
#            state only at T=0 and the full spectrum at T>0.
# dense_cutoff : Sectors smaller than this are diagonalized fully with
#            scipy.linalg.eigh, the larger ones with scipy.sparse.linalg.eigsh
#            (default 4000).
# bw_cutoff : Smallest Boltzmann weight kept in the partition function at T>0
#            (default 1e-8).
# matrix_free : Never store the Hamiltonian, the c^dag_i c_j operators or the
#            spin operators; apply them on the fly instead (default False).
#            Memory per sector drops from O(dim*norb^2) to O(dim), at a few
#            times the cost per matvec. Requires num_eig at T>0 for any sector
#            reaching dense_cutoff, and does not support the spin penalties.
# mf_lookup_max_norb : In matrix-free mode, build a direct Fock-state lookup
#            table of 2^norb entries when norb is at most this (default 22),
#            otherwise fall back to a binary search per applied term.

class SimpleED(gemSolver):
    '''
    Simple exact diagonalization solver for a general embedding Hamiltonian.

    Symmetry sectors are controlled by ``use_Ntot`` / ``use_Sz`` (which
    symmetries to exploit) and ``N_sector`` / ``Sz_sector`` (which sector(s)
    to include):

    * ``N_sector = <int>``  — solve only that particle-number sector.
    * ``N_sector = None``   — solve *all* particle-number sectors.
    * ``Sz_sector = <int>`` — solve only that Sz sector.
    * ``Sz_sector = None``  — solve *all* Sz sectors.

    The Hamiltonian is built and diagonalised independently in each sector.
    Expectation values (density matrix, energies) are obtained as
    thermal/degeneracy-weighted averages across all included sectors.

    **MPI.** The sectors are distributed over the ranks of ``comm`` (by default
    ``MPI.COMM_WORLD`` when mpi4py is installed, otherwise a serial stand-in).
    Each rank builds the basis and *all* operators only for the sectors it owns,
    diagonalises them, and the global ground-state energy, partition function
    and thermal expectation values are obtained by reduction over the ranks.
    ``self.my_sectors`` holds the global indices owned by this rank; every
    per-sector list (``basis_list``, ``evals_list``, ``bw_per_sector``, ...) is
    indexed by the *local* position inside it. Reduced quantities (``gs_ene``,
    ``Zpart``, ``Tstates``, ``deg``, ``dm``, the energies, ``gs_wf``) are
    identical on every rank.

    ``build_Hemb``, ``solve_Hemb``, ``calc_density_matrix``, ``compute_E1loc``,
    ``compute_E2loc`` and ``calc_double_occ`` are **collective**: every rank of
    ``comm`` must call them, in the same order. Calling one of them inside an
    ``if rank == 0:`` block deadlocks. ``h5write_gs`` is the exception, it
    writes from rank 0 and is a no-op elsewhere.

    **Matrix-free.** With ``solver_params['matrix_free'] = True`` nothing is
    stored per sector but the basis: the Hamiltonian is handed to ARPACK as a
    ``LinearOperator`` and the ``c^dag_i c_j`` operators are applied on the fly,
    so the memory goes from ``O(dim*norb^2)`` to ``O(dim)`` and the sector size
    stops being limited by memory. It costs a few times more per matvec, it
    needs ``num_eig`` at ``T>0`` once a sector reaches ``dense_cutoff`` (the
    full-spectrum fallback cannot work without the matrix), and it does not
    support the spin penalties. Results are otherwise identical to the stored
    path, sign for sign.

    **Precision.** ``dtype`` is a request, resolved into ``self.data_type`` at
    every ``build_Hemb``. A real one is honoured, and the whole solve then stays
    real (operators, diagonalisation, eigenvectors, density matrix, at about
    half the cost of the complex path in the stored branch), unless ``eloc``,
    ``D``, ``Lambda_c``, ``V2E`` or the ``sy_pen`` penalty is complex — then the
    solve is upcast to ``complex128`` and a warning is issued once.
    '''

    def __init__(self,
                 norb,
                 use_Ntot=False,
                 use_Sz=False,
                 dtype=np.complex128,
                 N_sector=None,
                 Sz_sector=None,
                 solver_params=None,
                 comm=None,
                 **kwargs):
        '''
        Initialize the solver with the given number of orbitals and symmetry settings.

        :param norb: int. Total number of spin-orbital levels in the embedding Hamiltonian.
        :param use_Ntot: bool, optional. Whether to exploit particle-number conservation (default: False).
        :param use_Sz: bool, optional. Whether to exploit Sz conservation (default: False).
        :param dtype: Working precision (default: ``np.complex128``). A real
            type is honoured only when the embedding problem really is real,
            otherwise the solve is upcast to ``complex128`` with a warning; see
            the class docstring.
        :param N_sector: int, optional. Particle-number sector to solve if use_Ntot is True (default: None).
        :param Sz_sector: int, optional. Sz sector to solve if use_Sz is True (default: None).
        :param solver_params: dict, optional. Parameters for the solver (default: None).
        :param comm: mpi4py communicator, optional. The sectors are distributed
            over its ranks. Defaults to ``MPI.COMM_WORLD`` when mpi4py is
            available (set ``solver_params['use_mpi'] = False`` to force the
            serial path) and to a serial stand-in otherwise.
        '''
        # backward compat: Nparticle into N_sector, will be removed after hearing from others
        if 'Nparticle' in kwargs:
            warnings.warn("Nparticle is deprecated, use N_sector instead.",
                          DeprecationWarning, stacklevel=2)
            N_sector = kwargs.pop('Nparticle')
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        super().__init__(solver_params=solver_params, solver_type="SimpleED")
        self.norb      = norb
        self.use_Ntot  = use_Ntot
        self.use_Sz    = use_Sz
        self.N_sector  = N_sector
        self.Sz_sector = Sz_sector

        # resolved at every build_Hemb: a real dtype is only a request
        self.data_type     = np.dtype(dtype)
        self._dtype_warned = False

        # matrix-free: apply the Hamiltonian and the c^dag_i c_j operators on
        # the fly instead of storing them
        self.matrix_free = self.solver_params.get('matrix_free', False)

        # MPI: the sector list is global and identical on every rank, the work
        # on it is not
        self.comm     = resolve_comm(comm, self.solver_params.get('use_mpi', True))
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()

        # determine the list of (N, Sz) sector labels to solve
        self.sectors     = self._get_sectors()
        self.sector_dims = [self._sector_dim(N, Sz) for (N, Sz) in self.sectors]

        # distribute them: sector_owner[s] is the rank owning global sector s,
        # my_sectors are the global indices this rank is responsible for
        self.sector_owner = self._distribute_sectors()
        self.my_sectors   = [s for s, r in enumerate(self.sector_owner)
                             if r == self.mpi_rank]
        self.nloc         = len(self.my_sectors)

        if self.mpi_rank == 0:
            for s, (N, Sz) in enumerate(self.sectors):
                tag = f'  [rank {self.sector_owner[s]}]' if self.mpi_size > 1 else ''
                print(f'Sector {s} (N={N}, Sz={Sz}): '
                      f'basis size = {self.sector_dims[s]}{tag}')

        # per-sector data structures, indexed by local position in my_sectors
        self.basis_list     = []
        self.hsize_list     = []
        self.denmat_op_list = []
        self.S2_list        = []
        self.Sz_list        = []
        self.Sx_list        = []
        self.Sy_list        = []
        self.index_list     = []
        self.Hone_list      = [None] * self.nloc
        self.Htwo_list      = [None] * self.nloc
        self.V2E            = None  # cached interaction tensor for rebuild check
        self.prev_gs_list   = [None] * self.nloc
        self.tb_terms       = None  # cached two-body term list, matrix-free only

        lookup_max = self.solver_params.get('mf_lookup_max_norb', 22)

        for s in self.my_sectors:
            N, Sz = self.sectors[s]
            basis_s = self._build_basis(N, Sz)
            hsize_s = len(basis_s)
            self.basis_list.append(basis_s)
            self.hsize_list.append(hsize_s)

            if self.matrix_free:
                # the operators are what costs O(dim*norb^2); keep only the
                # state-lookup table, and keep the lists aligned with None
                self.index_list.append(build_index_table(basis_s, self.norb,
                                                         max_norb=lookup_max))
                self.denmat_op_list.append(None)
                self.S2_list.append(None)
                self.Sz_list.append(None)
                self.Sx_list.append(None)
                self.Sy_list.append(None)
                continue

            self.index_list.append(np.empty(0, dtype=np.int64))

            denmat_op_s = self._build_denmat_op(basis_s, hsize_s)
            self.denmat_op_list.append(denmat_op_s)

            S2_s, Sz_s, Sx_s, Sy_s = self._build_S2_op(basis_s, hsize_s, denmat_op_s)
            self.S2_list.append(S2_s)
            self.Sz_list.append(Sz_s)
            self.Sx_list.append(Sx_s)
            self.Sy_list.append(Sy_s)

        # single-sector aliases for backward compatibility
        if len(self.sectors) == 1 and self.nloc == 1:
            self._set_single_sector_aliases(0)

    def _tag(self):
        '''Rank prefix for printouts, empty when running serially.'''
        return f'[rank {self.mpi_rank}] ' if self.mpi_size > 1 else ''

# -- Mandatory functions --

    def build_Hemb(self, D, eloc, Lambdac, V2E, mu=0.0, debug=False, verbose=0,
                   spin_pen=None, sz_pen=None, sx_pen=None, sy_pen=None):
        '''Build the embedding Hamiltonian in every sector owned by this rank.

        In matrix-free mode nothing is built: the one-body and two-body
        coefficients are flattened into term lists and each sector gets an
        ``EmbeddingHamiltonian`` that applies them on the fly. With
        ``debug=True`` the returned ``Ham_list`` then holds those operators
        rather than sparse matrices.

        Collective: must be called by every rank.

        :param D:           array. D matrix of the embedded Hamiltonian.
        :param eloc:        array. Local part of the Hamiltonian.
        :param Lambdac:     array. Lambda_c matrix of the embedded Hamiltonian.
        :param V2E:         array. Two-particle interaction on the impurity degrees of freedom.
        :param mu:          float. Chemical potential (default 0).
        :param debug:       bool. Flag for debugging (default False). If True, returns the list of one-body terms.
        :param verbose:     int. Level of verbosity (default 0).
        :param spin_pen:    float. Penalty for states with non-zero <S^2> (default 0).
        :param sz_pen:      float. Penalty for states with non-zero <S_z^2> (default 0).
        :param sx_pen:      float. Penalty for states with non-zero <S_x^2> (default 0).
        :param sy_pen:      float. Penalty for states with non-zero <S_y^2> (default 0).
        '''
        #Try to get penalties from solver_params if not given explicitly
        spin_pen = self.solver_params.get('spin_pen', 0) if spin_pen is None else spin_pen
        sz_pen   = self.solver_params.get('sz_pen',   0) if sz_pen   is None else sz_pen
        sx_pen   = self.solver_params.get('sx_pen',   0) if sx_pen   is None else sx_pen
        sy_pen   = self.solver_params.get('sy_pen',   0) if sy_pen   is None else sy_pen
        if(spin_pen != 0.0 or sz_pen != 0.0 or sx_pen != 0.0 or sy_pen != 0.0):
            if self.matrix_free:
                raise NotImplementedError(
                    "matrix_free=True does not support the spin penalties: S2, "
                    "Sz, Sx and Sy are not built. Drop spin_pen/sz_pen/sx_pen/"
                    "sy_pen, or use the stored path.")
            if self.mpi_rank == 0:
                warnings.warn(
                    "A spin penalty had been passed. Remember to use this ONLY from T=0 calculations," \
                    "otherwise the boltzmann weights will be wrong at T>0.")

        if(verbose >1): print('build one-body')
        self.build_h1e(eloc, D, Lambdac, mu, verbose=verbose)

        # a real data_type is a request: honour it only if nothing here is
        # complex, otherwise upcast the whole solve and say so once
        want_real = not np.issubdtype(self.data_type, np.complexfloating)
        real = (want_real and np.abs(self.h1e.imag).max() < 1e-12
                and np.abs(np.imag(V2E)).max() < 1e-12 and not sy_pen)
        if want_real and not real:
            if not self._dtype_warned and self.mpi_rank == 0:
                warnings.warn(
                    f"dtype={self.data_type} was requested but eloc/D/Lambda_c, "
                    "V2E or the Sy penalty is complex: solving in complex128.")
                self._dtype_warned = True
            self.V2E = None        # forces the two-body rebuild at the new dtype
            self.data_type = np.dtype(np.complex128)
        if real:
            self.h1e = self.h1e.real.copy()
            V2E      = np.real(V2E)

        # rebuild two-body only when V2E changes (shared across sectors)
        if self.matrix_free:
            rebuild_two = (self.V2E is None) or np.any(V2E != self.V2E) \
                          or self.tb_terms is None
        else:
            rebuild_two = (self.V2E is None) or np.any(V2E != self.V2E) \
                          or any(H is None for H in self.Htwo_list)
        if rebuild_two:
            if(verbose > 1): print('build two-body')
            self.V2E = V2E.copy()

        if self.matrix_free:
            if rebuild_two:
                self.tb_terms = two_body_terms(V2E)
            self.ob_terms = one_body_terms(self.h1e)
            self.Ham_list = [
                EmbeddingHamiltonian(self.basis_list[s], self.index_list[s],
                                     self.norb, self.ob_terms, self.tb_terms)
                for s in range(self.nloc)]
            if len(self.sectors) == 1 and self.nloc == 1:
                self.Ham = self.Ham_list[0]
            if debug:
                return self.Ham_list
            return

        if(verbose > 1): print('one-body + two-body')
        self.Ham_list = []
        for s in range(self.nloc):
            basis_s     = self.basis_list[s]
            hsize_s     = self.hsize_list[s]
            denmat_op_s = self.denmat_op_list[s]

            Hone_s = self._build_one_body(self.h1e, hsize_s, denmat_op_s)
            self.Hone_list[s] = Hone_s

            if rebuild_two:
                self.Htwo_list[s] = self._build_two_body(V2E, basis_s, hsize_s)

            # add only the penalties switched on: a zero coefficient would
            # still drag its operator's dtype into Ham (Sy is complex)
            Ham_s = Hone_s + self.Htwo_list[s]
            if spin_pen: Ham_s = Ham_s + spin_pen * self.S2_list[s]
            for pen, op in ((sz_pen, self.Sz_list[s]), (sx_pen, self.Sx_list[s]),
                            (sy_pen, self.Sy_list[s])):
                if pen: Ham_s = Ham_s + pen * op.dot(op)
            self.Ham_list.append(Ham_s)


        # single-sector backward compat
        if len(self.sectors) == 1 and self.nloc == 1:
            self.Hone = self.Hone_list[0]
            self.Htwo = self.Htwo_list[0]
            self.Ham  = self.Ham_list[0]

        if debug:
            return self.Ham_list

    def solve_Hemb(self, num_eig=None, which=None, tol=None, dense_cutoff=None,
                   bw_cutoff=None, verbose=0, T=0.0):
        '''
        Solve for the ground state, and some excited states if not all, of the embedded Hamiltonian of this fragment.
        For thermal calculation, the global partition function is required.

        :param num_eig: int, optional. Number of eigenvalues to compute. Read
            from solver_params when not given; if it is None there too, only
            the ground state is computed at T=0 while at T>0 the full
            Hamiltonian of each sector is diagonalised.
        :param dense_cutoff: int, optional. Sectors of dimension smaller than
            this are diagonalised fully (eigh), the larger ones partially
            (eigsh). Read from solver_params when not given, default 4000.
        :param bw_cutoff: float, optional. At T>0, states whose Boltzmann weight
            exp(-beta*(E-gs_ene)) falls below this are dropped from the
            partition function. Read from solver_params when not given,
            default 1e-8.
        :param verbose:     int. Level of verbosity (default 0).
        :param T:           float. Electronic temperature (default 0).

        Collective: every rank diagonalises its own sectors, then ``gs_ene``,
        ``Zpart``, ``Tstates``, ``deg``, ``bw_list`` and the returned
        ``(gs_wf, gs_ene)`` are reduced/broadcast and hold globally.
        '''
        if T > 0.0 and self.mpi_rank == 0 and \
           ((self.use_Ntot and self.N_sector is not None) or
            (self.use_Sz   and self.Sz_sector is not None)):
            warnings.warn(
                "A restricted symmetry sector is selected: the partition "
                "function may be incomplete at T>0.")

        which   = self.solver_params.get('which', 'SA') if which is None else which
        tol     = self.solver_params.get('tol',   1e-8) if tol   is None else tol
        num_eig = self.solver_params.get('num_eig') if num_eig is None else num_eig
        dense_cutoff = (self.solver_params.get('dense_cutoff', 4000)
                        if dense_cutoff is None else dense_cutoff)
        bw_cutoff    = (self.solver_params.get('bw_cutoff', 1e-8)
                        if bw_cutoff is None else bw_cutoff)

        # num_eig still unset: ground state only at T=0, full spectrum at T>0
        full_diag = (num_eig is None) and (T > 0.0)
        k_eig     = 1 if num_eig is None else num_eig

        # diagonalise the sectors owned by this rank
        self.evals_list = []
        self.evecs_list = []

        for s, Ham_s in enumerate(self.Ham_list):
            hsize_s = self.hsize_list[s]
            if(verbose > 0): print(f'{self._tag()}Sector {self.my_sectors[s]}: '
                                   f'diagonalising (dim={hsize_s})')
            if hsize_s < dense_cutoff:
                vals, vecs = eigh(Ham_s.to_dense() if self.matrix_free
                                  else Ham_s.toarray())
            elif full_diag:
                if self.matrix_free:
                    raise ValueError(
                        f"Sector {self.my_sectors[s]} has dimension {hsize_s} "
                        f"(>= dense_cutoff={dense_cutoff}) and T={T} > 0 with "
                        "num_eig unset: the full spectrum cannot be obtained "
                        "matrix-free. Set num_eig large enough to cover the "
                        "thermal window (bw_cutoff), or raise dense_cutoff.")
                vals, vecs = eigh(Ham_s.toarray())
            else:
                v0 = self.prev_gs_list[s] if T == 0.0 else None
                vals, vecs = eigsh(Ham_s, k=k_eig, which=which, tol=tol, v0=v0)
            so = vals.argsort()
            self.evals_list.append(vals[so])
            self.evecs_list.append(vecs[:, so])
            self.prev_gs_list[s] = vecs[:, so[0]].copy()

        # global ground-state energy: reduce the per-rank minima. allgather
        # rather than allreduce(MIN) so that every rank also learns which rank
        # holds the ground state (lowest rank wins a tie, so it is unique).
        local_min   = min((evals[0] for evals in self.evals_list), default=np.inf)
        rank_min    = self.comm.allgather(float(local_min))
        self.gs_rank = int(np.argmin(rank_min))
        self.gs_ene  = float(rank_min[self.gs_rank])

        # per-sector Boltzmann weights, then reduce the partition function
        self.bw_per_sector = []
        Zloc, Tloc, degloc = 0.0, 0, 0

        if T > 0.0:
            beta = 1.0 / T
            for s, evals_s in enumerate(self.evals_list):
                bw_s = []
                for eit in evals_s:
                    bw = float(np.exp(-beta * (eit - self.gs_ene)))
                    if bw > bw_cutoff:
                        bw_s.append(bw)
                        Zloc += bw
                        Tloc += 1
                    else:
                        break   # eigenvalues are sorted; remaining are smaller
                if( (self.hsize_list[s]>dense_cutoff) and (len(bw_s)==k_eig) and (not full_diag) ):
                    Nsec, Szsec = self.sectors[s]
                    warnings.warn(f"WARNING: For sector (N,Sz)=({Nsec},{Szsec}) the number of thermal states is equal to the ARPACK cutoff. The last Boltzmann weight is: {bw_s[-1]}")
                self.bw_per_sector.append(bw_s)
        else:
            if( verbose > 0 ):print(f'{self._tag()}Building GS partition function across sectors')
            for evals_s in self.evals_list:
                bw_s = []
                for eit in evals_s:
                    if abs(eit - self.gs_ene) < 1e-4:
                        bw_s.append(1.0)
                        Zloc   += 1.0
                        Tloc   += 1
                        degloc += 1
                    else:
                        break
                self.bw_per_sector.append(bw_s)

        self.Zpart_local = Zloc
        self.Zpart   = self.comm.allreduce(Zloc,   op=_MPI_SUM)
        self.Tstates = self.comm.allreduce(Tloc,   op=_MPI_SUM)
        self.deg     = self.comm.allreduce(degloc, op=_MPI_SUM)

        # trim per-sector evecs/evals to thermal states only
        for s in range(self.nloc):
            n_s = len(self.bw_per_sector[s])
            self.evals_list[s] = self.evals_list[s][:n_s]
            self.evecs_list[s] = self.evecs_list[s][:, :n_s]

        # flattening the bw_list over all ranks, now not ordered
        self.bw_list = [bw for chunk in self.comm.allgather(
                            [bw for bw_s in self.bw_per_sector for bw in bw_s])
                        for bw in chunk]

        # ground-state wavefunction (sector with lowest energy): it lives on
        # gs_rank, broadcast it so that gs_wf/gs_sector are defined everywhere
        self.gs_sector_local = None
        if self.mpi_rank == self.gs_rank:
            self.gs_sector_local = int(np.argmin(
                [evals[0] if len(evals) > 0 else np.inf
                 for evals in self.evals_list]))
            payload = (self.my_sectors[self.gs_sector_local],
                       self.evecs_list[self.gs_sector_local][:, 0].copy())
        else:
            payload = None
        self.gs_sector, self.gs_wf = self.comm.bcast(payload, root=self.gs_rank)

        # verbose: print quantum numbers for every included state
        if verbose > 1:
            # S2/Sz are not available matrix-free: print the rest of the table
            rows = []
            for s in range(self.nloc):
                for n in range(len(self.bw_per_sector[s])):
                    if self.matrix_free:
                        S2v = Szv = complex(np.nan)
                    else:
                        vec = self.evecs_list[s][:, n]
                        S2v = complex(vec.conj() @ self.S2_list[s].dot(vec))
                        Szv = complex(vec.conj() @ self.Sz_list[s].dot(vec))
                    rows.append((self.my_sectors[s], n, self.evals_list[s][n],
                                 S2v, Szv, self.bw_per_sector[s][n]))
            gathered = self.comm.gather(rows, root=0)
            if self.mpi_rank == 0:
                if self.matrix_free:
                    print('# s\tn\tEnergy\t\t\tBoltzmann   (S2/Sz unavailable '
                          'in matrix-free mode)')
                else:
                    print('# s\tn\tEnergy\t\t\tS2\t\tSz\t\tBoltzmann')
                for (s, n, ene, S2v, Szv, bw) in sorted(
                        [r for chunk in gathered for r in chunk],
                        key=lambda r: (r[0], r[1])):
                    if self.matrix_free:
                        print(f"  {s}\t{n}\t{ene:.10e}\t{bw:.6f}")
                    else:
                        print(f"  {s}\t{n}\t{ene:.10e}\t"
                              f"{S2v.real:.3f}+{S2v.imag:.1e}j\t"
                              f"{Szv.real:.3f}+{Szv.imag:.1e}j\t"
                              f"{bw:.6f}")
                print(f'deg={self.deg}  Zpart={self.Zpart}  Tstates={self.Tstates}')

        # single-sector backward compat
        if len(self.sectors) == 1 and self.nloc == 1:
            self.evals = self.evals_list[0]
            self.evecs = self.evecs_list[0]

        return self.gs_wf, self.gs_ene

    @staticmethod
    def _weighted_trace(op, U, bw):
        '''Boltzmann-weighted trace ``sum_n bw[n] <n|op|n>`` over the columns of U.

        Identical to ``np.trace(U.conj().T @ op @ U @ np.diag(bw))``, but the
        off-diagonal elements of ``U^dag op U`` are never formed and ``diag(bw)``
        is never built: O(nnz*nst + dim*nst) instead of O(dim*nst^2 + nst^3).
        '''
        return np.einsum('kn,kn,n->', U.conj(), op @ U, bw)

    def _mf_denmat_block(self, s, nmax, U_s, bw_s):
        '''Matrix-free ``sum_n bw[n] <n| c^dag_i c_j |n>`` for ``i, j < nmax``.

        Same result as calling ``_weighted_trace`` on every ``denmat_op[(i,j)]``
        of local sector ``s``, without those operators existing.
        '''
        blk = np.zeros((nmax, nmax), dtype=self.data_type)
        accum_denmat(self.basis_list[s], self.index_list[s],
                     2**(self.norb - 1), self.norb, nmax,
                     np.ascontiguousarray(U_s, dtype=self.data_type),
                     np.ascontiguousarray(bw_s, dtype=np.float64), blk)
        return blk

    def calc_density_matrix(self):
        '''
        Compute the one-body density matrix as a thermal/degeneracy average
        across all active sectors, reduced over the MPI ranks.

        Collective: must be called by every rank.

        Return:
            array. One-body density-matrix.
        '''
        dm = np.zeros((self.norb, self.norb), dtype=self.data_type)

        for s in range(self.nloc):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            U_s = self.evecs_list[s]   # (hsize_s, n_states_s)
            if self.matrix_free:
                dm += self._mf_denmat_block(s, self.norb, U_s, bw_s)
                continue
            for i in range(self.norb):
                for j in range(self.norb):
                    dm[i, j] += self._weighted_trace(
                        self.denmat_op_list[s][(i, j)], U_s, bw_s)

        dm = self.comm.allreduce(dm, op=_MPI_SUM)
        dm /= self.Zpart
        self.dm = dm
        return dm

    def compute_E1loc(self, nimp):
        '''
        One-body local energy (impurity block) averaged across sectors.
        Compute the one-body local energy (impurity block) averaged across sectors.

        Collective: must be called by every rank.

        :param nimp:    int. Number of impurity orbitals in the block Hamiltonian.

        Return:
            float. One-body local energy.
        '''
        result = 0.0
        for s in range(self.nloc):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            U_s = self.evecs_list[s]
            if self.matrix_free:
                dm_imp = self._mf_denmat_block(s, nimp, U_s, bw_s)
            else:
                dm_imp = np.zeros((nimp, nimp), dtype=self.data_type)
                for i in range(nimp):
                    for j in range(nimp):
                        dm_imp[i, j] = self._weighted_trace(
                            self.denmat_op_list[s][(i, j)], U_s, bw_s)
            result += np.trace(self.h1e[:nimp, :nimp] @ dm_imp.T)

        return self.comm.allreduce(result, op=_MPI_SUM) / self.Zpart

    def compute_E2loc(self):
        '''
        Two-body local energy averaged across sectors.
        Compute the two-body local energy averaged across sectors.

        Collective: must be called by every rank.

        Return:
            float. Two-body local energy.
        '''
        result = 0.0
        for s in range(self.nloc):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            U_s = self.evecs_list[s]
            if self.matrix_free:
                # sum_n bw[n] <n|Htwo|n>, one matrix-free application per state
                Ham_s = self.Ham_list[s]
                for n in range(len(bw_s)):
                    v = U_s[:, n]
                    result += bw_s[n] * np.vdot(v, Ham_s.apply_two_body(v))
                continue
            result += self._weighted_trace(self.Htwo_list[s], U_s, bw_s)

        return self.comm.allreduce(result, op=_MPI_SUM) / self.Zpart


# -- Utility functions for sectors --

    def _sector_dim(self, N, Sz):
        '''Dimension of sector (N, Sz), counted without building the basis.'''
        norb, n_half = self.norb, self.norb // 2
        if N is None and Sz is None:
            return 2**norb
        if Sz is None:
            return comb(norb, N)
        nup = (N + round(2*Sz)) // 2
        ndw = N - nup
        return comb(n_half, nup) * comb(n_half, ndw)

    def _distribute_sectors(self):
        '''Assign each global sector to a rank.

        Longest-processing-time-first greedy packing: the heaviest sector goes
        to the least loaded rank. The cost of a sector is taken as
        ``dim**mpi_weight_exp`` (default exponent 3, i.e. dense diagonalisation
        dominates, or 1 in matrix-free mode, where the cost is a number of
        matvecs).
        '''
        owner = [0] * len(self.sectors)
        if self.mpi_size == 1:
            return owner
        exp     = self.solver_params.get('mpi_weight_exp', 1 if self.matrix_free else 3)
        weights = [d**exp for d in self.sector_dims]
        load    = [0] * self.mpi_size
        for s in sorted(range(len(self.sectors)), key=lambda s: (-weights[s], s)):
            r        = load.index(min(load))   # lowest rank on a tie
            owner[s] = r
            load[r] += weights[s]
        return owner

    def _get_sectors(self):
        '''
        Return the list of (N, Sz) pairs to solve given the symmetry flags.

        Return:
            list of tuples. List of (N, Sz) sectors given the symmetry constraints.
        '''
        norb   = self.norb
        n_half = norb // 2

        if not self.use_Ntot and not self.use_Sz:
            return [(None, None)]

        if self.use_Ntot and not self.use_Sz:
            if self.N_sector is not None:
                return [(self.N_sector, None)]
            return [(n, None) for n in range(norb + 1)]

        if self.use_Ntot and self.use_Sz:
            N, Sz = self.N_sector, self.Sz_sector
            if N is not None and Sz is not None:
                return [(N, Sz)]
            if N is not None:           # all valid Sz for given N
                sectors = []
                for nup in range(max(0, N - n_half), min(N, n_half) + 1):
                    sectors.append((N, (nup - (N - nup)) / 2.0))
                return sectors
            if Sz is not None:          # all valid N for given Sz
                return self._all_N_for_Sz(Sz, norb)
            # both None: all (N, Sz) sectors
            sectors = []
            for n in range(norb + 1):
                for nup in range(max(0, n - n_half), min(n, n_half) + 1):
                    sectors.append((n, (nup - (n - nup)) / 2.0))
            return sectors

        # use_Sz only (useful only for superconductivity)
        if not self.use_Ntot and self.use_Sz:
            if self.Sz_sector is not None:
                return self._all_N_for_Sz(self.Sz_sector, norb)
            return [(None, None)]

    @staticmethod
    def _all_N_for_Sz(Sz, norb):
        '''
        Give all different N sectors for a given SZ.

        :param Sz:      int. Value of Sz.
        :param norb:    int. Number of total orbitals.

        Return:
            list. List of sectors (N) for a given Sz.
        '''
        n_half = norb // 2

        sz2 = round(2 * Sz)
        sectors = []
        for n in range(norb + 1):
            if (n + sz2) % 2 != 0:
                continue
            nup = (n + sz2) // 2
            ndw = (n - sz2) // 2
            if 0 <= nup <= n_half and 0 <= ndw <= n_half:
                sectors.append((n, Sz))
        return sectors

    def _build_basis(self, N, Sz):
        '''
        Build the Fock-state basis for sector (N, Sz).

        Return:
           array. List of states for a given (N, Sz) sector.
        '''
        norb = self.norb
        if not self.use_Ntot and not self.use_Sz:
            return np.arange(2**norb)
        if self.use_Ntot and not self.use_Sz:
            return table_ep(norb, N)
        if self.use_Ntot and self.use_Sz:
            return table_es(norb, N, Sz)
        # use_Sz only: need both N and Sz
        return table_es(norb, N, Sz)

    def _set_single_sector_aliases(self, s):
        '''
        Expose per-sector attributes as flat attributes for single-sector use.

        :param s:   int. Sector.
        '''
        self.basis     = self.basis_list[s]
        self.hsize     = self.hsize_list[s]
        if self.matrix_free:
            # denmat_op/S2/Sz/Sx/Sy do not exist in this mode; leave the
            # aliases unset rather than publishing None
            return
        self.denmat_op = self.denmat_op_list[s]
        self.S2        = self.S2_list[s]
        self.Sz        = self.Sz_list[s]
        self.Sx        = self.Sx_list[s]
        self.Sy        = self.Sy_list[s]


# -- operator builders (take basis/hsize explicitly) --

    def _build_denmat_op(self, basis, hsize):
        '''
        Build c^dag_i c_j operators in the given basis.
        '''
        denmat_op = {}
        bit_max = 2**(self.norb - 1)
        for i in range(self.norb):
            for j in range(self.norb):
                row_ind, col_ind, data = build_cid_cj_csc(
                    i, j, basis, bit_max, self.norb, debug=False)
                denmat_op[(i, j)] = csc_matrix(
                    (data, (row_ind, col_ind)),
                    shape=(hsize, hsize), dtype=self.data_type)
        return denmat_op

    def _build_S2_op(self, basis, hsize, denmat_op):
        '''
        Build S², Sz, Sx, Sy in the given basis.
        '''
        Sp = csc_matrix((hsize, hsize), dtype=self.data_type)
        Sm = csc_matrix((hsize, hsize), dtype=self.data_type)
        Sz = csc_matrix((hsize, hsize), dtype=self.data_type)
        for i in range(self.norb // 2):
            Sp += denmat_op[(2*i,   2*i+1)]
            Sm += denmat_op[(2*i+1, 2*i  )]
            Sz += 0.5*denmat_op[(2*i, 2*i)] - 0.5*denmat_op[(2*i+1, 2*i+1)]
        SmSp = csc_matrix((hsize, hsize), dtype=self.data_type)
        for i in range(self.norb // 2):
            SmSp += denmat_op[(2*i+1, 2*i+1)]
            for j in range(self.norb // 2):
                SmSp -= denmat_op[(2*i+1, 2*j+1)].dot(
                    denmat_op[(2*j, 2*i)])

        S2 = SmSp + Sz.dot(Sz) + Sz
        Sx = 0.5*(Sp + Sm)
        Sy = 0.5*(Sp - Sm) / 1j
        return S2, Sz, Sx, Sy

    def _build_one_body(self, H1E, hsize, denmat_op):
        '''
        Build one-body Hamiltonian block in a given basis.
        '''
        Hone = csc_matrix((hsize, hsize), dtype=self.data_type)
        for i in range(self.norb):
            for j in range(self.norb):
                if np.abs(H1E[i, j]) < 1e-8:
                    continue
                Hone += H1E[i, j] * denmat_op[(i, j)]
        return Hone

    def _build_two_body(self, Umatrix, basis, hsize):
        '''
        Build two-body Hamiltonian block in a given basis.
        '''
        bit_max = 2**(self.norb - 1)
        Htwo = csc_matrix((hsize, hsize), dtype=self.data_type)
        for i in range(Umatrix.shape[0]):
            for j in range(Umatrix.shape[1]):
                for k in range(Umatrix.shape[2]):
                    for l in range(Umatrix.shape[3]):
                        if l == j or i == k or abs(Umatrix[i,j,k,l]) < 1e-8:
                            continue
                        row_ind, col_ind, data = build_two_body_ijkl_csc_2(
                            i, j, k, l, basis, bit_max, self.norb)
                        Htwo += 0.5*Umatrix[i,j,k,l] * csc_matrix(
                            (data, (row_ind, col_ind)),
                            shape=(hsize, hsize), dtype=self.data_type)
        return Htwo



# -- other auxiliary function --

    def build_h1e(self, eloc, D, Lambdac, mu, verbose=0):
        '''
        Build one-body part of the embedded Hamiltonian.
        '''
        self.h1e = np.zeros((self.norb, self.norb), dtype=np.complex128)
        nimp = eloc.shape[0]
        self.h1e[:nimp, :nimp] = eloc - mu*np.eye(nimp)
        self.h1e[:nimp, nimp:] = D.T
        self.h1e[nimp:, nimp:] = -Lambdac
        self.h1e[nimp:, :nimp] = D.conj()
        if verbose > 3:
            print("h1e"); print(self.h1e)


    def _build_docc_op_sector(self, i, s):
        '''Double-occupancy operator in *local* sector s.'''
        basis   = self.basis_list[s]
        hsize   = self.hsize_list[s]
        bit_max = 2**(self.norb - 1)
        index   = self.index_list[s]
        row_ind, col_ind, data = [], [], []
        for bsrid in range(len(basis)):
            bsl, sign = docc_target(i, basis[bsrid], bit_max, self.norb)
            if sign == 0:
                continue
            id_bsl = lookup(bsl, basis, index)
            if id_bsl < 0:
                continue
            row_ind.append(id_bsl); col_ind.append(bsrid); data.append(sign)
        return csc_matrix((data, (row_ind, col_ind)),
                          shape=(hsize, hsize), dtype=self.data_type)

    def calc_double_occ(self, i):
        '''Thermal double occupancy on orbital (i, i+1), averaged over all sectors.

        Collective: must be called by every rank.
        '''
        result = 0.0
        for s in range(self.nloc):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            U_s = self.evecs_list[s]
            if self.matrix_free:
                result += accum_docc(
                    self.basis_list[s], self.index_list[s],
                    2**(self.norb - 1), self.norb, i,
                    np.ascontiguousarray(U_s, dtype=self.data_type),
                    np.ascontiguousarray(bw_s, dtype=np.float64)).real
                continue
            docc_op = self._build_docc_op_sector(i, s)
            result += self._weighted_trace(docc_op, U_s, bw_s).real
        return self.comm.allreduce(result, op=_MPI_SUM) / self.Zpart

    # kept for backward compat (single-sector callers)
    def _reject_if_matrix_free(self, what):
        if self.matrix_free:
            raise NotImplementedError(
                f"{what} stores an operator, which matrix_free=True exists to "
                "avoid. Build the solver without matrix_free to use it.")

    def build_denmat_op(self):
        self._reject_if_matrix_free('build_denmat_op')
        self.denmat_op = self._build_denmat_op(self.basis_list[0], self.hsize_list[0])
        self.denmat_op_list[0] = self.denmat_op

    def build_S2_op(self):
        self._reject_if_matrix_free('build_S2_op')
        S2, Sz, Sx, Sy = self._build_S2_op(
            self.basis_list[0], self.hsize_list[0], self.denmat_op_list[0])
        self.S2_list[0], self.Sz_list[0] = S2, Sz
        self.Sx_list[0], self.Sy_list[0] = Sx, Sy
        self.S2, self.Sz, self.Sx, self.Sy = S2, Sz, Sx, Sy

    def build_one_body(self, H1E):
        self._reject_if_matrix_free('build_one_body')
        self.Hone = self._build_one_body(H1E, self.hsize_list[0], self.denmat_op_list[0])
        self.Hone_list[0] = self.Hone

    def build_two_body(self, Umatrix, debug=False):
        self._reject_if_matrix_free('build_two_body')
        self.Htwo = self._build_two_body(Umatrix, self.basis_list[0], self.hsize_list[0])
        self.Htwo_list[0] = self.Htwo

    def h5write_gs(self, filename, group_path, name):
        # gs_wf is replicated on every rank: only rank 0 touches the file
        if self.mpi_rank != 0:
            return
        with h5py.File(filename, "a") as f:
            if group_path not in f:
                f.create_group(group_path)
            f[group_path][name] = self.gs_wf

    def h5read_state(self, filename, group_path, name):
        with h5py.File(filename, "r") as f:
            return f[group_path][name][:]


# -- binary basis utilities --


def table_ep(nstate, nparticle, dtype=np.int64):
    '''
    Binary basis for fixed particle number (no Sz constraint).
    '''
    result = np.zeros(factorial(nstate) // factorial(nparticle) // factorial(nstate - nparticle),
                      dtype=dtype)
    for i, v in enumerate(combinations(range(nstate), nparticle)):
        basis = 0
        for num in v:
            basis += (1 << num)
        result[i] = basis
    result.sort()
    return result

def table_es(nstate, nparticle, spinz, dtype=np.int64):
    '''
    Binary basis for fixed particle number and Sz.
    '''
    n    = nstate // 2
    nup  = (nparticle + int(2*spinz)) // 2
    ndw  = (nparticle - int(2*spinz)) // 2
    result = np.zeros(
        factorial(n)//factorial(nup)//factorial(n-nup) *
        factorial(n)//factorial(ndw)//factorial(n-ndw),
        dtype=dtype)
    buff_up = list(combinations(range(1, 2*n, 2), nup))
    buff_dw = list(combinations(range(0, 2*n, 2), ndw))
    count = 0
    for vup in buff_up:
        buff = sum(1 << num for num in vup)
        for vdw in buff_dw:
            basis = buff + sum(1 << num for num in vdw)
            result[count] = basis
            count += 1
    result.sort()
    return result


# The bit algebra itself lives in utilities/simple_ed_matvec.py, so that the
# stored-CSC builders below and the matrix-free kernels share one definition of
# the fermionic signs. find_count is re-exported from there for backward
# compatibility.

@jit(nopython=True)
def build_cid_cj_csc(i, j, basis, bit_max, norb, debug=False):
    row_ind, col_ind, data = [], [], []
    index = np.empty(0, dtype=np.int64)
    for bsrid in range(len(basis)):
        bsl, sign = cid_cj_target(i, j, basis[bsrid], bit_max, norb)
        if sign == 0:
            continue
        id_bsl = lookup(bsl, basis, index)
        if id_bsl < 0:
            continue
        row_ind.append(id_bsl); col_ind.append(bsrid); data.append(sign)
    return row_ind, col_ind, data

@jit(nopython=True)
def build_two_body_ijkl_csc_2(i, j, k, l, basis, bit_max, norb, debug=False):
    row_ind, col_ind, data = [], [], []
    index = np.empty(0, dtype=np.int64)
    for bsrid in range(len(basis)):
        bsl, sign = two_body_target(i, j, k, l, basis[bsrid], bit_max, norb)
        if sign == 0:
            continue
        id_bsl = lookup(bsl, basis, index)
        if id_bsl < 0:
            continue
        row_ind.append(id_bsl); col_ind.append(bsrid); data.append(sign)
    return row_ind, col_ind, data
