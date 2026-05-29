#######################################################
# Simple Exact Diagonalization solver
# Author: Tsung-Han Lee, Samuele Giuli
# Email:  henhans74716@gmail.com, samuele.giuli@gmail.com
#######################################################

import warnings
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.linalg import block_diag
import numpy as np
from numba import jit
import h5py

from math import factorial
from itertools import combinations

# List of what can be passed via solver_params:
# spin_pen : Coupling of (\hat{S})^2 to enforce spin singlet. Can be passed directly to solve_impurity()
# Sx_pen   : Coupling of (\hat{S}_x)^2 to unfavor magnetization in X direction
# Sy_pen   : Coupling of (\hat{S}_y)^2 to unfavor magnetization in Y direction
# Sz_pen   : Coupling of (\hat{S}_z)^2 to unfavor magnetization in Z direction
# which    : Parameter of scipy.sparse.linalg
# tol      : Parameter of scipy.sparse.linalg

class SimpleED(object):
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
    '''

    def __init__(self, norb, use_Ntot=False, use_Sz=False,
                 dtype=np.complex128, N_sector=None, Sz_sector=None,
                 solver_params=None, **kwargs):
        # backward compat: Nparticle into N_sector, will be removed after hearing from others
        if 'Nparticle' in kwargs:
            warnings.warn("Nparticle is deprecated, use N_sector instead.",
                          DeprecationWarning, stacklevel=2)
            N_sector = kwargs.pop('Nparticle')
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        self.type = "SimpleED"
        self.solver_params = solver_params if solver_params is not None else {}
        self.norb      = norb
        self.use_Ntot  = use_Ntot
        self.use_Sz    = use_Sz
        self.N_sector  = N_sector
        self.Sz_sector = Sz_sector

        # Not enforced hardly, be cautios to give float
        self.data_type = dtype

        # determine the list of (N, Sz) sector labels to solve
        self.sectors = self._get_sectors()

        # per-sector data structures
        self.basis_list     = []
        self.hsize_list     = []
        self.denmat_op_list = []
        self.S2_list        = []
        self.Sz_list        = []
        self.Sx_list        = []
        self.Sy_list        = []
        self.Hone_list      = [None] * len(self.sectors)
        self.Htwo_list      = [None] * len(self.sectors)
        self.V2E            = None  # cached interaction tensor for rebuild check
        self.prev_gs_list   = [None] * len(self.sectors)

        for s, (N, Sz) in enumerate(self.sectors):
            basis_s = self._build_basis(N, Sz)
            hsize_s = len(basis_s)
            self.basis_list.append(basis_s)
            self.hsize_list.append(hsize_s)
            print(f'Sector {s} (N={N}, Sz={Sz}): basis size = {hsize_s}')

            denmat_op_s = self._build_denmat_op(basis_s, hsize_s)
            self.denmat_op_list.append(denmat_op_s)

            S2_s, Sz_s, Sx_s, Sy_s = self._build_S2_op(basis_s, hsize_s, denmat_op_s)
            self.S2_list.append(S2_s)
            self.Sz_list.append(Sz_s)
            self.Sx_list.append(Sx_s)
            self.Sy_list.append(Sy_s)

        # single-sector aliases for backward compatibility
        if len(self.sectors) == 1:
            self._set_single_sector_aliases(0)

# -- Mandatory functions --

    def build_Hemb(self, D, eloc, Lambdac, V2E, mu=0.0, debug=False, verbose=0,
                   spin_pen=None, sz_pen=None, sx_pen=None, sy_pen=None):
        '''Build the embedding Hamiltonian in every active sector.'''
        #Try to get penalties from solver_params if not given explicitly
        spin_pen = self.solver_params.get('spin_pen', 0) if spin_pen is None else spin_pen
        sz_pen   = self.solver_params.get('sz_pen',   0) if sz_pen   is None else sz_pen
        sx_pen   = self.solver_params.get('sx_pen',   0) if sx_pen   is None else sx_pen
        sy_pen   = self.solver_params.get('sy_pen',   0) if sy_pen   is None else sy_pen

        print('build one-body')
        self.build_h1e(eloc, D, Lambdac, mu, verbose=verbose)

        # rebuild two-body only when V2E changes (shared across sectors)
        rebuild_two = (self.Htwo_list[0] is None) or np.any(V2E != self.V2E)
        if rebuild_two:
            print('build two-body')
            self.V2E = V2E.copy()

        print('one-body + two-body')
        self.Ham_list = []
        for s in range(len(self.sectors)):
            basis_s     = self.basis_list[s]
            hsize_s     = self.hsize_list[s]
            denmat_op_s = self.denmat_op_list[s]

            Hone_s = self._build_one_body(self.h1e, hsize_s, denmat_op_s)
            self.Hone_list[s] = Hone_s

            if rebuild_two:
                self.Htwo_list[s] = self._build_two_body(V2E, basis_s, hsize_s)

            Ham_s = (Hone_s + self.Htwo_list[s]
                     + spin_pen * self.S2_list[s]
                     + sz_pen   * self.Sz_list[s].dot(self.Sz_list[s])
                     + sx_pen   * self.Sx_list[s].dot(self.Sx_list[s])
                     + sy_pen   * self.Sy_list[s].dot(self.Sy_list[s]))
            self.Ham_list.append(Ham_s)


        # single-sector backward compat
        if len(self.sectors) == 1:
            self.Hone = self.Hone_list[0]
            self.Htwo = self.Htwo_list[0]
            self.Ham  = self.Ham_list[0]

        print('done')
        if debug:
            return self.Ham_list

    def solve_Hemb(self, num_eig=1, which=None, tol=None, verbose=0, T=0.0):
        '''Diagonalise each sector and build the global (thermal) partition function.'''
        if T > 0.0 and ((self.use_Ntot and self.N_sector is not None) or
                         (self.use_Sz   and self.Sz_sector is not None)):
            warnings.warn(
                "A restricted symmetry sector is selected: the partition "
                "function may be incomplete at T>0.")

        which = self.solver_params.get('which', 'SA') if which is None else which
        tol   = self.solver_params.get('tol',   1e-8) if tol   is None else tol

        # diagonalise every sector
        self.evals_list = []
        self.evecs_list = []

        for s, Ham_s in enumerate(self.Ham_list):
            hsize_s = self.hsize_list[s]
            print(f'Sector {s}: diagonalising (dim={hsize_s})')
            if hsize_s < 4000:
                vals, vecs = eigh(Ham_s.toarray())
            else:
                v0 = self.prev_gs_list[s] if T == 0.0 else None
                vals, vecs = eigsh(Ham_s, k=num_eig, which=which, tol=tol, v0=v0)
            so = vals.argsort()
            self.evals_list.append(vals[so])
            self.evecs_list.append(vecs[:, so])
            self.prev_gs_list[s] = vecs[:, so[0]].copy()

        # global ground-state energy
        self.gs_ene = min(evals[0] for evals in self.evals_list)

        # per-sector Boltzmann weights and global partition function
        self.bw_per_sector = []
        self.Zpart   = 0.0
        self.Tstates = 0
        self.deg     = 0

        if T > 0.0:
            beta = 1.0 / T
            for evals_s in self.evals_list:
                bw_s = []
                for eit in evals_s:
                    bw = float(np.exp(-beta * (eit - self.gs_ene)))
                    if bw > 1e-8:
                        bw_s.append(bw)
                        self.Zpart   += bw
                        self.Tstates += 1
                    else:
                        break   # eigenvalues are sorted; remaining are smaller
                self.bw_per_sector.append(bw_s)
        else:
            print('Building GS partition function across sectors')
            for evals_s in self.evals_list:
                bw_s = []
                for eit in evals_s:
                    if abs(eit - self.gs_ene) < 1e-4:
                        bw_s.append(1.0)
                        self.Zpart   += 1.0
                        self.Tstates += 1
                        self.deg     += 1
                    else:
                        break
                self.bw_per_sector.append(bw_s)

        # trim per-sector evecs/evals to thermal states only
        for s in range(len(self.sectors)):
            n_s = len(self.bw_per_sector[s])
            self.evals_list[s] = self.evals_list[s][:n_s]
            self.evecs_list[s] = self.evecs_list[s][:, :n_s]

        # flattening the bw_list, now not ordered
        self.bw_list = [bw for bw_s in self.bw_per_sector for bw in bw_s]

        # ground-state wavefunction (sector with lowest energy)
        self.gs_sector = int(np.argmin(
            [evals[0] if len(evals) > 0 else np.inf
             for evals in self.evals_list]))
        self.gs_wf  = self.evecs_list[self.gs_sector][:, 0]

        # verbose: print quantum numbers for every included state
        if verbose > 1:
            print('# s\tn\tEnergy\t\t\tS2\t\tSz\t\tBoltzmann')
            for s in range(len(self.sectors)):
                for n in range(len(self.bw_per_sector[s])):
                    vec = self.evecs_list[s][:, n]
                    S2v = vec.conj() @ self.S2_list[s].dot(vec)
                    Szv = vec.conj() @ self.Sz_list[s].dot(vec)
                    print(f"  {s}\t{n}\t{self.evals_list[s][n]:.10e}\t"
                          f"{S2v.real:.3f}+{S2v.imag:.1e}j\t"
                          f"{Szv.real:.3f}+{Szv.imag:.1e}j\t"
                          f"{self.bw_per_sector[s][n]:.6f}")
            print(f'deg={self.deg}  Zpart={self.Zpart}  Tstates={self.Tstates}')

        # single-sector backward compat
        if len(self.sectors) == 1:
            self.evals = self.evals_list[0]
            self.evecs = self.evecs_list[0]

        return self.gs_wf, self.gs_ene

    def calc_density_matrix(self):
        '''
        Compute the one-body density matrix as a thermal/degeneracy average
        across all active sectors.
        '''
        dm = np.zeros((self.norb, self.norb), dtype=self.data_type)

        for s in range(len(self.sectors)):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            U_s = self.evecs_list[s]   # (hsize_s, n_states_s)
            W_s = np.diag(bw_s)
            for i in range(self.norb):
                for j in range(self.norb):
                    dm[i, j] += np.trace(
                        U_s.conj().T @ self.denmat_op_list[s][(i, j)] @ U_s @ W_s)

        dm /= self.Zpart
        self.dm = dm
        return dm

    def compute_E1loc(self, nimp):
        '''One-body local energy (impurity block) averaged across sectors.'''
        result = 0.0
        for s in range(len(self.sectors)):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            U_s = self.evecs_list[s]
            W_s = np.diag(bw_s)
            dm_imp = np.zeros((nimp, nimp), dtype=self.data_type)
            for i in range(nimp):
                for j in range(nimp):
                    dm_imp[i, j] = np.trace(
                        U_s.conj().T @ self.denmat_op_list[s][(i, j)] @ U_s @ W_s)
            result += np.trace(self.h1e[:nimp, :nimp] @ dm_imp.T)

        return result / self.Zpart

    def compute_E2loc(self):
        '''Two-body local energy averaged across sectors.'''
        result = 0.0
        for s in range(len(self.sectors)):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            U_s = self.evecs_list[s]
            result += np.trace(
                U_s.conj().T @ self.Htwo_list[s] @ U_s @ np.diag(bw_s))

        return result / self.Zpart


# -- Utility functions for sectors --

    def _get_sectors(self):
        '''Return the list of (N, Sz) pairs to solve given the symmetry flags.'''
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
                return self._all_N_for_Sz(Sz, norb, n_half)
            # both None: all (N, Sz) sectors
            sectors = []
            for n in range(norb + 1):
                for nup in range(max(0, n - n_half), min(n, n_half) + 1):
                    sectors.append((n, (nup - (n - nup)) / 2.0))
            return sectors

        # use_Sz only (useful only for superconductivity)
        if not self.use_Ntot and self.use_Sz:
            if self.Sz_sector is not None:
                return self._all_N_for_Sz(self.Sz_sector, norb, n_half)
            return [(None, None)]

    @staticmethod
    def _all_N_for_Sz(Sz, norb, n_half):
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
        '''Build the Fock-state basis for sector (N, Sz).'''
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
        '''Expose per-sector attributes as flat attributes for single-sector use.'''
        self.basis     = self.basis_list[s]
        self.hsize     = self.hsize_list[s]
        self.denmat_op = self.denmat_op_list[s]
        self.S2        = self.S2_list[s]
        self.Sz        = self.Sz_list[s]
        self.Sx        = self.Sx_list[s]
        self.Sy        = self.Sy_list[s]


# -- operator builders (take basis/hsize explicitly) --

    def _build_denmat_op(self, basis, hsize):
        '''Build c†_i c_j operators in the given basis.'''
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
        '''Build S², Sz, Sx, Sy in the given basis.'''
        Sp = csc_matrix((hsize, hsize), dtype=self.data_type)
        Sm = csc_matrix((hsize, hsize), dtype=self.data_type)
        Sz = csc_matrix((hsize, hsize), dtype=self.data_type)
        for i in range(self.norb // 2):
            Sp += denmat_op[(2*i,   2*i+1)]
            Sm += denmat_op[(2*i+1, 2*i  )]
            Sz += 0.5*denmat_op[(2*i, 2*i)] - 0.5*denmat_op[(2*i+1, 2*i+1)]
        S2 = Sm.dot(Sp) + Sz.dot(Sz) + Sz
        Sx = 0.5*(Sp + Sm)
        Sy = 0.5*(Sp - Sm) / 1j
        return S2, Sz, Sx, Sy

    def _build_one_body(self, H1E, hsize, denmat_op):
        '''Build one-body Hamiltonian block in a given basis.'''
        Hone = csc_matrix((hsize, hsize), dtype=self.data_type)
        for i in range(self.norb):
            for j in range(self.norb):
                if np.abs(H1E[i, j]) < 1e-8:
                    continue
                Hone += H1E[i, j] * denmat_op[(i, j)]
        return Hone

    def _build_two_body(self, Umatrix, basis, hsize):
        '''Build two-body Hamiltonian block in a given basis.'''
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
        self.h1e = np.zeros((self.norb, self.norb), dtype=np.complex128)
        nimp = eloc.shape[0]
        self.h1e[:nimp, :nimp] = eloc - mu*np.eye(nimp)
        self.h1e[:nimp, nimp:] = D.T
        self.h1e[nimp:, nimp:] = -Lambdac
        self.h1e[nimp:, :nimp] = D.conj()
        if verbose > 3:
            print("h1e"); print(self.h1e)

    def build_docc_op(self, i, debug=False):
        '''
        Build double-occupancy operator for orbitals (i, i+1) in sector 0.
        For multi-sector use, call _build_docc_op_sector(i, s) directly.
        '''
        return self._build_docc_op_sector(i, s=0)

    def _build_docc_op_sector(self, i, s):
        basis   = self.basis_list[s]
        hsize   = self.hsize_list[s]
        bit_max = 2**(self.norb - 1)
        row_ind, col_ind, data = [], [], []
        for bsrid, bsr in enumerate(basis):
            tmp_bit1 = bit_max >> (i+1)
            tmp_bit3 = bit_max >>  i
            if (tmp_bit1 & bsr) != tmp_bit1 or (tmp_bit3 & bsr) != tmp_bit3:
                continue
            bsltmp1 = bsr    ^ tmp_bit1
            bsltmp2 = bsltmp1 | tmp_bit1
            bsltmp3 = bsltmp2 ^ tmp_bit3
            bsl     = bsltmp3 | tmp_bit3
            id_bsl  = np.searchsorted(basis, bsl)
            if bsl != basis[id_bsl]:
                continue
            count = 0
            for bsltmp, ii in [(bsr, i+1), (bsltmp1, i+1), (bsltmp2, i), (bsltmp3, i)]:
                bit_tmp = (((1 << ii) - 1) & (bsltmp >> (self.norb - ii)))
                while bit_tmp:
                    count += bit_tmp & 1
                    bit_tmp >>= 1
            sign = -1 if (count & 1) else 1
            row_ind.append(id_bsl); col_ind.append(bsrid); data.append(sign)
        return csc_matrix((data, (row_ind, col_ind)),
                          shape=(hsize, hsize), dtype=self.data_type)

    def calc_double_occ(self, i):
        '''Compute thermal double occupancy on orbital (i, i+1) averaged over all sectors.'''
        result = 0.0
        for s in range(len(self.sectors)):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            docc_op = self._build_docc_op_sector(i, s)
            U_s = self.evecs_list[s]
            result += np.trace(U_s.conj().T @ docc_op @ U_s @ np.diag(bw_s)).real
        return result / self.Zpart

    # kept for backward compat (single-sector callers)
    def build_denmat_op(self):
        self.denmat_op = self._build_denmat_op(self.basis_list[0], self.hsize_list[0])
        self.denmat_op_list[0] = self.denmat_op

    def build_S2_op(self):
        S2, Sz, Sx, Sy = self._build_S2_op(
            self.basis_list[0], self.hsize_list[0], self.denmat_op_list[0])
        self.S2_list[0], self.Sz_list[0] = S2, Sz
        self.Sx_list[0], self.Sy_list[0] = Sx, Sy
        self.S2, self.Sz, self.Sx, self.Sy = S2, Sz, Sx, Sy

    def build_one_body(self, H1E):
        self.Hone = self._build_one_body(H1E, self.hsize_list[0], self.denmat_op_list[0])
        self.Hone_list[0] = self.Hone

    def build_two_body(self, Umatrix, debug=False):
        self.Htwo = self._build_two_body(Umatrix, self.basis_list[0], self.hsize_list[0])
        self.Htwo_list[0] = self.Htwo

    def h5write_gs(self, filename, group_path, name):
        with h5py.File(filename, "a") as f:
            if group_path not in f:
                f.create_group(group_path)
            f[group_path][name] = self.gs_wf

    def h5read_state(self, filename, group_path, name):
        with h5py.File(filename, "r") as f:
            return f[group_path][name][:]

    def inner(self, bra, ket, operator=None):
        if operator is not None:
            return np.vdot(bra, operator.dot(ket))
        return np.vdot(bra, ket)


# -- binary basis utilities --


def table_ep(nstate, nparticle, dtype=np.int64):
    '''Binary basis for fixed particle number (no Sz constraint).'''
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
    '''Binary basis for fixed particle number and Sz.'''
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


@jit(nopython=True)
def residues(norb, determinant):
    residue_list = []
    nonzero = determinant.bit_count()
    for i in range(norb):
        mask1 = (1 << i)
        for j in range(i):
            mask2 = (1 << j)
            mask = mask1 ^ mask2
            if (determinant & ~mask).bit_count() == (nonzero - 2):
                residue_list.append(determinant & ~mask)
    return residue_list

@jit(nopython=True)
def add_particles(norb, residue_list):
    determinants = []
    for residue in residue_list:
        determinant = residue
        for i in range(norb):
            mask1 = (1 << i)
            if not bool(determinant & mask1):
                one_particle = determinant | mask1
                for j in range(i):
                    mask2 = (1 << j)
                    if not bool(one_particle & mask2):
                        two_particle = one_particle | mask2
                        determinants.append(two_particle)
    return list(set(determinants))


@jit(nopython=True)
def find_count(i, j, bsr, norb, bsltmp):
    '''Fermionic sign count for c_i^† c_j.'''
    bit_tmp = (((1 << j) - 1) & (bsr    >> (norb - j)))
    count = 0
    while bit_tmp:
        count  += bit_tmp & 1
        bit_tmp >>= 1
    bit_tmp = (((1 << i) - 1) & (bsltmp >> (norb - i)))
    while bit_tmp:
        count  += bit_tmp & 1
        bit_tmp >>= 1
    return count

@jit(nopython=True)
def build_cid_cj_csc(i, j, basis, bit_max, norb, debug=False):
    row_ind, col_ind, data = [], [], []
    for bsrid, bsr in enumerate(basis):
        tmp_bit1 = bit_max >> j
        tmp_bit2 = bit_max >> i
        if (tmp_bit1 & bsr) != tmp_bit1 or ((tmp_bit2 & bsr) == tmp_bit2 and i != j):
            continue
        bsltmp = bsr ^ tmp_bit1
        bsl    = bsltmp | tmp_bit2
        id_bsl = np.searchsorted(basis, bsl)
        if bsl != basis[id_bsl] or id_bsl >= len(basis):
            continue
        count = find_count(i, j, bsr, norb, bsltmp)
        sign  = -1 if (count & 1) else 1
        row_ind.append(id_bsl); col_ind.append(bsrid); data.append(sign)
    return row_ind, col_ind, data

@jit(nopython=True)
def build_two_body_ijkl_csc_2(i, j, k, l, basis, bit_max, norb, debug=False):
    row_ind, col_ind, data = [], [], []
    for bsrid, bsr in enumerate(basis):
        tmp_bit1 = bit_max >> j
        tmp_bit2 = bit_max >> l
        tmp_bit3 = bit_max >> k
        tmp_bit4 = bit_max >> i
        if (tmp_bit1 & bsr) != tmp_bit1 or (tmp_bit2 & bsr) != tmp_bit2:
            continue
        bsltmp1 = bsr    ^ tmp_bit1
        bsltmp2 = bsltmp1 ^ tmp_bit2
        bsltmp3 = bsltmp2 | tmp_bit3
        bsl     = bsltmp3 | tmp_bit4
        id_bsl  = np.searchsorted(basis, bsl)
        if bsl != basis[id_bsl] or id_bsl >= len(basis):
            continue
        count = 0
        for (bits, shift) in [(bsr,    j), (bsltmp1, l), (bsltmp2, k), (bsltmp3, i)]:
            bit_tmp = (((1 << shift) - 1) & (bits >> (norb - shift)))
            while bit_tmp:
                count  += bit_tmp & 1
                bit_tmp >>= 1
        sign = -1 if (count & 1) else 1
        row_ind.append(id_bsl); col_ind.append(bsrid); data.append(sign)
    return row_ind, col_ind, data

@jit(nopython=True)
def build_rholoc_onfly(basis, gs_wf, rholoc, bipart_smap):
    for i in range(len(basis)):
        for j in range(len(basis)):
            if bipart_smap[i,2] == bipart_smap[j,2] and abs(gs_wf[i]*gs_wf[j]) > 1e-12:
                rholoc[bipart_smap[i,1], bipart_smap[j,1]] += gs_wf[i]*gs_wf[j]
    return rholoc
