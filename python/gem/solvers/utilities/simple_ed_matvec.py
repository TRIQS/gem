#######################################################
# Matrix-free kernels for the SimpleED solver
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
#######################################################
'''
On-the-fly application of the embedding Hamiltonian, so that ARPACK never needs
the matrix.

``SimpleED`` normally stores, per symmetry sector, ``norb**2`` sparse ``c^dag_i
c_j`` operators plus ``Hone``, ``Htwo`` and ``Ham``. That is ``O(dim * norb**2)``
memory and it, not ARPACK, sets the largest reachable sector. The kernels here
recompute the Fock-space bit algebra on every application instead, so the only
per-sector storage left is the basis array (and the eigenvectors that are kept).

The price is speed: every applied term redoes the mask/sign arithmetic and a
lookup, so a matvec costs a few times what the stored CSC costs. This path is
opt-in through ``solver_params['matrix_free']``.

**The fermionic signs are not duplicated.** ``cid_cj_target``,
``two_body_target`` and ``docc_target`` hold the bit logic once, and both the
stored builders in ``simple_ed.py`` (``build_cid_cj_csc``,
``build_two_body_ijkl_csc_2``, ``_build_docc_op_sector``) and the kernels below
go through them, so the two paths cannot disagree.

Bit convention, inherited from ``simple_ed.py``: ``bit_max = 2**(norb-1)`` and
spin-orbital ``i`` is bit ``norb-1-i``, i.e. MSB-first.
'''

import numpy as np
from numba import jit
from scipy.sparse.linalg import LinearOperator


# -- state lookup --------------------------------------------------------

def build_index_table(basis, norb, max_norb=22):
    '''Direct Fock-state -> basis-position table, ``-1`` where absent.

    Removes the ``log(dim)`` binary search from every applied term, at the cost
    of an array of ``2**norb`` entries. Above ``max_norb`` that array is no
    longer worth it, and a zero-length array is returned instead; ``lookup``
    then falls back to ``np.searchsorted``.
    '''
    if norb > max_norb:
        return np.empty(0, dtype=np.int64)
    dtype = np.int32 if basis.size < 2**31 else np.int64
    index = np.full(1 << norb, -1, dtype=dtype)
    index[basis] = np.arange(basis.size, dtype=dtype)
    return index


@jit(nopython=True)
def lookup(bsl, basis, index):
    '''Position of the Fock state ``bsl`` in the sorted ``basis``, -1 if absent.'''
    if index.size > 0:
        return np.int64(index[bsl])
    # np.searchsorted returns len(basis) when bsl is past the end, so the bound
    # has to be tested *before* indexing (the stored builders get this wrong,
    # and under nopython there is no bounds check to catch it).
    pos = np.searchsorted(basis, bsl)
    if pos >= basis.size or basis[pos] != bsl:
        return np.int64(-1)
    return np.int64(pos)


# -- shared bit algebra --------------------------------------------------

@jit(nopython=True)
def find_count(i, j, bsr, norb, bsltmp):
    '''Fermionic sign count for c_i^dag c_j.'''
    bit_tmp = (((1 << j) - 1) & (bsr >> (norb - j)))
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
def cid_cj_target(i, j, bsr, bit_max, norb):
    '''Apply ``c^dag_i c_j`` to the Fock state ``bsr``.

    Returns ``(bsl, sign)``, with ``sign == 0`` when the state is annihilated.
    '''
    tmp_bit1 = bit_max >> j
    tmp_bit2 = bit_max >> i
    if (tmp_bit1 & bsr) != tmp_bit1 or ((tmp_bit2 & bsr) == tmp_bit2 and i != j):
        return np.int64(0), 0
    bsltmp = bsr ^ tmp_bit1
    bsl    = bsltmp | tmp_bit2
    count  = find_count(i, j, bsr, norb, bsltmp)
    if count & 1:
        return np.int64(bsl), -1
    return np.int64(bsl), 1


@jit(nopython=True)
def two_body_target(i, j, k, l, bsr, bit_max, norb):
    '''Apply ``c^dag_i c^dag_k c_l c_j`` to the Fock state ``bsr``.

    Returns ``(bsl, sign)``, with ``sign == 0`` when the state is annihilated.
    Callers must skip ``l == j`` and ``i == k`` (those terms vanish by Pauli and
    are not handled here), exactly as ``_build_two_body`` does.

    Known pre-existing limitation, kept verbatim from the stored builder so the
    two paths agree. The two creation operators use ``|`` without checking that
    the orbital is empty, so ``c^dag_k`` (or ``c^dag_i``) acting on an occupied
    orbital yields a state with the wrong particle number instead of zero. The
    term is handled correctly iff ``{i, k} == {j, l}``, i.e. iff both created
    orbitals are ones that ``c_l c_j`` has just emptied; density-density
    (``j == i``, ``l == k``) and exchange (``j == k``, ``l == i``) terms satisfy
    this, the Kanamori spin-flip and pair-hopping terms do not.

    It is harmless whenever the basis has a fixed particle number, which is
    every ``use_Ntot=True`` sector and every ``(N, Sz)`` sector: the bad target
    state has ``N-1`` or ``N-2`` particles, so it is not in the basis, the
    lookup discards it, and zero is the right answer. Verified against a
    brute-force fermionic reference: ``use_Ntot=True`` reproduces the exact
    ground state to 1e-15 with a Kanamori ``V2E``.

    It does bite in the ``use_Ntot=False`` full Fock space, where every state is
    present so the spurious element survives: there a Kanamori ``V2E`` gives an
    ``Htwo`` that is non-Hermitian by ``J`` and a ground-state energy wrong at
    the 1e-2 level. Use ``use_Ntot=True`` with such interactions.
    '''
    tmp_bit1 = bit_max >> j
    tmp_bit2 = bit_max >> l
    tmp_bit3 = bit_max >> k
    tmp_bit4 = bit_max >> i
    if (tmp_bit1 & bsr) != tmp_bit1 or (tmp_bit2 & bsr) != tmp_bit2:
        return np.int64(0), 0
    bsltmp1 = bsr     ^ tmp_bit1
    bsltmp2 = bsltmp1 ^ tmp_bit2
    bsltmp3 = bsltmp2 | tmp_bit3
    bsl     = bsltmp3 | tmp_bit4
    count = 0
    for (bits, shift) in [(bsr, j), (bsltmp1, l), (bsltmp2, k), (bsltmp3, i)]:
        bit_tmp = (((1 << shift) - 1) & (bits >> (norb - shift)))
        while bit_tmp:
            count  += bit_tmp & 1
            bit_tmp >>= 1
    if count & 1:
        return np.int64(bsl), -1
    return np.int64(bsl), 1


@jit(nopython=True)
def docc_target(i, bsr, bit_max, norb):
    '''Apply the double-occupancy operator on orbitals ``(i, i+1)`` to ``bsr``.'''
    tmp_bit1 = bit_max >> (i + 1)
    tmp_bit3 = bit_max >> i
    if (tmp_bit1 & bsr) != tmp_bit1 or (tmp_bit3 & bsr) != tmp_bit3:
        return np.int64(0), 0
    bsltmp1 = bsr     ^ tmp_bit1
    bsltmp2 = bsltmp1 | tmp_bit1
    bsltmp3 = bsltmp2 ^ tmp_bit3
    bsl     = bsltmp3 | tmp_bit3
    count = 0
    for (bits, shift) in [(bsr, i + 1), (bsltmp1, i + 1), (bsltmp2, i), (bsltmp3, i)]:
        bit_tmp = (((1 << shift) - 1) & (bits >> (norb - shift)))
        while bit_tmp:
            count  += bit_tmp & 1
            bit_tmp >>= 1
    if count & 1:
        return np.int64(bsl), -1
    return np.int64(bsl), 1


# -- term lists ----------------------------------------------------------
#
# The Hamiltonian coefficients are flattened once into contiguous arrays, with
# exactly the filters the stored builders apply, so that the kernels loop over
# the nonzero terms only.

def one_body_terms(h1e, tol=1e-8):
    '''Nonzero ``h1e[i, j]`` as ``(i, j, value)`` arrays. Mirrors _build_one_body.'''
    ii, jj = np.nonzero(np.abs(h1e) >= tol)
    return (np.ascontiguousarray(ii, dtype=np.int64),
            np.ascontiguousarray(jj, dtype=np.int64),
            np.ascontiguousarray(h1e[ii, jj], dtype=np.complex128))


def two_body_terms(V2E, tol=1e-8):
    '''Nonzero ``V2E[i, j, k, l]`` as ``(i, j, k, l, value)`` arrays.

    Mirrors ``_build_two_body``: the ``l == j`` / ``i == k`` Pauli-zero terms are
    dropped and the ``0.5`` prefactor is folded into the value.
    '''
    ii, jj, kk, ll, vv = [], [], [], [], []
    for i in range(V2E.shape[0]):
        for j in range(V2E.shape[1]):
            for k in range(V2E.shape[2]):
                for l in range(V2E.shape[3]):
                    if l == j or i == k or abs(V2E[i, j, k, l]) < tol:
                        continue
                    ii.append(i); jj.append(j); kk.append(k); ll.append(l)
                    vv.append(0.5 * V2E[i, j, k, l])
    return (np.array(ii, dtype=np.int64), np.array(jj, dtype=np.int64),
            np.array(kk, dtype=np.int64), np.array(ll, dtype=np.int64),
            np.array(vv, dtype=np.complex128))


def empty_one_body_terms():
    return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.complex128))


def empty_two_body_terms():
    return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.complex128))


# -- kernels -------------------------------------------------------------

@jit(nopython=True)
def apply_terms(ob_i, ob_j, ob_v, tb_i, tb_j, tb_k, tb_l, tb_v,
                basis, index, bit_max, norb, x, out):
    '''``out += H @ x`` for the one-body + two-body term lists.'''
    for c in range(basis.size):
        xc = x[c]
        if xc == 0.0:
            continue
        bsr = basis[c]
        for t in range(ob_i.size):
            bsl, sign = cid_cj_target(ob_i[t], ob_j[t], bsr, bit_max, norb)
            if sign == 0:
                continue
            r = lookup(bsl, basis, index)
            if r < 0:
                continue
            out[r] += ob_v[t] * sign * xc
        for t in range(tb_i.size):
            bsl, sign = two_body_target(tb_i[t], tb_j[t], tb_k[t], tb_l[t],
                                        bsr, bit_max, norb)
            if sign == 0:
                continue
            r = lookup(bsl, basis, index)
            if r < 0:
                continue
            out[r] += tb_v[t] * sign * xc


@jit(nopython=True)
def build_dense(ob_i, ob_j, ob_v, tb_i, tb_j, tb_k, tb_l, tb_v,
                basis, index, bit_max, norb, Hd):
    '''``Hd += H`` as a dense array, for the small-sector eigh branch.'''
    for c in range(basis.size):
        bsr = basis[c]
        for t in range(ob_i.size):
            bsl, sign = cid_cj_target(ob_i[t], ob_j[t], bsr, bit_max, norb)
            if sign == 0:
                continue
            r = lookup(bsl, basis, index)
            if r < 0:
                continue
            Hd[r, c] += ob_v[t] * sign
        for t in range(tb_i.size):
            bsl, sign = two_body_target(tb_i[t], tb_j[t], tb_k[t], tb_l[t],
                                        bsr, bit_max, norb)
            if sign == 0:
                continue
            r = lookup(bsl, basis, index)
            if r < 0:
                continue
            Hd[r, c] += tb_v[t] * sign


@jit(nopython=True)
def accum_denmat(basis, index, bit_max, norb, nmax, U, bw, dm):
    '''``dm[i,j] += sum_n bw[n] <n| c^dag_i c_j |n>`` for ``i, j < nmax``.

    Algebraically identical to ``_weighted_trace(denmat_op[(i,j)], U, bw)`` for
    every pair, and the same ``O(dim * nmax**2 * nstates)`` cost, but the
    operators are never stored.
    '''
    nst = bw.size
    for c in range(basis.size):
        bsr = basis[c]
        for i in range(nmax):
            for j in range(nmax):
                bsl, sign = cid_cj_target(i, j, bsr, bit_max, norb)
                if sign == 0:
                    continue
                r = lookup(bsl, basis, index)
                if r < 0:
                    continue
                acc = 0.0 + 0.0j
                for n in range(nst):
                    acc += bw[n] * np.conj(U[r, n]) * U[c, n]
                dm[i, j] += sign * acc


@jit(nopython=True)
def accum_docc(basis, index, bit_max, norb, i, U, bw):
    '''``sum_n bw[n] <n| n_i n_{i+1} |n>``.'''
    nst = bw.size
    total = 0.0 + 0.0j
    for c in range(basis.size):
        bsl, sign = docc_target(i, basis[c], bit_max, norb)
        if sign == 0:
            continue
        r = lookup(bsl, basis, index)
        if r < 0:
            continue
        acc = 0.0 + 0.0j
        for n in range(nst):
            acc += bw[n] * np.conj(U[r, n]) * U[c, n]
        total += sign * acc
    return total


# -- operator ------------------------------------------------------------

class EmbeddingHamiltonian(LinearOperator):
    '''Matrix-free ``H = H_one + H_two`` in one symmetry sector.

    Passed straight to ``scipy.sparse.linalg.eigsh``. ``dtype`` is always
    ``complex128``: ``build_h1e`` builds ``h1e`` complex, so the stored CSC path
    upcasts to complex too.
    '''

    def __init__(self, basis, index, norb, ob_terms, tb_terms):
        self.basis    = basis
        self.index    = index
        self.norb     = norb
        self.bit_max  = 2**(norb - 1)
        self.ob_terms = ob_terms
        self.tb_terms = tb_terms
        n = basis.size
        super().__init__(dtype=np.complex128, shape=(n, n))

    def _matvec(self, x):
        x   = np.ascontiguousarray(np.asarray(x).ravel(), dtype=np.complex128)
        out = np.zeros(self.shape[0], dtype=np.complex128)
        apply_terms(*self.ob_terms, *self.tb_terms,
                    self.basis, self.index, self.bit_max, self.norb, x, out)
        return out

    def apply_two_body(self, x):
        '''``H_two @ x``, needed by compute_E2loc.'''
        x   = np.ascontiguousarray(np.asarray(x).ravel(), dtype=np.complex128)
        out = np.zeros(self.shape[0], dtype=np.complex128)
        apply_terms(*empty_one_body_terms(), *self.tb_terms,
                    self.basis, self.index, self.bit_max, self.norb, x, out)
        return out

    def to_dense(self):
        '''Dense ``H``, for the sectors small enough to go through eigh.'''
        Hd = np.zeros(self.shape, dtype=np.complex128)
        build_dense(*self.ob_terms, *self.tb_terms,
                    self.basis, self.index, self.bit_max, self.norb, Hd)
        return Hd
