#######################################################
# GPU-only Exact Diagonalization solver (NVIDIA CUDA via JAX)
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
#
# Requires: jax[cuda12] (or equivalent)
# Raises RuntimeError at construction if no CUDA/ROCm backend is found.
#######################################################

import warnings
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _detect_gpu_backend():
    """Return True if JAX has a CUDA or ROCm backend, False otherwise."""
    if not _JAX_AVAILABLE:
        return False
    try:
        return jax.default_backend().lower() in ('cuda', 'rocm')
    except Exception:
        return False


_JAX_GPU = _detect_gpu_backend()


from gem.solvers.simple_ed import SimpleED


class SimpleEDGPU(SimpleED):
    """GPU-only (NVIDIA CUDA) drop-in replacement for SimpleED.

    Uses JAX with x64 enabled: full complex128/float64 precision.
    Raises RuntimeError at construction if no CUDA/ROCm backend is found.
    Hamiltonian construction stays on CPU; eigh and density-matrix run on GPU.

    Parameters
    ----------
    gpu_thresh : int
        Minimum sector Hilbert-space size to offload to GPU (default 0).
    All other parameters are identical to SimpleED.
    """

    def __init__(self, norb, use_Ntot=False, use_Sz=False,
                 dtype=np.complex128, N_sector=None, Sz_sector=None,
                 solver_params=None, gpu_thresh=0, **kwargs):
        if not _JAX_GPU:
            raise RuntimeError(
                "SimpleEDGPU requires a CUDA or ROCm GPU with JAX. "
                "Install with: pip install jax[cuda12]")
        self.gpu_thresh = gpu_thresh
        super().__init__(norb, use_Ntot=use_Ntot, use_Sz=use_Sz,
                         dtype=dtype, N_sector=N_sector, Sz_sector=Sz_sector,
                         solver_params=solver_params, **kwargs)

    # -- GPU kernels --

    def _eigh_gpu(self, mat):
        """Diagonalise a dense Hermitian matrix on CUDA via JAX (full float64)."""
        vals_j, vecs_j = jnp.linalg.eigh(jnp.array(mat))
        return np.array(vals_j).astype(np.float64), np.array(vecs_j).astype(mat.dtype)

    def _dm_sector_gpu(self, s):
        """Density-matrix contribution from sector s on GPU.

        Uses: tr(U† op U diag(w)) = (U.conj() * (op @ U) * w).sum()
        Operators are densified one at a time to bound GPU memory.
        """
        bw_s = np.asarray(self.bw_per_sector[s])
        U_g  = jnp.array(self.evecs_list[s])
        w_g  = jnp.array(bw_s)
        rows = []
        for i in range(self.norb):
            cols = []
            for j in range(self.norb):
                op_g = jnp.array(self.denmat_op_list[s][(i, j)].toarray())
                cols.append((U_g.conj() * (op_g @ U_g) * w_g).sum())
            rows.append(jnp.stack(cols))
        return np.array(jnp.stack(rows)).astype(self.data_type)

    # -- Overridden mandatory methods --

    def solve_Hemb(self, num_eig=1, which=None, tol=None, verbose=0, T=0.0):
        """Like SimpleED.solve_Hemb but uses GPU eigh for sectors >= gpu_thresh."""
        if T > 0.0 and ((self.use_Ntot and self.N_sector is not None) or
                         (self.use_Sz   and self.Sz_sector is not None)):
            warnings.warn(
                "A restricted symmetry sector is selected: the partition "
                "function may be incomplete at T>0.")

        which = self.solver_params.get('which', 'SA') if which is None else which
        tol   = self.solver_params.get('tol',   1e-8) if tol   is None else tol

        self.evals_list = []
        self.evecs_list = []

        for s, Ham_s in enumerate(self.Ham_list):
            hsize_s = self.hsize_list[s]
            if verbose > 0:
                print(f'Sector {s}: diagonalising (dim={hsize_s})')
            if hsize_s < 4000:
                mat = Ham_s.toarray()
                if hsize_s >= self.gpu_thresh:
                    vals, vecs = self._eigh_gpu(mat)
                else:
                    vals, vecs = eigh(mat)
            else:
                v0 = self.prev_gs_list[s] if T == 0.0 else None
                vals, vecs = eigsh(Ham_s, k=num_eig, which=which, tol=tol, v0=v0)
            so = vals.argsort()
            self.evals_list.append(vals[so])
            self.evecs_list.append(vecs[:, so])
            self.prev_gs_list[s] = vecs[:, so[0]].copy()

        self.gs_ene = min(evals[0] for evals in self.evals_list)
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
                        break
                self.bw_per_sector.append(bw_s)
        else:
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

        for s in range(len(self.sectors)):
            n_s = len(self.bw_per_sector[s])
            self.evals_list[s] = self.evals_list[s][:n_s]
            self.evecs_list[s] = self.evecs_list[s][:, :n_s]

        self.bw_list = [bw for bw_s in self.bw_per_sector for bw in bw_s]
        self.gs_sector = int(np.argmin(
            [evals[0] if len(evals) > 0 else np.inf for evals in self.evals_list]))
        self.gs_wf = self.evecs_list[self.gs_sector][:, 0]

        if len(self.sectors) == 1:
            self.evals = self.evals_list[0]
            self.evecs = self.evecs_list[0]

        return self.gs_wf, self.gs_ene

    def calc_density_matrix(self):
        """Like SimpleED.calc_density_matrix but uses GPU for sectors >= gpu_thresh."""
        dm = np.zeros((self.norb, self.norb), dtype=self.data_type)

        for s in range(len(self.sectors)):
            bw_s = np.asarray(self.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            hsize_s = self.hsize_list[s]
            if hsize_s >= self.gpu_thresh:
                dm += self._dm_sector_gpu(s)
            else:
                U_s = self.evecs_list[s]
                W_s = np.diag(bw_s)
                for i in range(self.norb):
                    for j in range(self.norb):
                        dm[i, j] += np.trace(
                            U_s.conj().T @ self.denmat_op_list[s][(i, j)] @ U_s @ W_s)

        dm /= self.Zpart
        self.dm = dm
        return dm
