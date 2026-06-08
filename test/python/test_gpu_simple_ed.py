#!/usr/bin/env python
"""
GPU vs CPU correctness and timing comparison for SimpleED (no-symmetry mode).

Mirrors the fixture from test_symmetries_simple_ed.py and adds:
  - a correctness check that GPU and CPU give the same physical results, and
  - a wall-clock timing comparison printed as part of the test output.

The whole test class is skipped automatically when no GPU device is found
(jax not installed, or no GPU backend found).

Backend
-------
  JAX is the single GPU framework for both eigh and density-matrix ops.
  Install jax-metal for Apple Silicon or jax[cuda12] for NVIDIA.
  _JAX_BACKEND_NAME reports the active backend ('METAL', 'cuda', …).

Precision tolerances
--------------------
  Metal (Apple Silicon)  float32 internally  atol = 1e-4
  CUDA                   float64 via x64     atol = 1e-10
"""
import unittest
import numpy as np
import time

from gem.solvers.simple_ed import SimpleED
from gem.solvers.simple_ed_gpu import SimpleEDGPU, _JAX_GPU


_GPU_AVAILABLE = _JAX_GPU
_BACKEND       = "CUDA" if _JAX_GPU else "none"
_ATOL          = 1e-10


# ---------------------------------------------------------------------------
# Shared fixtures (identical to test_symmetries_simple_ed)
# ---------------------------------------------------------------------------

def _make_hemb_inputs(U=1.5, B=3, seed=42):
    """Return (ntot, nimp, nbath, eloc, D, Lambdac, V2E) for a single-orbital
    embedding Hamiltonian with B bath levels per spin."""
    nimp  = 2
    nbath = nimp * B
    ntot  = nimp + nbath

    np.random.seed(seed)

    eloc = np.array([[-U/2, 0], [0, -U/2]], dtype=np.complex128)

    V2E = np.zeros((nimp, nimp, nimp, nimp), dtype=np.float64)
    V2E[0, 0, 1, 1] = U
    V2E[1, 1, 0, 0] = U

    bath_energies = np.kron(np.linspace(-0.6, 0.6, B), np.ones(2))
    Lambdac = np.diag(bath_energies).astype(np.complex128)

    D_spin = np.abs(np.random.randn(B, 1)) * 0.3 + 0.1
    D = np.kron(D_spin, np.eye(2)).astype(np.complex128)

    return ntot, nimp, nbath, eloc, D, Lambdac, V2E


def _solve(solver, nimp, eloc, D, Lambdac, V2E, T, mu=0.0):
    """One full build + solve + observables cycle."""
    solver.build_Hemb(D, eloc, Lambdac, V2E, mu=mu)
    solver.solve_Hemb(T=T)
    dm   = solver.calc_density_matrix().copy()
    e1   = solver.compute_E1loc(nimp)
    e2   = solver.compute_E2loc()
    docc = solver.calc_double_occ(0)
    return dm, e1, e2, docc


def _solve_timed(solver, nimp, eloc, D, Lambdac, V2E, T, mu=0.0, n_repeats=5):
    """Run _solve n_repeats times after one warmup call (which triggers numba JIT
    and GPU lazy initialisation).  Returns (dm, e1, e2, docc, mean_s, std_s)."""
    _solve(solver, nimp, eloc, D, Lambdac, V2E, T, mu=mu)   # warmup
    times  = []
    result = None
    for _ in range(n_repeats):
        t0     = time.perf_counter()
        result = _solve(solver, nimp, eloc, D, Lambdac, V2E, T, mu=mu)
        times.append(time.perf_counter() - t0)
    dm, e1, e2, docc = result
    return dm, e1, e2, docc, float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@unittest.skipUnless(_GPU_AVAILABLE,
                     "No CUDA GPU available (install JAX with: pip install jax[cuda12])")
class TestGpuVsCpu(unittest.TestCase):
    """GPU (no-symmetry) vs CPU (no-symmetry) — correctness and timing."""

    def _assert_close(self, dm_cpu, dm_gpu, e1_cpu, e1_gpu,
                      e2_cpu, e2_gpu, docc_cpu, docc_gpu):
        np.testing.assert_allclose(
            dm_gpu.real, dm_cpu.real, atol=_ATOL,
            err_msg="Real part of density matrix: GPU vs CPU")
        np.testing.assert_allclose(
            dm_gpu.imag, dm_cpu.imag, atol=_ATOL,
            err_msg="Imag part of density matrix: GPU vs CPU")
        np.testing.assert_allclose(e1_gpu, e1_cpu, atol=_ATOL,
                                   err_msg="E1loc: GPU vs CPU")
        np.testing.assert_allclose(e2_gpu, e2_cpu, atol=_ATOL,
                                   err_msg="E2loc: GPU vs CPU")
        np.testing.assert_allclose(docc_gpu, docc_cpu, atol=_ATOL,
                                   err_msg="Double occupancy: GPU vs CPU")

    def test_correctness_small(self):
        """B=3 ntot=8 hsize=256 — gpu_thresh=0 sends every sector to GPU."""
        T = 0.25
        ntot, nimp, _, eloc, D, Lambdac, V2E = _make_hemb_inputs(U=1.5, B=3)

        cpu = SimpleED(ntot, use_Ntot=False, use_Sz=False, dtype=np.complex128)
        gpu = SimpleEDGPU(ntot, use_Ntot=False, use_Sz=False, dtype=np.complex128,
                          gpu_thresh=0)

        dm_cpu, e1_cpu, e2_cpu, docc_cpu = _solve(cpu, nimp, eloc, D, Lambdac, V2E, T)
        dm_gpu, e1_gpu, e2_gpu, docc_gpu = _solve(gpu, nimp, eloc, D, Lambdac, V2E, T)

        print(f"\n[{_BACKEND}] correctness — B=3 ntot={ntot} hsize={2**ntot} T={T}")
        print(f"  dm_cpu[:2,:2].real:\n{dm_cpu[:nimp,:nimp].real}")
        print(f"  dm_gpu[:2,:2].real:\n{dm_gpu[:nimp,:nimp].real}")
        print(f"  E1:   cpu={e1_cpu:.8f}  gpu={e1_gpu:.8f}  Δ={abs(e1_gpu-e1_cpu):.2e}")
        print(f"  E2:   cpu={e2_cpu:.8f}  gpu={e2_gpu:.8f}  Δ={abs(e2_gpu-e2_cpu):.2e}")
        print(f"  docc: cpu={docc_cpu:.8f}  gpu={docc_gpu:.8f}  Δ={abs(docc_gpu-docc_cpu):.2e}")
        print(f"  tolerance: atol={_ATOL:.0e}")

        self._assert_close(dm_cpu, dm_gpu, e1_cpu, e1_gpu,
                           e2_cpu, e2_gpu, docc_cpu, docc_gpu)

    def test_timing_medium(self):
        """B=4 ntot=10 hsize=1024 — correctness + timing, n_repeats=5."""
        T      = 0.25
        N_REPS = 5
        ntot, nimp, _, eloc, D, Lambdac, V2E = _make_hemb_inputs(U=1.5, B=4)

        cpu = SimpleED(ntot, use_Ntot=False, use_Sz=False, dtype=np.complex128)
        gpu = SimpleEDGPU(ntot, use_Ntot=False, use_Sz=False, dtype=np.complex128,
                          gpu_thresh=0)

        dm_cpu, e1_cpu, e2_cpu, docc_cpu, t_cpu, s_cpu = _solve_timed(
            cpu, nimp, eloc, D, Lambdac, V2E, T, n_repeats=N_REPS)
        dm_gpu, e1_gpu, e2_gpu, docc_gpu, t_gpu, s_gpu = _solve_timed(
            gpu, nimp, eloc, D, Lambdac, V2E, T, n_repeats=N_REPS)

        speedup = t_cpu / t_gpu if t_gpu > 0 else float('inf')
        winner  = "GPU faster" if speedup > 1 else "CPU faster"

        print(f"\n[{_BACKEND}] timing — B=4 ntot={ntot} hsize={2**ntot} "
              f"T={T}  n_repeats={N_REPS}")
        print(f"  CPU: {t_cpu*1e3:7.1f} ± {s_cpu*1e3:.1f} ms")
        print(f"  GPU: {t_gpu*1e3:7.1f} ± {s_gpu*1e3:.1f} ms")
        print(f"  Speedup: {speedup:.2f}×  ({winner})")

        self._assert_close(dm_cpu, dm_gpu, e1_cpu, e1_gpu,
                           e2_cpu, e2_gpu, docc_cpu, docc_gpu)


if __name__ == '__main__':
    unittest.main(verbosity=2)
