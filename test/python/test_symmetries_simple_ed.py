#!/usr/bin/env python
"""
Compare SimpleED with no-symmetry (full Hilbert space) versus all (N,Sz) sectors
for a given embedding Hamiltonian at finite temperature T=0.1, U=1.5.

Both solvers span the same Hilbert space and must give identical density
matrices, double occupancy, and energies.
"""
import unittest
import numpy as np
from gem.solvers.simple_ed import SimpleED


def _make_hemb_inputs(U=1.5, B=3, seed=42):
    """
    Return (nimp, ntot, eloc, D, Lambdac, V2E) for a single-orbital
    embedding Hamiltonian with B bath levels.
    Values are representative of a converged ghost-GA at moderate U.
    """
    nimp  = 2
    nbath = nimp * B
    ntot  = nimp + nbath

    np.random.seed(seed)

    eloc = np.array([[-U/2, 0], [0, -U/2]], dtype=np.complex128)

    V2E = np.zeros((nimp, nimp, nimp, nimp), dtype=np.float64)
    V2E[0, 0, 1, 1] = U
    V2E[1, 1, 0, 0] = U

    # Bath: B=3 levels per spin, Lambdac diagonal, D hybridisation
    bath_energies = np.kron(np.linspace(-0.6, 0.6, B), np.ones(2))
    Lambdac = np.diag(bath_energies).astype(np.complex128)

    D_spin = np.abs(np.random.randn(B, 1)) * 0.3 + 0.1
    D = np.kron(D_spin, np.eye(2)).astype(np.complex128)

    return ntot, nimp, nbath, eloc, D, Lambdac, V2E


def _solve(solver, nimp, eloc, D, Lambdac, V2E, T, mu=0.0):
    solver.build_Hemb(D, eloc, Lambdac, V2E, mu=mu)
    solver.solve_Hemb(T=T)
    dm   = solver.calc_density_matrix().copy()
    e1   = solver.compute_E1loc(nimp)
    e2   = solver.compute_E2loc()
    docc = solver.calc_double_occ(0)
    return dm, e1, e2, docc


class TestSymVsNoSym(unittest.TestCase):

    def test_nosym_vs_allsectors(self):
        T = 0.25
        U = 1.5

        ntot, nimp, _, eloc, D, Lambdac, V2E = _make_hemb_inputs(U)

        # Solver 1: no symmetry — full 2^ntot Hilbert space as one block
        solver_nosym = SimpleED(ntot, use_Ntot=False, use_Sz=False,
                                dtype=np.complex128)

        # Solver 2: all (N, Sz) sectors — block-diagonalised, same Hilbert space
        solver_sym = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                              N_sector=None, Sz_sector=None,
                              dtype=np.complex128)

        dm_n, e1_n, e2_n, docc_n = _solve(solver_nosym, nimp, eloc, D, Lambdac, V2E, T)
        dm_s, e1_s, e2_s, docc_s = _solve(solver_sym,   nimp, eloc, D, Lambdac, V2E, T)

        print(f"\nT={T}, U={U}")
        print(f"  dm_nosym[:2,:2]:\n{dm_n[:nimp,:nimp].real}")
        print(f"  dm_sym  [:2,:2]:\n{dm_s[:nimp,:nimp].real}")
        print(f"  E1loc: nosym={e1_n:.8f}  sym={e1_s:.8f}")
        print(f"  E2loc: nosym={e2_n:.8f}  sym={e2_s:.8f}")
        print(f"  docc:  nosym={docc_n:.8f}  sym={docc_s:.8f}")

        atol = 1e-6
        np.testing.assert_allclose(dm_n.real, dm_s.real, atol=atol,
                                   err_msg="Real part of density matrix disagrees")
        np.testing.assert_allclose(dm_n.imag, dm_s.imag, atol=atol,
                                   err_msg="Imag part of density matrix disagrees")
        np.testing.assert_allclose(e1_n, e1_s, atol=atol,
                                   err_msg="E1loc disagrees")
        np.testing.assert_allclose(e2_n, e2_s, atol=atol,
                                   err_msg="E2loc disagrees")
        np.testing.assert_allclose(docc_n, docc_s, atol=atol,
                                   err_msg="Double occupancy disagrees")


if __name__ == '__main__':
    unittest.main()
