.. _example_bethe_1orb:

Single-Orbital Hubbard Model (Bethe Lattice)
============================================

**Script:** ``examples/Bethe_1orb.py``

The canonical ghost-GA example: a sweep over the Hubbard :math:`U` for the
single-orbital model on the Bethe lattice (semicircular DOS, half-bandwidth
:math:`D = 1`).  The calculation enforces spin SU(2) symmetry and tracks the
quasiparticle weight :math:`Z` as a function of :math:`U`, revealing the
Mott metal-insulator transition.

Setup
-----

The semicircular DOS is sampled on a fine energy mesh and used to build the
dispersion array::

    import numpy as np
    from triqs_ghostGA.lattice import Lattice
    from triqs_ghostGA.fragment import Fragment
    from triqs_ghostGA.solvers.simple_ed import SimpleED

    # 1 orbital x 2 spins, B=3 bath orbitals per spin
    nimp, B = 2, 3
    nbath, ntot = nimp * B, nimp + nimp * B

    # semicircular DOS on a fine energy mesh
    e_list = np.linspace(-1, 1, 5001)
    wks = np.sqrt(1 - e_list**2); wks /= wks.sum()
    eks = np.array([np.kron([[e]], np.eye(2)) for e in e_list])

    lattice = Lattice(eks, wk_list=wks)

Self-consistency loop
---------------------

For each value of :math:`U` the embedding Hamiltonian is built and the
ghost-GA equations are iterated to convergence::

    U = 2.5
    eloc = np.diag([-U/2, -U/2])
    Utensor = np.zeros((nimp,)*4)
    Utensor[0, 0, 1, 1] = Utensor[1, 1, 0, 0] = U

    solver   = SimpleED(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
    fragment = Fragment(nimp, nbath, eloc, Utensor, solver)

    mu, T, mix, tol = 0.0, 0.0, 0.2, 1e-5
    for it in range(100):
        lattice.solve_qp([fragment], T=T)
        fragment.update_hybridization(T=T)
        fragment.impose_spin_SU2_symmetry()
        fragment.solve_impurity(mu, T=T, num_eig=10)

        Lambda_old, R_old = fragment.Lambda.copy(), fragment.R.copy()
        fragment.update_self_energy(T=T)
        fragment.impose_spin_SU2_symmetry()

        diff = max(np.abs(fragment.Lambda - Lambda_old).max(),
                   np.abs(fragment.R      - R_old     ).max())
        fragment.Lambda = (1 - mix)*fragment.Lambda + mix*Lambda_old
        fragment.R      = (1 - mix)*fragment.R      + mix*R_old
        fragment.impose_spin_SU2_symmetry()
        if diff < tol and it > 2:
            break

    Z = fragment.compute_Z()

The call to ``impose_spin_SU2_symmetry`` after each step enforces that the
:math:`\uparrow` and :math:`\downarrow` sectors remain identical, keeping the
solution in the paramagnetic state throughout the sweep.
