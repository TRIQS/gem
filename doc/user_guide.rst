.. _user_guide:

User Guide
**********

This section provides a brief introduction to the theory behind ghostGA and
worked examples showing how to set up and run calculations.

.. contents::
   :local:
   :depth: 2

Theoretical Background
======================

The Gutzwiller Approximation
-----------------------------

The Gutzwiller approximation (GA) is a variational method for lattice models of
strongly correlated electrons.  The trial state is obtained by applying a local
projector :math:`\hat{P} = \prod_i \hat{P}_i` to an uncorrelated Slater
determinant :math:`|\Psi_0\rangle`.  Each on-site projector :math:`\hat{P}_i`
re-weights the local many-body configurations, suppressing or enhancing
occupancies relative to the non-interacting reference.

In the limit of infinite lattice coordination (or equivalently within the
Gutzwiller mean-field decoupling), the expectation value of the Hamiltonian
reduces to an effective quasiparticle band problem with renormalised hopping
amplitudes controlled by orbital-dependent quasiparticle weights :math:`Z`.

The Ghost Extension
--------------------

The standard GA is limited because its embedding Hamiltonian is a *non-interacting*
single-impurity model with at most one bath orbital per correlated orbital.  This
restricts the representable spectral structures and prevents a systematic
improvement of the approximation.

Ghost-GA extends the embedding Hamiltonian by adding *auxiliary* (ghost) orbitals to
the bath.  With :math:`N_g` ghost orbitals per correlated orbital the embedding problem
becomes a :math:`(1 + N_g)`-orbital Anderson impurity model.  The key properties are:

* At :math:`N_g = 0` the method reduces to the standard GA.
* As :math:`N_g \to \infty` the method converges to the exact DMFT solution.
* For finite :math:`N_g` it provides a controlled, systematically improvable
  approximation whose cost grows polynomially with :math:`N_g`.

The variational parameters are the bath hybridisation matrix and the on-site
embedding energies; they are determined self-consistently by minimising the
ghost-GA energy functional.

Finite-Temperature Extension
-----------------------------

At finite temperature the variational principle is replaced by a free-energy
minimisation.  The ghost-DMFT (GDMFT) framework embeds the correlated site in
an effective Anderson impurity model at temperature :math:`T`, and the
self-consistency equations relate the impurity Green's function to the local
lattice Green's function via a Dyson equation.

The method interpolates smoothly between ghost-GA (:math:`T \to 0`) and
conventional DMFT (large :math:`N_g`), retaining the low computational cost of
ghost-GA for moderate :math:`N_g`.

Self-Consistency Loop
----------------------

A typical ghost-GA calculation proceeds as follows:

1. **Build the dispersion** — construct the array ``eks`` of shape
   ``(n_k, nimp, nimp)`` and the corresponding k-point weights ``wks``.
2. **Create the lattice** — ``Lattice(eks, wk_list=wks)``.
3. **Choose a solver and create a fragment** —
   ``Fragment(nimp, nbath, eloc, Utensor, solver)``.
4. **Iterate until convergence**:

   a. ``lattice.solve_qp([fragment], T=T)`` — update the quasiparticle
      Green's function on the lattice.
   b. ``fragment.update_hybridization(T=T)`` — project the lattice Green's
      function onto the embedding bath.
   c. ``fragment.solve_impurity(mu, T=T)`` — solve the embedding Hamiltonian.
   d. ``fragment.update_self_energy(T=T)`` — extract updated variational
      parameters :math:`R` and :math:`\Lambda` from the density matrix.
   e. Mix old and new parameters and check convergence.

5. **Extract observables** — ``fragment.compute_Z()`` for the quasiparticle
   weight, ``fragment.denMat`` for the density matrix.

Examples
========

Single-Orbital Hubbard Model (Bethe Lattice)
---------------------------------------------

The canonical example is a sweep over the Hubbard :math:`U` for the
single-orbital model on the Bethe lattice (semicircular DOS, half-bandwidth
:math:`D = 1`).  See ``examples/Bethe_1orb.py`` for the full script.
The key steps are::

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

Finite-Temperature Calculation
--------------------------------

For finite-temperature calculations pass ``T`` to the ``solve_qp``, ``solve_impurity``, ``update_self_energy``, and ``update_hybridization`` methods.
The mixing is then handled internally by ``update_hybridization`` and ``update_self_energy``.
See ``examples/Bethe_1orb_finiteT.py`` for a complete example::

    lattice = Lattice(eks, wk_list=wks)
    fragment = Fragment(nimp, nbath, eloc, Utensor, solver)

    for it in range(itmax):
        lattice.solve_qp([fragment])
        fragment.update_hybridization(T=T)
        fragment.solve_impurity(mu)
        Lambda_old, R_old = fragment.Lambda.copy(), fragment.R.copy()
        fragment.update_self_energy(T=T)
        diff = max(np.abs(fragment.Lambda - Lambda_old).max(),
                   np.abs(fragment.R      - R_old     ).max())
        if diff < tol and it > 1:
            break

    Z = fragment.compute_Z(mu=mu)
