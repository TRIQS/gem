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

A proper presentation of the method has been given in many works for
zero :cite:`Lanata2017` and finite :cite:`Giuli2026` temperature.
Here, we give a brief recap.

The Gutzwiller Approximation
-----------------------------

The Gutzwiller approximation (GA) is a variational method for lattice models of
strongly correlated electrons with local interaction :cite:`Gutzwiller1965,Bunemann1998`.
The trial state is obtained by applying a linear map :math:`\hat{P} = \prod_i \hat{P}_i`
(often called projector despite not being one strictly speakign) to an uncorrelated Slater
determinant :math:`|\Psi_0\rangle`.
Each on-site operator :math:`\hat{P}_i` re-weights the local many-body configurations,
suppressing or enhancing occupancies relative to the non-interacting reference.
This method provided the first explanation for the Mott transition :cite:`Mott1968` via a renormalization of the
hopping parameters thanks to Brinkmann and Rice :cite:`BrinkmanRice1970`

Many slave-particle mean-field formulations, most notably Kotliar–Ruckenstein and
rotationally invariant slave-boson (RISB) approaches, are equivalent at the saddle-point level
to the multiorbital Gutzwiller approximation, differing mainly in the auxiliary-field representation
of the same variational energy functional :cite:`KotliarRuckenstein1986,Lechermann2007,BuenemannGebhard2007`.

The Ghost Extension
--------------------

The standard GA is limited because its embedding Hamiltonian is a *interacting*
single-impurity model with at most one bath orbital per correlated orbital.  This
restricts the representable spectral structures, missing high-energy features like Hubbard bands, and prevents a systematic
improvement of the approximation toward a dynamical description of strongly correalted systems as in Dynamical Mean-Field Theory (DMFT) :cite:`Georges1996`.

Ghost-GA :cite:`Lanata2017` extends the embedding Hamiltonian by adding *auxiliary* (ghost) orbitals to
the uncorrelated Slater determinant :math:`|\Psi_0\rangle` considering a number of electronic levels that is
a multiple :math:`B` of the physical ones.

* At :math:`B = 1` the method reduces to the standard GA.
* As :math:`B \to \infty` it has been proved that the method converges to DMFT :cite:`Giuli2026`
* For finite :math:`B > 1` it provides a controlled, systematically improvable
  approximation :cite:`Lee2023`.

The variational parameters are the bath hybridisation matrix and the on-site
embedding energies; One of the advantages of this method comes from a self-consistency based on static
expectation values instead of dynamical correlators (as it is for DMFT) that allow for much faster calculations of correlated matter.

Finite-Temperature Extension
-----------------------------

In Ref. :cite:`Giuli2026` it has been shown that Ghost-GA has a functional
formulation that connects it with DMFT and allows for a finite temperature extension
of the method. This framework embeds the correlated site in
an effective Anderson impurity model at temperature :math:`T`, and the
self-consistency equations are still rooted in static expectation values that are now thermal

The method interpolates smoothly between ghost-GA (:math:`T \to 0`) and
conventional DMFT ( :math:`B = \infty`), retaining the low computational cost of
ghost-GA for moderate :math:`B`.

HERE EXPLAIN 3 HAMILTONIANS AND SELF_CONSISTENCY EQUATIONS.

Self-Consistency Loop
----------------------

A typical ghost-GA calculation for a single correlated fragment proceeds as follows:

1. **Build the dispersion** — construct the array ``eks`` of shape
   ``(n_k, nimp, nimp)`` and the corresponding k-point weights ``wks``.
2. **Create the lattice** — ``Lattice(eks, wk_list=wks)``.
3. **Choose a solver and create a fragment** —
   ``Fragment(nimp, nbath, eloc, Utensor, solver)``.
4. **Iterate until convergence**:

   a. ``lattice.solve_qp([fragment], T=T)`` — solve the quasiparticle problem and update the quasiparticle target expectation values.
   b. ``fragment.update_hybridization(T=T)`` — Update the parameters :math:`D` and :math:`\Lambda_c` of the hybridization function.
   c. ``fragment.solve_impurity(mu, T=T)`` — solve the embedding Hamiltonian and update the impurity target expectation values.
   d. ``fragment.update_self_energy(T=T)`` — Update the parameters  :math:`R` and :math:`\Lambda` of the self-rnergy.
      parameters :math:`R` and :math:`\Lambda` from the density matrix.
   e. Mix old and new parameters and check convergence.

5. **Extract observables** — ``fragment.compute_Z()`` for the quasiparticle
   weight, ``fragment.denMat`` for the impurity density matrix.





Examples
========

.. toctree::
   :maxdepth: 1

   examples/bethe_1orb
   examples/square_afm
   examples/square_phase

