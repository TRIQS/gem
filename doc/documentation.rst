.. _documentation:

API Reference
*************

The package is organised into two core objects that represent the main components of the ghost-GA method, namely the :class:`~gem.fragment.Fragment` and the :class:`~gem.lattice.Lattice` objects.
The :class:`~gem.fragment.Fragment` class represents the locally correlated space (site or cluster) and takes care of solving the correspondig embedding problem and updating the self-energy and hybridization parameters.
The :class:`~gem.lattice.Lattice` class represents the lattice, takes care of the Brillouin-zone integration, and makes the fragments talk to each other.

The software provides a collection of impurity solvers, which are organised in the :mod:`gem.solvers` subpackage.
Each solver implements the common interface defined in :mod:`gem.solvers.solver_template`, which allows them to be used interchangeably within the self-consistency loop.

Say something about the gdmft object too?

Core Modules
============

.. autosummary::
   :toctree: _autosummary
   :template: autosummary_module_template.rst
   :recursive:

   gem.solvers
   gem.fragment
   gem.lattice
   gem.gdmft


``fragment``
----------------------------------

The :mod:`gem.fragment` module provides the :class:`~gem.fragment.Fragment`
class, which represents a single correlated site (or cluster) together with its ghost
orbital bath.  It holds the embedding Hamiltonian, the quasiparticle weights, and the
self-energy parameters (:math:`R`, :math:`\Lambda`) and hybridisation parameters
(:math:`D`, :math:`\Lambda_c`) that connect the fragment to the lattice.
The key methods are:

- :meth:`~gem.fragment.Fragment.solve_impurity` — solves the quantum-impurity
  (embedding) problem at a given chemical potential and temperature, returning the
  ground-state (or thermal) density matrix.

- :meth:`~gem.fragment.Fragment.update_self_energy` — updates the ghost-GA
  self-energy parameters :math:`R` and :math:`\Lambda` by minimising the ghost-GA
  energy functional with respect to these parameters, given the current impurity solution.

- :meth:`~gem.fragment.Fragment.update_hybridization` — updates the
  hybridisation parameters :math:`D` and :math:`\Lambda_c` by minimising the ghost-GA
  energy functional with respect to these parameters, given the current impurity solution.

``lattice``
-----------------------------------------

The :mod:`gem.lattice` module provides the :class:`~gem.lattice.Lattice`
class, which wraps the Brillouin-zone integration over the non-interacting dispersion.
It assembles the quasiparticle Hamiltonian from the self-energy parameters of all
fragments, integrates it over the Brillouin zone to obtain the thermal expectation values
to update the hybridization function, and feeds these results back to the fragments.
The key methods are:

- :meth:`~gem.lattice.Lattice.solve_qp` — performs the quasiparticle
  Brillouin-zone integration given the current self-energy parameters of each fragment,
  computes the thermal expectation values to update the hybridization function,
  and feeds these results back to the fragments.

- :meth:`~gem.lattice.Lattice.fit_mu` — adjusts the chemical potential
  iteratively until the total lattice filling matches a prescribed target.

``gdmft`` — Simple Self-Consistency Driver
------------------------------------------

The :mod:`gem.gdmft` module provides the :class:`~gem.gdmft.Gdmft`
class, which drives the ghost-DMFT self-consistency loop for a simple single fragment case.
It manages the exchange of hybridisation functions between the lattice and the impurity fragments and evaluates
the total-energy functional.


Solvers
=======

.. autosummary::
   :toctree: _autosummary
   :template: autosummary_module_template.rst
   :recursive:

   gem.solvers

The :mod:`gem.solvers` subpackage collects all supported impurity solvers.
Each solver implements the common interface defined in
:mod:`gem.solvers.solver_template`.

Available solvers:

* **SimpleED** (:mod:`~gem.solvers.simple_ed`) — lightweight exact
  diagonalisation, no extra dependencies.
* **PySCF** (:mod:`~gem.solvers.pyscf_solvers`) — quantum chemistry
  methods via `PySCF <https://pyscf.org>`_.
* **PyBlock2** (:mod:`~gem.solvers.pyblock2`) — DMRG via the
  `block2 <https://block2.readthedocs.io>`_ library.
* **EDIpack** (:mod:`~gem.solvers.edipack`) — parallel exact
  diagonalisation via EDIpack2.
* **MPS/ITensor** (:mod:`~gem.solvers.mps`) — tensor-network solver
  using ITensors.jl through juliacall.
* **SVD Solver** (:mod:`~gem.solvers.svd_solver2`) — low-rank SVD
  compression of the hybridisation function.

Utilities
=========

.. autosummary::
   :toctree: _autosummary
   :template: autosummary_module_template.rst
   :recursive:

   gem.utility

Helper routines used throughout the code:

* :mod:`~gem.utility.utilities` — general linear-algebra and Green's-function helpers.
* :mod:`~gem.utility.delta_fit` — routines to perform thermal density matrix fitting of the hybridisation function and self-energy parameters.
  to a discrete bath.
