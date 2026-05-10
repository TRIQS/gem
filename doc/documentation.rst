.. _documentation:

API Reference
*************

The package is organised into two core objects that represent the main components of the ghost-GA method, namely the :class:`~triqs_ghostGA.fragment.Fragment` and the :class:`~triqs_ghostGA.lattice.Lattice`.
The :class:`~triqs_ghostGA.fragment.Fragment` class represents the locally correlated space (site or cluster) and takes care of solvinf the correspondig embedding problem and updating the self-energy and hybridization parameters.
The :class:`~triqs_ghostGA.lattice.Lattice` class represents the lattice, takes care of the Brillouin-zone integration, and makes the fragments talk to each other.

The software provides a collection of impurity solvers, which are organised in the :mod:`triqs_ghostGA.solvers` subpackage.  Each solver implements the common interface defined in :mod:`triqs_ghostGA.solvers.solver_template`, which allows them to be used interchangeably within the self-consistency loop.

Say something about the gdmft object too?

Core Modules
============

.. autosummary::
   :toctree: _autosummary
   :template: autosummary_module_template.rst
   :recursive:

   triqs_ghostGA.solvers
   triqs_ghostGA.fragment
   triqs_ghostGA.lattice
   triqs_ghostGA.gdmft


``fragment`` 
----------------------------------

The :mod:`triqs_ghostGA.fragment` module provides the :class:`~triqs_ghostGA.fragment.Fragment`
class, which represents a single correlated site (or cluster) together with its ghost
orbital bath.  It exposes methods for solving the embedding problem, computing
quasiparticle weights, and enforcing symmetry constraints.

``lattice`` 
-----------------------------------------

The :mod:`triqs_ghostGA.lattice` module provides the :class:`~triqs_ghostGA.lattice.Lattice`
class, which wraps the Brillouin-zone integration.  It computes the local lattice
Green's function, the kinetic energy, adjusts the chemical potential to enforce
a target filling, and most importantly, it updates the relevant observables to update the hybridisation function of each fragment.

``gdmft`` — Simple Self-Consistency Driver
-------------------------------------

The :mod:`triqs_ghostGA.gdmft` module provides the :class:`~triqs_ghostGA.gdmft.Gdmft`
class, which drives the ghost-DMFT self-consistency loop for a simple single fragment case.
It manages the exchange of hybridisation functions between the lattice and the impurity fragments and evaluates
the total-energy functional.


Solvers
=======

.. autosummary::
   :toctree: _autosummary
   :template: autosummary_module_template.rst
   :recursive:

   triqs_ghostGA.solvers

The :mod:`triqs_ghostGA.solvers` subpackage collects all supported impurity solvers.
Each solver implements the common interface defined in
:mod:`triqs_ghostGA.solvers.solver_template`.

Available solvers:

* **SimpleED** (:mod:`~triqs_ghostGA.solvers.simple_ed`) — lightweight exact
  diagonalisation, no extra dependencies.
* **PySCF** (:mod:`~triqs_ghostGA.solvers.pyscf_solvers`) — quantum chemistry
  methods via `PySCF <https://pyscf.org>`_.
* **PyBlock2** (:mod:`~triqs_ghostGA.solvers.pyblock2`) — DMRG via the
  `block2 <https://block2.readthedocs.io>`_ library.
* **EDIpack** (:mod:`~triqs_ghostGA.solvers.edipack`) — parallel exact
  diagonalisation via EDIpack2.
* **MPS/ITensor** (:mod:`~triqs_ghostGA.solvers.mps`) — tensor-network solver
  using ITensors.jl through juliacall.
* **SVD Solver** (:mod:`~triqs_ghostGA.solvers.svd_solver2`) — low-rank SVD
  compression of the Green's function.

Utilities
=========

.. autosummary::
   :toctree: _autosummary
   :template: autosummary_module_template.rst
   :recursive:

   triqs_ghostGA.utility

Helper routines used throughout the code:

* :mod:`~triqs_ghostGA.utility.utilities` — general linear-algebra and Green's-function helpers.
* :mod:`~triqs_ghostGA.utility.delta_fit` — fitting of the hybridisation function
  to a discrete bath.
