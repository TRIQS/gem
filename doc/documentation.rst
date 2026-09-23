.. _documentation:

API Reference
*************

The package is organised into two core objects that represent the main components of the ghost-GA method, namely the :class:`~gem.fragment.Fragment` and the :class:`~gem.lattice.Lattice` objects.
The :class:`~gem.fragment.Fragment` class represents the locally correlated space (site or cluster) and takes care of solving the correspondig embedding problem and updating the self-energy and hybridization parameters.
The :class:`~gem.lattice.Lattice` class represents the lattice, takes care of the Brillouin-zone integration, and makes the fragments talk to each other.

The software provides a collection of impurity solvers, which are organised in the :mod:`gem.solvers` subpackage.
Every solver inherits from :class:`~gem.solvers.gem_solver.gemSolver` and implements the common interface documented in :mod:`gem.solvers.solver_template`, which allows them to be used interchangeably within the self-consistency loop.
:class:`~gem.fragment.Fragment` checks that the solver it is given is a :class:`~gem.solvers.gem_solver.gemSolver`.

Complete calculations, from the construction of these objects to the
self-consistency loop, are given in the :ref:`user guide <user_guide>`.

Core Modules
============

.. autosummary::
   :toctree: _autosummary
   :template: autosummary_module_template.rst
   :recursive:

   gem.fragment
   gem.lattice
   gem.solvers

Everything is expressed in **spin-orbital** indices, with spin as the *fastest*
index: level ``2*a`` is the down spin of orbital ``a`` and level ``2*a+1`` is its
up spin. An orbital-space matrix is lifted to spin-orbital space with
``np.kron(m, np.eye(2))``. The same ordering is assumed by the :math:`S_z`
sectors of :class:`~gem.solvers.simple_ed.SimpleED` and by the spin operators it
builds.

``fragment``
----------------------------------

The :mod:`gem.fragment` module provides the :class:`~gem.fragment.Fragment`
class, which represents a single correlated site (or cluster) together with its ghost
orbital bath.  It holds the embedding Hamiltonian, the quasiparticle weights, and the
self-energy parameters (:math:`R`, :math:`\Lambda`) and hybridisation parameters
(:math:`D`, :math:`\Lambda_c`) that connect the fragment to the lattice.

It is constructed as::

    Fragment(nimp, nbath, eloc, Utensor, solver,
             Lambda=None, R=None, Lambda_c=None, D=None, verbose=0)

.. list-table::
   :header-rows: 1
   :widths: 14 24 62

   * - Argument
     - Type / shape
     - Meaning
   * - ``nimp``
     - int
     - Number of impurity spin-orbital levels (orbitals × spins).
   * - ``nbath``
     - int
     - Number of ghost bath spin-orbital levels. It **must be a multiple of**
       ``nimp``.
   * - ``eloc``
     - ndarray ``(nimp, nimp)``
     - Local one-body Hamiltonian of the correlated space **without** the chemical
       potential.
   * - ``Utensor``
     - ndarray ``(nimp,)*4``
     - Local interaction tensor in the convention
       :math:`\frac{1}{2}\sum U_{ijkl} c^\dagger_i c_j c^\dagger_k c_l`, i.e. a
       Hubbard :math:`U n_\uparrow n_\downarrow` is entered as
       ``Utensor[0,0,1,1] = Utensor[1,1,0,0] = U``.
   * - ``solver``
     - :class:`~gem.solvers.gem_solver.gemSolver`
     - The impurity solver, already constructed for ``ntot`` levels.
   * - ``Lambda``
     - ndarray ``(nbath, nbath)``, optional
     - Starting self-energy parameter :math:`\Lambda`. Default: a
       ``tanh``-staggered diagonal.
   * - ``R``
     - ndarray ``(nbath, nimp)``, optional
     - Starting self-energy parameter :math:`R`. Default: uniform.
   * - ``Lambda_c``
     - ndarray ``(nbath, nbath)``, optional
     - Starting hybridisation parameter :math:`\Lambda_c`. Default: a
       ``tanh``-staggered diagonal.
   * - ``D``
     - ndarray ``(nbath, nimp)``, optional
     - Starting hybridisation parameter :math:`D`. Default: uniform.
   * - ``verbose``
     - int, default ``0``
     - Verbosity of the printout; maximum for ``verbose >= 3``.

All four parameter matrices are shape-validated, so a wrongly sized array raises
immediately. Passing the converged matrices of a previous run is the standard way
to continue a sweep in :math:`U`.

The key methods are:

- :meth:`~gem.fragment.Fragment.solve_impurity` — solves the quantum-impurity
  (embedding) problem at a given chemical potential and temperature, returning the
  ground-state (or thermal) density matrix.

- :meth:`~gem.fragment.Fragment.update_self_energy` — updates the ghost-GA
  self-energy parameters :math:`R` and :math:`\Lambda` by imposing the stationarity condition of the ghost-GA
  energy functional with respect to these parameters.

- :meth:`~gem.fragment.Fragment.update_hybridization` — updates the
  hybridisation parameters :math:`D` and :math:`\Lambda_c` by imposing the stationarity condition of the ghost-GA
  energy functional with respect to these parameters.

``lattice``
-----------------------------------------

The :mod:`gem.lattice` module provides the :class:`~gem.lattice.Lattice`
class, which wraps the Brillouin-zone integration over the non-interacting dispersion.
It assembles the quasiparticle Hamiltonian from the self-energy parameters of all
fragments, integrates it over the Brillouin zone to obtain the thermal expectation values
to update the hybridization function, and feeds these results back to the fragments.

It is constructed as::

    Lattice(ek_list, wk_list=None, verbose=0, use_mpi=True, comm=None)

.. list-table::
   :header-rows: 1
   :widths: 14 24 62

   * - Argument
     - Type / shape
     - Meaning
   * - ``ek_list``
     - ndarray ``(nk, nimp_tot, nimp_tot)``
     - Non-interacting inter-fragment dispersion, one matrix per k-point (or per energy,
       for a density-of-states calculation). ``nimp_tot`` is the sum of ``nimp``
       over all fragments. It
       must hold the **non-local hopping only** — the local part belongs in each
       fragment's ``eloc``.
   * - ``wk_list``
     - ndarray ``(nk,)``, optional
     - Integration weights, which **must sum to 1** . Default: uniform weights ``1/nk``.
   * - ``verbose``
     - int, default ``0``
     - Verbosity of the printout.
   * - ``use_mpi``
     - bool, default ``True``
     - Distribute the k-point sums over the ranks of ``comm``. Set to ``False``
       to force the serial path; ignored when mpi4py is absent.
   * - ``comm``
     - mpi4py communicator, optional
     - Defaults to ``MPI.COMM_WORLD``. Pass a sub-communicator to nest the
       k-point splitting inside another level of parallelism.

The k-points are split into contiguous chunks once, at construction, and every k
sum is closed by an *allreduce*, so all ranks leave with identical results — which
matters for :meth:`~gem.lattice.Lattice.fit_mu_quasiparticle`, whose bisection
must take the same branch everywhere. A warning is issued when there are more
ranks than k-points.

The key methods are:

- :meth:`~gem.lattice.Lattice.solve_qp` — performs the quasiparticle
  Brillouin-zone integration given the current self-energy parameters of each fragment,
  computes the quasiparticle thermal expectation values and feeds these results back to the fragments.

- :meth:`~gem.lattice.Lattice.fit_mu` — adjusts the chemical potential
  iteratively until the total lattice filling matches a prescribed target.

``solvers``
-----------

The :mod:`gem.solvers` subpackage collects all supported impurity solvers.
Each solver inherits from :class:`~gem.solvers.gem_solver.gemSolver` and
implements the common interface documented in
:mod:`gem.solvers.solver_template`.
Solver-specific parameters are never passed by the :class:`~gem.fragment.Fragment`:
they are given once, as the ``solver_params`` dictionary, when the solver is built
(it is deep-copied, and a non-dict raises ``TypeError``). They can still be changed
afterwards through ``fragment.solver.solver_params``, and unknown keys are ignored.

Available solvers:

* **SimpleED** (:mod:`~gem.solvers.simple_ed`) — lightweight exact
  diagonalisation, no extra dependencies.

.. _simple_ed:

``simple_ed``
^^^^^^^^^^^^^

The :mod:`gem.solvers.simple_ed` module provides
:class:`~gem.solvers.simple_ed.SimpleED`, which diagonalises the embedding
Hamiltonian exactly, sector by sector, and obtains the expectation values as
thermal/degeneracy-weighted averages over the included sectors.

It is constructed as::

    SimpleED(norb, use_Ntot=False, use_Sz=False, dtype=np.complex128,
             N_sector=None, Sz_sector=None, solver_params=None, comm=None)

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Argument
     - Type / default
     - Meaning
   * - ``norb``
     - int
     - Number of spin-orbital levels of the **embedding** problem, i.e. the
       fragment's ``ntot = nimp + nbath``, not ``nimp``.
   * - ``use_Ntot``
     - bool, default ``False``
     - Exploit particle-number conservation.
   * - ``use_Sz``
     - bool, default ``False``
     - Exploit :math:`S_z` conservation.
   * - ``N_sector``
     - int or ``None``
     - Particle-number sector to solve. ``None`` (the default) includes *all* of
       them.
   * - ``Sz_sector``
     - int/half-integer or ``None``
     - :math:`S_z` sector to solve. ``None`` (the default) includes *all* of
       them.
   * - ``dtype``
     - default ``np.complex128``
     - Working precision. ``np.float64`` keeps the whole solve real — operators,
       diagonalisation, eigenvectors and density matrix — which costs roughly
       half the time of the complex path in the stored branch. See
       :ref:`precision <simple_ed_precision>` for when the request is honoured.
   * - ``solver_params``
     - dict, optional
     - Solver-specific options; see the table below.
   * - ``comm``
     - mpi4py communicator, optional
     - Communicator over whose ranks the **sectors** are distributed (default
       ``MPI.COMM_WORLD``).

The symmetry flags only decide whether the Hamiltonian is block-diagonalised;
*which* blocks are kept is decided by ``N_sector`` and ``Sz_sector``. At half
filling with no symmetry breaking the usual — and by far the cheapest — choice is
``use_Ntot=True, use_Sz=True, N_sector=ntot//2, Sz_sector=0``; drop
``N_sector``/``Sz_sector`` when the ground state may leave that sector, and drop
``use_Sz`` when the calculation is magnetic or has off-diagonal spin terms.

Under MPI each rank builds and diagonalises only the sectors it owns, while the
reduced quantities are identical on every rank. ``build_Hemb``, ``solve_Hemb``,
``calc_density_matrix``, ``compute_E1loc``, ``compute_E2loc`` and
``calc_double_occ`` are therefore **collective** — calling one of them inside an
``if rank == 0:`` block deadlocks.

.. _simple_ed_precision:

Real and complex precision
""""""""""""""""""""""""""

``dtype`` is a *request*, not a guarantee. A real type is honoured only when
nothing in the embedding problem carries an imaginary part, which is checked at
every ``build_Hemb``:

* the one-body Hamiltonian, assembled from ``eloc``, ``D`` and ``Lambda_c``,
* the interaction tensor ``Utensor``,
* the ``sy_pen`` penalty, whose :math:`\hat{S}_y` operator is complex by
  construction.

If one of these is not real, the fallback is using ``complex128`` and
a warning is issued once — the results stay correct, only the speedup is lost.
``solver.data_type`` always holds the precision actually in use.

The keys :class:`~gem.solvers.simple_ed.SimpleED` reads from ``solver_params``
are:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Key
     - Default
     - Meaning
   * - ``num_eig``
     - ``None``
     - Number of eigenvalues to compute per sector. ``None`` means the ground
       state only at :math:`T = 0` and the *full* spectrum at :math:`T > 0`.
       Required at :math:`T > 0` in matrix-free mode for any sector reaching
       ``dense_cutoff``.
   * - ``dense_cutoff``
     - ``4000``
     - Sectors with a dimension below this are diagonalised fully with
       ``scipy.linalg.eigh``; larger ones go to ``scipy.sparse.linalg.eigsh``.
   * - ``which``
     - ``'SA'``
     - ``scipy.sparse.linalg.eigsh`` spectrum selector.
   * - ``tol``
     - ``1e-8``
     - ``scipy.sparse.linalg.eigsh`` convergence tolerance.
   * - ``bw_cutoff``
     - ``1e-8``
     - Smallest Boltzmann weight kept in the partition function at
       :math:`T > 0`.
   * - ``spin_pen``
     - ``0``
     - Coefficient of a :math:`\hat{S}^2` penalty added to the embedding
       Hamiltonian, to enforce a spin singlet.
   * - ``sx_pen``, ``sy_pen``, ``sz_pen``
     - ``0``
     - Coefficients of :math:`\hat{S}_x^2`, :math:`\hat{S}_y^2`,
       :math:`\hat{S}_z^2` penalties, to disfavour magnetisation along that
       axis.
   * - ``matrix_free``
     - ``False``
     - Never store the Hamiltonian, the :math:`c^\dagger_i c_j` operators or the
       spin operators; apply them on the fly instead. Memory per sector drops
       from :math:`O(\dim \cdot \texttt{norb}^2)` to :math:`O(\dim)`, at a few
       times the cost per matvec.
   * - ``mf_lookup_max_norb``
     - ``22``
     - In matrix-free mode, build a direct Fock-state lookup table of
       :math:`2^{\texttt{norb}}` entries when ``norb`` is at most this,
       otherwise fall back to a binary search per applied term.
   * - ``use_mpi``
     - ``True``
     - Set to ``False`` to force the serial path even when mpi4py is available.
   * - ``mpi_weight_exp``
     - ``3`` (``1`` if ``matrix_free``)
     - Exponent in the sector cost model ``dim**mpi_weight_exp`` used by the
       greedy load balancing across ranks. The default assumes dense
       diagonalisation dominates; in matrix-free mode the cost is a number of
       matvecs and scales linearly.

Two caveats on the spin penalties. They are only meaningful at :math:`T = 0` —
they change the spectrum, so at :math:`T > 0` the Boltzmann weights come out
wrong, and a warning is issued. And they are unavailable in matrix-free mode,
which does not build :math:`\hat{S}^2`, :math:`\hat{S}_x`, :math:`\hat{S}_y`,
:math:`\hat{S}_z` at all; combining the two raises ``NotImplementedError``.
They can also be passed directly to ``build_Hemb``, in which case the explicit
argument wins over the ``solver_params`` entry.


Utilities
=========

.. autosummary::
   :toctree: _autosummary
   :template: autosummary_module_template.rst
   :recursive:


Helper routines used throughout the code:

* :mod:`gem.utilities` — general linear-algebra and Green's-function helpers.
* :mod:`gem.delta_fit` — routines to perform thermal density matrix fitting of the hybridisation function and self-energy parameters.
  to a discrete bath.
* :mod:`gem.mpi` — common helper routines for MPI parallelization.
