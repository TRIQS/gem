.. _example_square_phase:

Square Lattice — AFM Phase Diagram (Finite Temperature)
=======================================================

**Script:** ``examples/Square_1orb_2frag_phase.py``

**Plotting:** ``examples/plot_phase.py``

This example maps out the finite-temperature antiferromagnetic (AFM)
phase diagram of the single-orbital Hubbard model on the square lattice.
For each value of :math:`U`, the staggered magnetisation :math:`|m|` is
tracked as a function of temperature :math:`T`, sweeping from low to
high :math:`T` on a linear grid.  The result is a dataset from which
the critical temperature :math:`T_c(U)` can be extracted, tracing the
boundary between the AFM ordered phase and the paramagnetic phase in the
:math:`(U, T)` plane.

Setup
-----

We import the necessary modules::

    import numpy as np
    from gem.fragment import Fragment
    from gem.lattice  import Lattice
    from gem.solvers.simple_ed import SimpleED
    import h5py

The square-lattice setup follows the same structure as
:ref:`example_square_afm`, but we use a larger bath (``B = 3`` orbitals
per spin channel) for better accuracy at finite temperature, and a finer
k-grid (``Nk = 200``) to resolve the Fermi surface::

    B     = 3
    nimp  = 2
    nbath = nimp * B
    ntot  = nimp + nbath

    Nk = 200
    Kx = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    Ky = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    t  = 0.25

    ek_list = []
    for kx in Kx:
        for ky in Ky:
            gamma_k = -t * (1.0
                        + np.exp(-1j * (kx + ky))
                        + np.exp(-1j * ky)
                        + np.exp(-1j * kx))
            Hk_spinless = np.array([[0.0, gamma_k],
                                    [gamma_k.conj(), 0.0]], dtype=np.complex128)
            ek_list.append(np.kron(Hk_spinless, np.eye(2)))

    eks = np.array(ek_list)
    wks = np.ones(len(ek_list)) / len(ek_list)
    lattice = Lattice(eks, wk_list=wks)

For each value of :math:`U`, the magnetisation is computed on a
temperature grid and accumulated in the array ``mag_grid``::

    U_list = np.array([0.5, 1.0, 1.5, 2.0])
    T_list = np.linspace(0.02, 0.12, 21)

    mag_grid = np.zeros((len(U_list), len(T_list)))

Seeding AFM order
-----------------

The AFM seed field is applied only at the **lowest temperature point**
(``iT == 0``) for the first few self-consistency iterations, with
opposite sign on the two sublattices::

    if iT == 0 and it < 3:
        fragmentA.eloc = eloc + bfield * np.diag([-1,  1])
        fragmentB.eloc = eloc - bfield * np.diag([-1,  1])
    else:
        fragmentA.eloc = eloc.copy()
        fragmentB.eloc = eloc.copy()

This is sufficient because the warm-start strategy (see below) propagates
the ordered solution from one temperature to the next without requiring a
bias field at each :math:`T`.

Warm-starting across temperatures
----------------------------------

A crucial ingredient of this calculation is warm-starting: the converged
variational parameters :math:`R` and :math:`\Lambda` from one temperature
point are used as the initial guess for the next, higher temperature.
This dramatically reduces the number of iterations needed near the
transition and allows the ordered solution to survive into the high-:math:`T`
regime until it is destabilised by thermal fluctuations.

The warm-start is implemented by carrying the converged matrices between
iterations of the temperature loop::

    Lambda_A = L_t0.copy()
    R_A      = R_t0.copy()
    Lambda_B = -L_t0.copy()   # opposite sign seeds opposite magnetisation
    R_B      = R_t0.copy()

    for iT, T in enumerate(T_list):
        fragmentA = Fragment(..., Lambda=Lambda_A, R=R_A)
        fragmentB = Fragment(..., Lambda=Lambda_B, R=R_B)

        # ... self-consistency loop ...

        Lambda_A = fragmentA.Lambda.copy()
        R_A      = fragmentA.R.copy()
        Lambda_B = fragmentB.Lambda.copy()
        R_B      = fragmentB.R.copy()

Note that the initial guess for :math:`\Lambda_B` is set to minus the
guess for :math:`\Lambda_A`, encoding the antiferromagnetic ansatz from
the start.

Storing results
---------------

After each :math:`U` is completed, the full one-body density matrix, the
variational parameters :math:`R` and :math:`\Lambda`, and the
quasiparticle weight :math:`Z` for both fragments are written to an HDF5
file, one group per :math:`U` value::

    with h5py.File('Square_1orb_2frag_phase.h5', 'a') as h5f:
        grp = h5f.create_group(f'U_{U:.2f}')
        grp.create_dataset('T_list',   data=T_list)
        grp.create_dataset('denMat_A', data=arr_denMat_A)  # shape (nT, nimp, nimp)
        grp.create_dataset('denMat_B', data=arr_denMat_B)
        grp.create_dataset('R_A',      data=arr_R_A)       # shape (nT, nbath, nimp)
        grp.create_dataset('R_B',      data=arr_R_B)
        grp.create_dataset('Lambda_A', data=arr_Lambda_A)  # shape (nT, nbath, nbath)
        grp.create_dataset('Lambda_B', data=arr_Lambda_B)
        grp.create_dataset('Z_A',      data=arr_Z_A)       # shape (nT, nimp, nimp)
        grp.create_dataset('Z_B',      data=arr_Z_B)

Each group is flushed to disk as soon as the :math:`U` sweep completes,
so partial runs are not lost if the script is interrupted.

Mean-field criticality fit
---------------------------

The companion script ``plot_phase.py`` reads the HDF5 file and fits the
mean-field order-parameter form

.. math::

   |m(T)| = A\,(T_c - T)^{\beta}

to the magnetisation near each critical temperature.  Only data points in
the critical region (:math:`|m| < 0.4\,|m|_\mathrm{max}`) are included
in the fit to avoid the saturated low-:math:`T` regime.

Results
-------

The order parameter :math:`|m|` as a function of temperature for
different values of :math:`U` is shown below.  The critical temperature
:math:`T_c` is identified as the temperature at which :math:`|m|` goes
to zero, and it increases with :math:`U` in the range studied.

.. image:: ./images/Example_3_fig_1.png
   :width: 70%
   :align: center

The full phase boundary :math:`T_c(U)` is mapped out in the
:math:`(U, T)` plane, shown as both an extracted curve and a colour-map
of :math:`|m|`:

.. image:: ./images/Example_3_fig_2.png
   :width: 80%
   :align: center

The extracted critical exponent :math:`\beta` is consistent with the
mean-field value :math:`\beta = 1/2`, as expected for a ghost-GA
calculation which does not include fluctuation corrections beyond
mean-field.
