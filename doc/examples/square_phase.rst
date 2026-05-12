.. _example_square_phase:

Square Lattice — AFM Phase Diagram (Finite Temperature)
=======================================================

**Script:** ``examples/Square_1orb_2frag_phase.py``

**Plotting:** ``examples/plot_phase.py``

This example maps out the finite-temperature antiferromagnetic (AFM) phase
diagram of the single-orbital Hubbard model on the square lattice.  For each
value of :math:`U` the staggered magnetisation :math:`|m|` is tracked as a
function of temperature :math:`T`, sweeping from low to high :math:`T` on a
logarithmic grid.

Warm-starting across temperatures
----------------------------------

For each :math:`U`, the self-consistency loop is run at successively higher
temperatures.  The converged :math:`\Lambda` and :math:`R` matrices from one
temperature point are used as the starting guess for the next, reducing the
number of iterations needed near the transition::

    Lambda_A = Lambda0.copy()
    R_A      = R0.copy()
    Lambda_B = -Lambda0.copy()
    R_B      = R0.copy()

    for iT, T in enumerate(T_list):
        fragmentA = Fragment(..., Lambda=Lambda_A, R=R_A)
        fragmentB = Fragment(..., Lambda=Lambda_B, R=R_B)

        # ... self-consistency loop ...

        Lambda_A = fragmentA.Lambda.copy()
        R_A      = fragmentA.R.copy()
        Lambda_B = fragmentB.Lambda.copy()
        R_B      = fragmentB.R.copy()

The AFM seed field is applied only at the lowest temperature point
(``iT == 0``) for the first few iterations; the warm start propagates the
ordered solution to higher :math:`T` without requiring a bias field.

Storing results
---------------

After each :math:`U` is completed, the full one-body density matrix, the
variational parameters :math:`R` and :math:`\Lambda`, and the quasiparticle
weight :math:`Z` for both fragments are written to an HDF5 file, one group
per :math:`U` value::

    with h5py.File('Square_1orb_2frag_phase.h5', 'a') as h5f:
        grp = h5f.create_group(f'U_{U:.2f}')
        grp.create_dataset('T_list',   data=T_list)
        grp.create_dataset('denMat_A', data=arr_denMat_A)   # (nT, ntot, ntot)
        grp.create_dataset('R_A',      data=arr_R_A)        # (nT, nimp, nbath)
        grp.create_dataset('Lambda_A', data=arr_Lambda_A)   # (nT, nimp, nimp)
        grp.create_dataset('Z_A',      data=arr_Z_A)        # (nT, nimp, nimp)
        # same for fragment B ...

Each group is flushed to disk as soon as the :math:`U` sweep completes, so
partial runs are not lost if the script is interrupted.

Mean-field criticality fit
---------------------------

The companion plotting script ``plot_phase.py`` reads the HDF5 file and fits
the mean-field order-parameter form

.. math::

   |m(T)| = A\,(T_c - T)^{\beta}

to the magnetisation near each critical temperature.  Only data points in the
critical region (:math:`|m| < 0.4\,|m|_\mathrm{max}`) are included in the
fit to avoid the saturated low-:math:`T` regime.  The script produces:

* :math:`|m|` vs :math:`T` with the fitted curves overlaid.
* A colour-map of :math:`|m|` in the :math:`(U,\,T)` plane (phase diagram).
* Extracted :math:`T_c(U)` and :math:`\beta(U)` with error bars.
