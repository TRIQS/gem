.. _example_square_afm:

Square Lattice — Antiferromagnetic Order (T = 0)
================================================

**Script:** ``examples/Square_1orb_2frag.py``

This example looks for antiferromagnetic (AFM) order in the single-orbital
Hubbard model on the square lattice at zero temperature.  A two-site unit
cell with one fragment per sublattice is used, and spin SU(2) symmetry is
*not* imposed, allowing the two fragments to develop opposite spin
polarisations.

The calculation sweeps :math:`U` and extracts both the quasiparticle weight
:math:`Z` and the staggered magnetisation :math:`m = n_\uparrow - n_\downarrow`
from the converged impurity density matrix.

Seeding AFM order
-----------------

A small symmetry-breaking field is applied to the on-site energies during
the first few iterations to seed the AFM solution.  The field has opposite
sign on the two sublattices::

    if it < 3:
        fragmentA.eloc = eloc + bfield * np.diag([-1, 1])
        fragmentB.eloc = eloc - bfield * np.diag([-1, 1])
    else:
        fragmentA.eloc = eloc.copy()
        fragmentB.eloc = eloc.copy()

Once the self-consistency is established the field is removed; if the system
is in the ordered phase the solution remains magnetised.

Extracting the magnetisation
-----------------------------

After convergence the magnetisation on each fragment is read from the
impurity block of the one-body density matrix::

    dm_A = fragmentA.denMat[:nimp, :nimp].real
    dm_B = fragmentB.denMat[:nimp, :nimp].real
    mA = dm_A[0, 0] - dm_A[1, 1]   # n_up - n_down on sublattice A
    mB = dm_B[0, 0] - dm_B[1, 1]   # n_up - n_down on sublattice B

In the ordered phase :math:`m_A = -m_B \neq 0`.
