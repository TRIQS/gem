.. _welcome:

TRIQS / ghostGA
***************

.. sidebar:: ghostGA |PROJECT_VERSION|

   Latest release: |PROJECT_VERSION|

   .. image:: _static/logo_github.png
      :width: 75%
      :align: center
      :target: https://github.com/TRIQS/ghostGA


**ghostGA** is a Python implementation of the Ghost Gutzwiller Approximation (ghost-GA) 
and its finite-temperature extension, built on top of the
`TRIQS <https://triqs.github.io>`_ library.

The Ghost Gutzwiller Approximation
===================================

The Ghost Gutzwiller Approximation is a variational embedding method for strongly
correlated electron systems :cite:`Lanata2017`. It extends the standard Gutzwiller approximation
by introducing auxiliary *ghost* orbitals in the embedding Hamiltonian, which allow
the method to capture the full quasiparticle structure of Mott insulators and
correlated metals, including satellite features and non-trivial spectral weight
distributions.

Unlike standard slave-boson or Gutzwiller methods, ghost-GA systematically converges
towards the exact result as the number of ghost orbitals is increased, and it
reproduces the Dynamical Mean-Field Theory (DMFT) solution :cite:`Giuli2026` in the limit of a
infinite ghost-orbital basis.

Finite-Temperature Extension
=====================================

The finite-temperature extension of ghost-GA provides a unified variational framework that interpolates between the
zero-temperature ghost-GA and full DMFT :cite:`Giuli2026`. It retains the computational
efficiency of ghost-GA while accounting for thermal fluctuations, making it suitable
for calculations of thermodynamic properties and finite-temperature phase diagrams
in correlated materials.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents

   install
   documentation
   user_guide
   references
   issues
   ChangeLog.md
   about
