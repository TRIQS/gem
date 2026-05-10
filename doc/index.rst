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
correlated electron systems [Lanata2017]_. It extends the standard Gutzwiller approximation
by introducing auxiliary *ghost* orbitals in the embedding Hamiltonian, which allow
the method to capture the full quasiparticle structure of Mott insulators and
correlated metals, including satellite features and non-trivial spectral weight
distributions.

Unlike standard slave-boson or Gutzwiller methods, ghost-GA systematically converges
towards the exact result as the number of ghost orbitals is increased, and it
reproduces the Dynamical Mean-Field Theory (DMFT) solution [Giuli2025]_ in the limit of a
infinite ghost-orbital basis.

Finite-Temperature Extension
=====================================

The finite-temperature extension of ghost-GA provides a unified variational framework that interpolates between the
zero-temperature ghost-GA and full DMFT [Giuli2025]_. It retains the computational
efficiency of ghost-GA while accounting for thermal fluctuations, making it suitable
for calculations of thermodynamic properties and finite-temperature phase diagrams
in correlated materials.

.. rubric:: References

.. [Lanata2017] N. Lanatà, T.-H. Lee, Y.-X. Yao, and V. Dobrosavljević,
   *Emergent Bloch excitations in Mott matter*,
   Phys. Rev. B **96**, 195126 (2017).
   https://doi.org/10.1103/PhysRevB.96.195126

.. [Giuli2025] S. Giuli, T.-H. Lee, Y.-X. Yao, G. Kotliar, A. E. Ruckenstein, O. Gingras, and N. Lanatà,
   *Unifying Variational and Dynamical Quantum Embedding: From Ghost Gutzwiller Approximation
   to Dynamical Mean-Field Theory*,
   arXiv:2603.20559.
   https://doi.org/10.48550/arXiv.2603.20559


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents

   install
   documentation
   user_guide
   ChangeLog.md
   about
