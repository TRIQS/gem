(changelog)=

# Changelog

## Version 1.0.0

GEM version 1.0.0 is the first release for this project.
* Implementation of the ghost Gutzwiller Approximation, extended to finite temperature.
* Support for multiple fragments.
* MPI parallelization of the impurity solver sectors and of the k-point sums on the lattice. A serial installation without `mpi4py` is still supported.
* Save and load routines for fragments and lattices.
* Impurity solvers provided: SimpleED, built on the `gemSolver` class that all solvers must inherit from.
* Sphinx documentation with API reference, installation guide, and user guide.

### Contributors

The authors of GEM are S. Giuli, T.-H. Lee, Y.-X. Yao, I. Park, H. LaBollita, I. Pasqua, N. Wentzell, N. Lanatà and O. Gingras.

* GEM is based on an original zero-temperature code by N. Lanatà, Y.-X. Yao and T.-H. Lee, adapted by N. Wentzell and O. Gingras.
* The new code structure was designed by S. Giuli, T.-H. Lee, Y.-X. Yao and O. Gingras.
* The new features (finite temperature, multiple fragments and MPI parallelization) were written by S. Giuli.
* Code review and the direction of the project were carried out by S. Giuli and O. Gingras.
* I. Park, H. LaBollita and I. Pasqua beta-tested the code.

We thank all of them for their contributions.
