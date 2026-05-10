# TRIQS ghostGA

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://triqs.github.io/ghostGA)
[![docs](https://github.com/TRIQS/ghostGA/actions/workflows/docs.yml/badge.svg)](https://github.com/TRIQS/ghostGA/actions/workflows/docs.yml)

About
-----

This is an implement of the ghost-Gutwiller approximation [1,2] (ghost-GA) based
on the TRIQS library.

Dependencies
------------

### Core

| Package | Purpose |
|---------|---------|
| numpy | Array computing |
| scipy | Sparse matrices, linear algebra, optimization |
| numba | JIT compilation for ED basis construction |
| h5py | HDF5 file I/O |

### Solvers (install only what you use)

| Solver | Extra packages required |
|--------|------------------------|
| `SimpleED` | *(none beyond core)* |
| `PySCF` | [pyscf](https://pyscf.org) |
| `PyBlock2` | [pyblock2](https://block2.readthedocs.io), block2 |
| `EDIpack` | [edipack2triqs](https://github.com/edipack/edipack2triqs), mpi4py |
| `MPS` (ITensor) | [juliacall](https://juliapy.github.io/PythonCall.jl), Julia ≥ 1.9, ITensors.jl — see `README_JULIA.txt` |

### Testing

| Package | Purpose |
|---------|---------|
| pytest | Test runner (`make test`) |

Initial Setup
-------------

To install this package, run the following commands in order:

```bash
git clone https://github.com/TRIQS/ghostGA.git

mkdir ghostGA.build && cd ghostGA.build

cmake ../ghostGA

make
make test
make install
```

### References ###

[1]: N. Lanatà, T.-H. Lee, Y.-X. Yao, and V. Dobrosavljević, [Emergent Bloch excitations in Mott matter, Phys. Rev. B 96, 195126 (2017).](https://doi.org/10.1103/PhysRevB.96.195126)

[2]: S. Giuli, T.-H. Lee, Y.-X. Yao, G. Kotliar, A. E. Ruckenstein, O. Gingras, and N. Lanatà, [Unifying Variational and Dynamical Quantum Embedding: From Ghost Gutzwiller Approximation to Dynamical Mean-Field Theory](https://doi.org/10.48550/arXiv.2603.20559)

----------------
