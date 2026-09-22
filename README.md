# GEM - Ghost Embedding Method

> **Disclaimer:** This software is in **beta stage**. It is provided as-is, and no guarantee is made that it works for your use case or is free from bugs.
>
> Use at your own risk and verify results independently.

-----

> **Disclaimer:** The software will have soon a paper associated to it. In the meantime, if you use it for your own research, please cite Ref. [1,2]
>

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://triqs.github.io/gem)
[![docs](https://github.com/triqs/gem/actions/workflows/docs.yml/badge.svg)](https://github.com/triqs/gem/actions/workflows/docs.yml)

Start to learn about GEM on our website at [https://triqs.github.io/gem/](triqs.github.io/gem).

About
-----

This is an implement of the ghost-Gutwiller approximation and its finite temperature extension [1,2] (ghost-GA).

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

### Testing

| Package | Purpose |
|---------|---------|
| pytest | Test runner (`make test`) |

Initial Setup
-------------

To install this package, run the following commands in order:

```bash
git clone https://github.com/TRIQS/gem.git

mkdir gem.build && cd gem.build

cmake ../gem

make
make test
make install
```

### Environment setup

GEM provides a small script (`gemvars.sh`)
to load its installation into your environment variables.
Please source it with the proper replacement of (`INSTALL_PREFIX`):

        source $INSTALL_PREFIX/share/gem/gemvars.sh

To automate this process, please add this line to your [~/.bash_profile](<https://en.wikipedia.org/wiki/Bash_(Unix_shell)#Startup_scripts>)
(or [~/.zprofile](http://zsh.sourceforge.net/FAQ/zshfaq03.html#l19>)).


References
----------

[1]: N. Lanatà, T.-H. Lee, Y.-X. Yao, and V. Dobrosavljević, [Emergent Bloch excitations in Mott matter, Phys. Rev. B 96, 195126 (2017).](https://doi.org/10.1103/PhysRevB.96.195126)

[2]: S. Giuli, T.-H. Lee, Y.-X. Yao, G. Kotliar, A. E. Ruckenstein, O. Gingras, and N. Lanatà, [Unifying Variational and Dynamical Quantum Embedding: From Ghost Gutzwiller Approximation to Dynamical Mean-Field Theory, arXiv:2603.20559.](https://doi.org/10.48550/arXiv.2603.20559)


Support
-------

triqs/gem is supported by the [Flatiron Institute](https://www.simonsfoundation.org/flatiron/), a division of the [Simons Foundation](https://www.simonsfoundation.org/).

<picture>
  <source media="(prefers-color-scheme: dark)" width="20%" srcset="doc/_static/logo_ccq.png">
  <img alt="Flatiron Center for Computational Quantum Physics logo." width="20%" src="doc/_static/logo_ccq.png">
</picture>

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" width="20%" srcset="doc/_static/logo_simons.png">
  <img alt="Simons Foundation logo." width="20%" src="doc/_static/logo_simons.png">
</picture>

----------------
