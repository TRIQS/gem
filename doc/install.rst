.. highlight:: bash

.. _install:

Installation
************

Dependencies
============

Core
----

The following packages are required for the core functionality of the software:

+----------+----------------------------------------------+
| Package  | Purpose                                      |
+==========+==============================================+
| numpy    | Array computing                              |
+----------+----------------------------------------------+
| scipy    | Sparse matrices, linear algebra, optimization|
+----------+----------------------------------------------+
| numba    | JIT compilation for ED basis construction    |
+----------+----------------------------------------------+
| h5py     | HDF5 file I/O                                |
+----------+----------------------------------------------+

Solvers
-------

Install only the packages corresponding to the solvers you intend to use:

+-------------+---------------------------------------------------------------+
| Solver      | Extra packages required                                       |
+=============+===============================================================+
| SimpleED    | *(none)*                                                      |
+-------------+---------------------------------------------------------------+
| PySCF       | `pyscf <https://pyscf.org>`_                                  |
+-------------+---------------------------------------------------------------+
| PyBlock2    | `pyblock2 <https://block2.readthedocs.io>`_, block2           |
+-------------+---------------------------------------------------------------+
| EDIpack     | `edipack2triqs <https://github.com/edipack/edipack2triqs>`_,  |
|             | mpi4py                                                        |
+-------------+---------------------------------------------------------------+
| MPS/ITensor | `juliacall <https://juliapy.github.io/PythonCall.jl>`_,       |
|             | Julia ≥ 1.9, ITensors.jl — see ``README_JULIA.txt``           |
+-------------+---------------------------------------------------------------+

Testing
-------

+--------+-----------------------------+
| Package| Purpose                     |
+========+=============================+
| pytest | Test runner (``make test``) |
+--------+-----------------------------+

Installation Steps
==================

#. Clone the repository::

     git clone https://github.com/TRIQS/ghostGA.git

#. Create a build directory and run CMake::

     mkdir ghostGA.build && cd ghostGA.build
     cmake ../ghostGA

#. Build and install::

     make
     make test
     make install

Custom CMake Options
====================

The build can be configured with CMake options::

    cmake ../ghostGA -DOPTION1=value1 -DOPTION2=value2 ...

+--------------------------------------------------------------+-----------------------------------------------+
| Option                                                       | Syntax                                        |
+==============================================================+===============================================+
| Specify a custom installation prefix                         | ``-DCMAKE_INSTALL_PREFIX=<path>``             |
+--------------------------------------------------------------+-----------------------------------------------+
| Build in debug mode                                          | ``-DCMAKE_BUILD_TYPE=Debug``                  |
+--------------------------------------------------------------+-----------------------------------------------+
| Disable tests (not recommended)                              | ``-DBuild_Tests=OFF``                         |
+--------------------------------------------------------------+-----------------------------------------------+
| Build the documentation                                      | ``-DBuild_Documentation=ON``                  |
+--------------------------------------------------------------+-----------------------------------------------+

MPS / ITensor Solver (Julia)
============================

The MPS solver relies on ITensors.jl through ``juliacall``. See ``README_JULIA.txt``
in the repository root for step-by-step setup instructions specific to the Julia
environment.
