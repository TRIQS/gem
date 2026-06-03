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

|bold_project_name| comes with a simple internal exact diagonlization solver and with interfaces to external solvers
If you want to use an external solver then install the packages corresponding to the solvers you intend to use.
When using a solver different from ``SimpleED`` please make sure to cite the appropriate references along the |bold_project_name| one.

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

The following python packages are required to build and run the tests.
To enable building and runnning the tests locally pass the appropriate flag to cmake (``cmake -DBuild_Tests=ON``):

+--------+-----------------------------+
| Package| Purpose                     |
+========+=============================+
| pytest | Test runner (``make test``) |
+--------+-----------------------------+

Documentation
-------------

The following python packages are required to build the HTML documentation.
To enable building the documentation locally pass the appropriate flag to cmake (``cmake -DBuild_Documentation=ON``):

+---------------------+--------------------------------------------+
| Package             | Purpose                                    |
+=====================+============================================+
| sphinx-rtd-theme    | Read the Docs HTML theme                   |
+---------------------+--------------------------------------------+
| nbsphinx            | Jupyter notebook support                   |
+---------------------+--------------------------------------------+
| myst-parser         | Markdown source file support               |
+---------------------+--------------------------------------------+
| sphinx-autorun      | Executable code blocks in docs             |
+---------------------+--------------------------------------------+
| numpydoc            | NumPy-style docstring parser               |
+---------------------+--------------------------------------------+
| linkify-it-py       | Auto-linking of bare URLs                  |
+---------------------+--------------------------------------------+
| matplotlib          | Plots in documentation via plot directive  |
+---------------------+--------------------------------------------+
| sphinxcontrib-bibtex| BibTeX citations and reference list        |
+---------------------+--------------------------------------------+

Installation Steps
==================

#. Clone the repository::

     git clone https://github.com/TRIQS/gem.git

#. Create a build directory and run CMake::

     mkdir gem.build && cd gem.build
     cmake ../gem

#. Build and install::

     make
     make test
     make install

Custom CMake Options
====================

The build can be configured with the following CMake options::

    cmake ../gem -DOPTION1=value1 -DOPTION2=value2 ...

+--------------------------------------------------------------+-----------------------------------------------+
| Option                                                       | Syntax                                        |
+==============================================================+===============================================+
| Specify a custom installation prefix                         | ``-DCMAKE_INSTALL_PREFIX=<path>``             |
+--------------------------------------------------------------+-----------------------------------------------+
| Build in debug mode                                          | ``-DCMAKE_BUILD_TYPE=Release/Debug``          |
+--------------------------------------------------------------+-----------------------------------------------+
| Disable tests (not recommended)                              | ``-DBuild_Tests=OFF/ON``                      |
+--------------------------------------------------------------+-----------------------------------------------+
| Build the documentation                                      | ``-DBuild_Documentation=OFF/ON``              |
+--------------------------------------------------------------+-----------------------------------------------+
| Path to sphinx                                               | ``-DSPHINXBUILD_EXECUTABLE=<path>``           |
+--------------------------------------------------------------+-----------------------------------------------+

