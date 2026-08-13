#######################################################
# Base class of the solvers of the embedding Hamiltonian in GEM.
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
#######################################################
import copy


class gemSolver(object):
    '''
    Base class of every GEM impurity solver.

    Fragment checks that the solver it receives is a gemSolver, so any new
    solver must inherit from this class. SolverTemplate is a commented skeleton
    to copy, SimpleED a working implementation.

    A subclass is expected to:

    * call ``super().__init__(solver_params=..., solver_type=...)``, which sets
      ``self.type`` and ``self.solver_params``;
    * override every method listed in ``mandatory_methods``.

    Solver-specific parameters are *not* passed by Fragment: they are read from
    ``self.solver_params`` and fixed once when the solver is built, e.g.
    ``num_eig = self.solver_params.get('num_eig', default)``.
    In principle they can be changed from fragment.solver.solver_params.

    The whole surface the rest of GEM touches is:

    * ``build_Hemb(D, eloc, Lambdac, V2E)``, called positionally, and
      ``solve_Hemb(verbose=, T=)``, called with these two keywords only;
    * ``calc_density_matrix()`` and ``compute_E2loc()``, without arguments;
      all four from Fragment.solve_impurity;
    * the attributes ``gs_ene`` (ground-state energy of the embedding
      Hamiltonian) and ``Zpart`` (partition function divided by exp(-gs_ene/T),
      hence 1 at zero temperature), set by solve_Hemb and read by
      Lattice.compute_functional;
    * ``self.type``, only used in the printouts;
    * optionally ``compute_E1loc(nimp)``, used by Fragment.compute_energy, and
      ``calc_double_occ(i)``, used by Gdmft.run.

    Everything else is up to the solver. In particular the constructor arguments
    (dimensions, symmetry sectors, whether the calculation is thermal, ...) are
    decided when the solver is built: the Fragment only ever sees the object and
    never sets them.
    '''

    # methods a solver must override for Fragment.solve_impurity to work
    mandatory_methods = ('build_Hemb', 'solve_Hemb',
                         'calc_density_matrix', 'compute_E2loc')

    def __init__(self, solver_params=None, solver_type=None):
        '''
        :param solver_params: dict, optional. Solver-specific parameters. Keys
            depend on the solver (default: None, i.e. an empty dict). It is
            copied, so later changes to the dict passed here do not affect the
            solver (and vice versa).
        :param solver_type: str, optional. Name of the solver, used in the
            printout (default: None, i.e. the name of the class).
        '''
        if solver_params is not None and not isinstance(solver_params, dict):
            raise TypeError(f"solver_params must be a dict, got {type(solver_params)}")
        self.type = type(self).__name__ if solver_type is None else solver_type
        self.solver_params = {} if solver_params is None else copy.deepcopy(solver_params)

    def missing_methods(self):
        '''
        Names of the mandatory methods that are still the gemSolver stubs,
        i.e. that this solver has not implemented.
        '''
        cls = type(self)
        return tuple(name for name in self.mandatory_methods
                     if getattr(cls, name) is getattr(gemSolver, name))

# -- Mandatory interface: see SolverTemplate for the meaning of the arguments --

    def build_Hemb(self, D, eloc, Lambdac, V2E):
        raise NotImplementedError(
            f"{self.type} does not implement build_Hemb")

    def solve_Hemb(self, verbose=0, T=0.0):
        raise NotImplementedError(
            f"{self.type} does not implement solve_Hemb")

    def calc_density_matrix(self):
        raise NotImplementedError(
            f"{self.type} does not implement calc_density_matrix")

    def compute_E2loc(self):
        raise NotImplementedError(
            f"{self.type} does not implement compute_E2loc")

# -- Optional interface, only needed by some consumers --

    def compute_E1loc(self, nimp):
        '''One-body local energy of the impurity block. Used by Fragment.compute_energy.'''
        raise NotImplementedError(
            f"{self.type} does not implement compute_E1loc")

    def calc_double_occ(self, i):
        '''Double occupancy of the impurity level i. Used by Gdmft.run.'''
        raise NotImplementedError(
            f"{self.type} does not implement calc_double_occ")
