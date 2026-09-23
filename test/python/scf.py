'''The ghost-GA self-consistency loop, as the regression tests drive it.
Transcribed from the ``Gdmft`` driver that used to live in ``gem/gdmft.py``.
'''

import numpy as np

from gem.utilities import calc_nf


def _diff(R_new, L_new, R_old, L_old):
    '''
    Gauge-invariant change in the self-energy parameters, one spin block.
    '''
    L_eval_new, UL_new = np.linalg.eigh(L_new[::2, ::2])
    L_eval_old, UL_old = np.linalg.eigh(L_old[::2, ::2])
    diff_R = np.abs(np.abs(UL_old @ R_old[::2, ::2])
                    - np.abs(UL_new @ R_new[::2, ::2])).max()
    return max(diff_R, np.abs(L_eval_new - L_eval_old).max())


def run_scf(lattice, fragment, mu=0.0, itmax=200, mix=0.5, tol=1e-6, T=1e-3,
            n_target=None, n_tolerance=1e-3, n_fit_method='qp',
            spin_sym=True, orb_sym=False, verbose=False):
    '''Iterate the ghost-GA cycle to convergence.
    '''
    nimp = fragment.nimp
    diff = np.inf

    for it in range(itmax):
        lattice.solve_qp([fragment], T=T)
        fragment.update_hybridization(T=T)
        fragment.solve_impurity(mu, T=T)

        # refit only when the filling has actually drifted; fit_mu returns None
        # when its bracketing fails, and then mu simply stays where it was
        if n_target is not None:
            nfill = np.trace(fragment.denMat[:nimp, :nimp])
            if abs(nfill - n_target) > n_tolerance:
                mu_new = lattice.fit_mu(n_target, [fragment], T=T,
                                        mode=n_fit_method, mu_old=mu,
                                        ntol=n_tolerance)
                if mu_new is not None:
                    mu = mu_new
                    fragment.solve_impurity(mu, T=T)

        Lambda_old, R_old = fragment.Lambda.copy(), fragment.R.copy()
        R_new, Lambda_new = fragment.update_self_energy(T=T)

        diff = _diff(R_new, Lambda_new, R_old, Lambda_old)

        # mixed after the measurement, projected after the mixing
        fragment.Lambda = (1 - mix) * Lambda_new + mix * Lambda_old
        fragment.R = (1 - mix) * R_new + mix * R_old
        if spin_sym:
            fragment.impose_spin_SU2_symmetry()
        if orb_sym:
            fragment.impose_orbital_symmetry()

        if verbose:
            print(f'iteration: {it}  diff={diff:.3e}  mu={mu:.6f}')

        if (diff < tol and it > 2) or it == itmax - 1:
            break

    docc = [fragment.solver.calc_double_occ(i) for i in range(0, nimp, 2)]
    return mu, docc, diff


def total_energy(lattice, fragment, beta, mu=0.0):
    '''Total energy of the converged solution.
    '''
    R, Lambda = fragment.R, fragment.Lambda
    # the bracketing is the original one: R @ (ek @ R^dag), not (R @ ek) @ R^dag,
    # which rounds differently
    ekin = sum(np.sum((R @ (ek @ R.conj().T))
                      * calc_nf(R @ (ek @ R.conj().T) + Lambda, 1. / beta).T) * wk
               for ek, wk in zip(lattice.eks, lattice.wks))
    nimp = fragment.nimp
    epot = (fragment.E2loc
            + np.trace(fragment.eloc @ fragment.denMat[:nimp, :nimp].T))
    nfill = np.trace(fragment.denMat[:nimp, :nimp])
    return ekin + epot - mu * nfill
