######################################################################
# Post-processing for Example 2.
#
# Scans an h5 file produced by Square_1orb_2frag_phase.py, picks up every
# U group available at a given B, and extracts the critical temperature
# and the order-parameter exponent from
#
#       m(T) = A * (Tc - T)^beta .
#
# The power law only holds asymptotically close to Tc: fitting the whole
# ordered branch mixes in the saturated low-T region and biases beta down
# (the previous fits, using m < 0.7*m_max, returned beta ~ 0.33 instead of
# the mean-field 0.5). Here the fit is restricted to a window in reduced
# temperature, t = (Tc - T)/Tc <= t_max, with Tc determined self
# consistently, and the whole thing is repeated for a sequence of shrinking
# t_max so that the approach of beta to its asymptotic value is visible.
#
# Examples
#   python fit_Tc.py                       # B=3, default file name
#   python fit_Tc.py -B 1
#   python fit_Tc.py -f data_Square_1orb_2frag_B3_phase.h5 --save
#   python fit_Tc.py --t-max 0.6 0.4 0.3 0.2 0.1 --min-points 5
######################################################################

import argparse
import os
import re

import numpy as np
import h5py
from scipy.optimize import curve_fit

GROUP_RE = re.compile(r'^U(?P<U>[-+0-9.eE]+)_B(?P<B>\d+)$')


# ---------------------------------------------------------------- loading ----

def load_scans(h5_file, B):
    """Return {U: (T, m)} for every group of the file that matches this B.

    m is the staggered magnetisation, (m_A - m_B)/2 when both sublattices are
    stored and |m_A| otherwise.
    """
    scans = {}
    with h5py.File(h5_file, 'r') as h5f:
        for key in h5f:
            match = GROUP_RE.match(key)
            if match is None or int(match.group('B')) != B:
                continue
            grp = h5f[key]
            T = np.asarray(grp['T_list'][:], dtype=float)

            if 'mag_grid' in grp:                       # current layout
                mag = np.asarray(grp['mag_grid'][:]).real
                m = 0.5 * (mag[0] - mag[1])
            elif 'denMat_A' in grp:                     # older runs
                dmA = np.asarray(grp['denMat_A'][:]).real
                mA = dmA[:, 0, 0] - dmA[:, 1, 1]
                if 'denMat_B' in grp:
                    dmB = np.asarray(grp['denMat_B'][:]).real
                    mB = dmB[:, 0, 0] - dmB[:, 1, 1]
                    m = 0.5 * (mA - mB)
                else:
                    m = mA
            else:
                print(f'  skipping {key}: no magnetisation stored')
                continue

            order = np.argsort(T)
            scans[float(match.group('U'))] = (T[order], np.abs(m)[order])
    return scans


# ---------------------------------------------------------------- fitting ----

def estimate_Tc(T, m, m_min, frac=0.6):
    """Rough Tc from a straight-line fit of m^2 vs T.

    Mean field gives m^2 ~ (Tc - T), so the zero crossing of the line through
    the points that have already lost most of their moment locates Tc. Used
    only to seed the non-linear fits below.
    """
    mask = (m > m_min) & (m < frac * m.max())
    if mask.sum() < 2:
        mask = m > m_min
    if mask.sum() < 2:
        return None
    slope, intercept = np.polyfit(T[mask], m[mask] ** 2, 1)
    if slope >= 0:
        return None
    Tc = -intercept / slope
    if not (T[mask].max() < Tc <= 2.0 * T.max()):
        return None
    return Tc


def fit_window(T, m, Tc_seed, t_max, m_min, min_points, beta_fixed=None):
    """Power-law fit restricted to (Tc - T)/Tc <= t_max, Tc self consistent.

    The fit is done on log m vs log(Tc - T), which weights the critical region
    (where m is small) as much as the rest of the window. Returns None if the
    window never holds enough points or the fit does not converge.
    """
    Tc = Tc_seed
    result = None

    for _ in range(50):
        mask = (m > m_min) & (T < Tc) & ((Tc - T) <= t_max * Tc)
        npts = int(mask.sum())
        free = 2 if beta_fixed is None else 1
        if npts < max(min_points, free + 1):
            return None

        Tf, mf = T[mask], m[mask]
        # Tc has to stay above every temperature in the window
        Tc_low = Tf.max() + 1e-12
        Tc_high = max(2.0 * T.max(), Tc_low * 1.5)

        if beta_fixed is None:
            def model(x, logA, Tc_, beta):
                return logA + beta * np.log(np.clip(Tc_ - x, 1e-14, None))
            p0 = [np.log(max(m.max(), 1e-12)), max(Tc, Tc_low * 1.0001), 0.5]
            bounds = ([-50.0, Tc_low, 0.05], [50.0, Tc_high, 3.0])
        else:
            def model(x, logA, Tc_):
                return logA + beta_fixed * np.log(np.clip(Tc_ - x, 1e-14, None))
            p0 = [np.log(max(m.max(), 1e-12)), max(Tc, Tc_low * 1.0001)]
            bounds = ([-50.0, Tc_low], [50.0, Tc_high])

        p0[1] = min(max(p0[1], Tc_low * (1 + 1e-9)), Tc_high)
        try:
            popt, pcov = curve_fit(model, Tf, np.log(mf), p0=p0,
                                   bounds=bounds, maxfev=20000)
        except Exception as exc:
            print(f'    fit failed (t_max={t_max}): {exc}')
            return None

        perr = np.sqrt(np.abs(np.diag(pcov)))
        beta = popt[2] if beta_fixed is None else beta_fixed
        beta_err = perr[2] if beta_fixed is None else 0.0
        result = dict(A=float(np.exp(popt[0])), Tc=float(popt[1]), beta=float(beta),
                      Tc_err=float(perr[1]), beta_err=float(beta_err),
                      npts=npts, t_max=float(t_max),
                      T_lo=float(Tf.min()), T_hi=float(Tf.max()))

        if abs(popt[1] - Tc) < 1e-10:
            break
        Tc = popt[1]

    return result


def analyse(T, m, t_max_list, m_min_abs, m_min_rel, min_points):
    """Window scan at one U. Returns None when there is no ordered phase."""
    m_max = m.max()
    m_min = max(m_min_abs, m_min_rel * m_max)
    if m_max < 5.0 * m_min_abs:
        return None

    Tc_seed = estimate_Tc(T, m, m_min)
    if Tc_seed is None:
        return None

    scan = []
    for t_max in t_max_list:
        res = fit_window(T, m, Tc_seed, t_max, m_min, min_points)
        if res is not None:
            scan.append(res)
            Tc_seed = res['Tc']          # warm start for the next, smaller window
    if not scan:
        return None

    best = scan[-1]                      # narrowest window that still had points
    fixed = fit_window(T, m, best['Tc'], best['t_max'], m_min, min_points,
                       beta_fixed=0.5)

    # linear extrapolation beta(t_max -> 0), a crude check that the residual
    # drift with the window size is small
    beta_0 = None
    if len(scan) >= 3:
        tm = np.array([r['t_max'] for r in scan])
        bt = np.array([r['beta'] for r in scan])
        beta_0 = float(np.polyval(np.polyfit(tm, bt, 1), 0.0))

    return dict(best=best, scan=scan, fixed=fixed, beta_0=beta_0,
                m_max=float(m_max), m_min=float(m_min))


# ------------------------------------------------------------------- main ----

def main():
    parser = argparse.ArgumentParser(
        description='Extract Tc and beta from the E2 temperature scans.')
    parser.add_argument('-B', '--nghost', type=int, default=3,
                        help='number of ghost bath sites per orbital (default 3)')
    parser.add_argument('-f', '--file', default=None,
                        help='h5 file (default data_Square_1orb_2frag_B<B>_phase.h5 '
                             'next to this script)')
    parser.add_argument('--t-max', type=float, nargs='+',
                        default=[0.5, 0.4, 0.3, 0.2, 0.15, 0.1],
                        help='reduced-temperature windows, largest first')
    parser.add_argument('--min-points', type=int, default=4,
                        help='smallest number of points accepted in a window')
    parser.add_argument('--m-min', type=float, default=2e-3,
                        help='absolute noise floor on the order parameter')
    parser.add_argument('--m-min-rel', type=float, default=0.0,
                        help='noise floor relative to max(m), combined with --m-min')
    parser.add_argument('--save', action='store_true',
                        help='write the fits back into the h5 file')
    args = parser.parse_args()

    B = args.nghost
    h5_file = args.file
    if h5_file is None:
        h5_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f'data_Square_1orb_2frag_B{B}_phase.h5')
    if not os.path.exists(h5_file):
        raise SystemExit(f'{h5_file} not found')

    scans = load_scans(h5_file, B)
    if not scans:
        raise SystemExit(f'no U group with B={B} in {h5_file}')

    U_list = np.array(sorted(scans))
    print(f'{h5_file}: B={B}, U = ' + ', '.join(f'{U:.2f}' for U in U_list))

    t_max_list = sorted(args.t_max, reverse=True)
    fits = {}
    for U in U_list:
        T, m = scans[U]
        print(f'\nU = {U:.2f}   {len(T)} temperatures, max|m| = {m.max():.4f}')
        res = analyse(T, m, t_max_list, args.m_min, args.m_min_rel,
                      args.min_points)
        fits[U] = res
        if res is None:
            print('  no ordered phase / not enough points to fit')
            continue

        print('   t_max   pts   T window            Tc                  beta')
        for r in res['scan']:
            print(f'   {r["t_max"]:.2f}   {r["npts"]:3d}   '
                  f'[{r["T_lo"]:.4f},{r["T_hi"]:.4f}]   '
                  f'{r["Tc"]:.5f}+-{r["Tc_err"]:.5f}   '
                  f'{r["beta"]:.4f}+-{r["beta_err"]:.4f}')
        best = res['best']
        print(f'  -> Tc = {best["Tc"]:.5f} +- {best["Tc_err"]:.5f}   '
              f'beta = {best["beta"]:.4f} +- {best["beta_err"]:.4f}   '
              f'(t_max = {best["t_max"]:.2f}, {best["npts"]} points)')
        if res['beta_0'] is not None:
            print(f'     beta extrapolated to t_max -> 0: {res["beta_0"]:.4f}')
        if res['fixed'] is not None:
            print(f'     Tc at beta fixed to 1/2: {res["fixed"]["Tc"]:.5f} '
                  f'+- {res["fixed"]["Tc_err"]:.5f}')

    good = [U for U in U_list if fits[U] is not None]
    print('\n' + '-' * 68)
    print(f'{"U":>6} {"Tc":>10} {"Tc_err":>10} {"beta":>8} {"beta_err":>9} '
          f'{"Tc(b=1/2)":>11}')
    for U in good:
        best = fits[U]['best']
        fixed = fits[U]['fixed']
        Tc_fixed = fixed['Tc'] if fixed is not None else np.nan
        print(f'{U:6.2f} {best["Tc"]:10.5f} {best["Tc_err"]:10.5f} '
              f'{best["beta"]:8.4f} {best["beta_err"]:9.4f} {Tc_fixed:11.5f}')

    if args.save and good:
        with h5py.File(h5_file, 'a') as h5f:
            name = f'fits_B{B}'
            if name in h5f:
                del h5f[name]
            grp = h5f.create_group(name)
            grp.create_dataset('U_list', data=np.array(good))
            for key in ('A', 'Tc', 'Tc_err', 'beta', 'beta_err', 't_max'):
                grp.create_dataset(key,
                                   data=np.array([fits[U]['best'][key] for U in good]))
            grp.create_dataset(
                'Tc_beta_half',
                data=np.array([fits[U]['fixed']['Tc'] if fits[U]['fixed'] else np.nan
                               for U in good]))
            grp.create_dataset('t_max_scan', data=np.array(t_max_list))
            # window scan, NaN where that window could not be fitted
            beta_scan = np.full((len(good), len(t_max_list)), np.nan)
            Tc_scan = np.full((len(good), len(t_max_list)), np.nan)
            for iU, U in enumerate(good):
                for r in fits[U]['scan']:
                    it = t_max_list.index(r['t_max'])
                    beta_scan[iU, it] = r['beta']
                    Tc_scan[iU, it] = r['Tc']
            grp.create_dataset('beta_scan', data=beta_scan)
            grp.create_dataset('Tc_scan', data=Tc_scan)
        print(f'\nfits written to {h5_file}:fits_B{B}')


if __name__ == '__main__':
    main()
