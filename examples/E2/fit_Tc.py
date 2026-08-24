######################################################################
# Post-processing for Example 2. Reads every U group stored at a given B
# by Square_1orb_2frag_phase.py and fits m(T) = A * (Tc - T)^beta.
# The power law only holds close to Tc: fitting the whole ordered branch
# includes the saturated low-T part and gives beta ~ 0.33 instead of 0.5,
# so the fit is restricted to (Tc - T)/Tc < t_max with Tc self-consistent.
######################################################################

import re
import numpy as np
import h5py
from scipy.optimize import curve_fit

B = 3
h5_file = f'data_Square_1orb_2frag_B{B}_phase.h5'
t_max = 0.2    # width of the critical window in reduced temperature
m_min = 2e-3   # noise floor on the order parameter

# staggered magnetisation of each U group available at this B
scans = {}
with h5py.File(h5_file, 'r') as h5f:
    for key in h5f:
        match = re.match(rf'U(.+)_B{B}$', key)
        if match is None:
            continue
        mag = h5f[key]['mag_grid'][:].real
        scans[float(match.group(1))] = (h5f[key]['T_list'][:],
                                        np.abs(0.5 * (mag[0] - mag[1])))

# fitted in log-log, so that the small-m points near Tc are weighted as well
def model(T, logA, Tc, beta):
    return logA + beta * np.log(np.clip(Tc - T, 1e-14, None))

for U in sorted(scans):
    T, m = scans[U]
    if m.max() < 10 * m_min:
        print(f'U={U:.2f}  no ordered phase')
        continue

    # seed Tc with the zero crossing of m^2 ~ (Tc - T), then iterate the window
    keep = (m > m_min) & (m < 0.6 * m.max())
    slope, intercept = np.polyfit(T[keep], m[keep] ** 2, 1)
    Tc, popt = -intercept / slope, None
    for _ in range(20):
        win = (m > m_min) & (T < Tc) & (Tc - T < t_max * Tc)
        if win.sum() < 4:
            popt = None
            break
        popt, pcov = curve_fit(model, T[win], np.log(m[win]),
                               p0=[np.log(m.max()), Tc, 0.5],
                               bounds=([-50, T[win].max(), 0.05],
                                       [50, 2 * T.max(), 3.0]))
        if abs(popt[1] - Tc) < 1e-10:
            break
        Tc = popt[1]

    if popt is None:
        print(f'U={U:.2f}  fewer than 4 points inside t_max={t_max}')
        continue
    err = np.sqrt(np.diag(pcov))
    print(f'U={U:.2f}  Tc={popt[1]:.5f}+-{err[1]:.5f}  '
          f'beta={popt[2]:.4f}+-{err[2]:.4f}  ({win.sum()} points)')
