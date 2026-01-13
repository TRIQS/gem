import numpy as np
from triqs_ghostGA.utility.delta_fit import *

m, n = 23,7
R = np.random.rand( m, n) + 1j*np.random.rand( m, n)
L = np.random.rand( m, m) + 1j*np.random.rand( m, m)
L = L + L.T.conj()

x=pack_params(L,R)

Lu, Ru = unpack_params(x, m, n)

print(np.sum(np.abs(L-Lu)))
print(np.sum(np.abs(R-Ru)))
