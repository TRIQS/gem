###########################################
#      List of energies for simple models
#      Author: Tsung-Han Lee
###########################################
import numpy as np
from numpy import sqrt, heaviside as hside, pi, arcsin
import matplotlib.pyplot as plt
from scipy.optimize import bisect

############################################

class EList ():
    r"""Object that contains a list of energy points that follow the
        DOS distribution of the non-interacting model chosen:

    .. math::
       \epsilon_k \rightarrow A(\omega) = \frac{1}{N_k} \sum_k \delta(\omega - \epsilon_k)

    where :math:`\epsilon_k` is the electronic dispersion that leads to the DOS
    :math:`A(\omega)`.
    """

    def __init__ (self):
        self.k_list = []
        self.e_list = []

    def __str__ (self): return "EList object. Empty."

############################################

class EList_TB_2D (EList):
    r"""Get the energy list from a TRIQS tight-binding object and
        stores it.

    Input:
        H_r: TB Hamiltonian object from TRIQS.
        n_k: Number of k-points in one direction, homogeneous and interpolated from the Fourier interpolation.
        n_fourier: Number of k-points in the Fourier interpolation.
    Output:
        e_list: list of e points.
    """

    def __init__ (self, H_r, n_k=100, n_fourier=20):
        kmesh = H_r.get_kmesh(n_k=n_fourier)
        e_k = H_r.fourier(kmesh)

        k = np.linspace(-np.pi, np.pi, num=n_k)
        kx, ky = np.meshgrid(k, k)

        e_k_interp = np.vectorize(lambda kx, ky : e_k((kx, ky, 0)).real)(kx, ky)
        self.k_list = [kx, ky]
        self.e_list = np.reshape(e_k_interp, n_k**2)

    def __str__ (self): return "EList object. Created from TB " + str(H_r)

############################################

class EList_1D (EList):
    r"""Get the energy list of a 1D chain with dispersion

    .. math::
       \epsilon_k = 2 t \cos(k)

    with :math:`k \ \in \ \[-\pi, \pi\[`.

    Input:
        nmesh: number of e points
        d: half-bandwidth
    Output:
        e_list: list of e points
    """

    def __init__ (self, nmesh = 500, d = 1):
        self.nmesh = nmesh
        self.d = d
        self.k_list = np.linspace(-pi,pi,nmesh)
        self.e_list = d*np.cos(k_list)

    def __str__ (self): return "EList object. 1D chain: %d pts and %.2f half-width." % (self.nmesh, self.d)

############################################

class EList_SemiCircular (EList):
    r"""Get the energy list with a semi-circular density of state of half-width  d, that is

    .. math::
       A(\omega) = \frac{2}{\pi d^2} \sqrt{d^2 - \omega^2}.

    Input:
        nmesh: number of e points
        d: half-bandwidth
    Output:
        e_list: list of e points
    """

    def __init__ (self, nmesh = 500, d = 1.0):
        self.nmesh = nmesh
        self.d = d

        # While the DOS is given by A(\omega) above, we do the following to sample a distribution
        # the produce A(\omega) as the DOS.

        dos = lambda e: 2./(pi*d**2) * sqrt(d**2-e**2)

        # It can be checked by plotting the histogram with matplotlib.pyplot.hist(e_list)
        # and comparing with A(\omega).
        cdos = lambda e: ( e/d**2*sqrt(d**2-e**2) + arcsin(e/sqrt(d**2)) ) / (pi) + 0.5

        cdos_list = np.linspace(0,1,nmesh+1)
        e_list = [bisect(lambda x: cdos(x)-a, -d ,d) for a in cdos_list]
        e_list = np.asarray(e_list)
        self.e_list = (e_list[1:] + e_list[0:-1])/2

    def __str__ (self): return "EList object. Semi-Circular density of states: $d pts and %.2f half-width." % (self.nmesh, self.d)

############################################

class EList_Flat (EList):
    r"""Get the energy list with a flat density of state of half-width d, that is

    .. math::
       A(\omega) = \frac{1}{2d} \Theta(|d| - \omega)

    where `\Theta(x)` is the Heaviside function.

    Input:
        nmesh: number of e points
        d: half-bandwidth
    Output:
        e_list: list of e points
    """

    def __init__ (self, nmesh = 500, d = 1.0):
        self.nmesh = nmesh
        self.d = d

        # While the DOS is given by A(\omega) above, we do the following to sample a distribution
        # the produce A(\omega) as the DOS.

        dos = lambda e: 1./(2*d)*hside(e+d,0.5)*hside(-e+d,0.5)

        # It can be checked by plotting the histogram with matplotlib.pyplot.hist(e_list)
        # and comparing with A(\omega).
        cdos = lambda e: (1./(2*d)*hside(d+e,0.5)*
                          ((-d+e+2*d*hside(d,0.5))*hside(-d-e,0.5)*hside(d-e,0.5)
                           +hside(d,0.5)*(2*d+(-d+e)*hside(d-e,0.5))*hside(d+e,0.5)))

        cdos_list = np.linspace(0,1,nmesh+1)
        e_list = [bisect(lambda x: cdos(x)-a, -d ,d) for a in cdos_list]
        e_list = np.asarray(e_list)
        self.e_list = (e_list[1:] + e_list[0:-1])/2

############################################
