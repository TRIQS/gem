import numpy as np
from triqs_ghostGA.utility.utilities import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, Hermitian_list, funcMat, calc_nf, dF


def calc_rhoks(R, Lambda, eks, T, mu=0):
    r"""Calculate rho_ks, an object that contains elements necessary to compute Delta_p and D.
    It requires R, Lambda, the dispersion espilon_k and the temperature. It is given as

    .. math::
        \rho_k = f(R \ \epsilon_k \ R^\dagger + \lambda - \mu, T)^T

    where :math:`f` is the Fermi-Dirac function, :math:`R, Lambda` are objects used in the ghostGA formalism,
    :math:`\epsilon_k` is the non-interacting dispersion, :math:`T` is the temperature (necessary for the Fermi-Dirac function)
    and :math:`\mu` is the chemical potential.

    Now there is an additional transpose here, that is necessary in the equation later but can be performed here already.

    The resulting object is an array of the length of the number of k-points, where each element is a matrix with the shape of Lambda.
    """

    return [calc_nf( np.dot(R, np.dot(x, R.conj().T ) ) + Lambda - mu*np.eye(Lambda.shape[0]), T).T for x in eks]

#########################################

def calc_D(R, Lambda, Delta_p, eks, rhoks):
    r"""Compute the D matrix from R, Lambda and Delta, which is given by

    .. math::
        D_{d \alpha} = \frac{1}{N_k} \sum_k ([ \Delta ( 1 - \Delta ) ]^{-1/2} ] [ \rho_k^T R^* \epsilon_k^T ]_{d \alpha}.

    TODO: Somehow the equation is different. Actually D is transposed in the calculation. Is this propagated in the other equations?
    """
    Left=[np.dot( np.dot(eks[x], R.conj().T ), rhoks[x].T ) for x in range(len(rhoks))]
    Left=sum(Left)/float(len(rhoks))
    Right=funcMat(Delta_p, denR)
    return np.dot(Right,np.transpose(Left))

#########################################

def calc_Lambda_c(R, Lambda, Delta_p, D, H_list):
    r"""Compute Lambda_c matrix from R, D, Lambda and Delta. It is given by

    .. math::
        \Lambda_c = \sum_s l^c_s h_s

    where :math:`h_s` are a set of matrices that forms a basis for the space of Hermitian matrices
    with dimension :math:`\nu`. The elements :math:`l^c_s` are obtained by calculating

    .. math::
        l^c_s = -l_s - \sum_{\alpha} \text{Tr} \left[ \frac{\partial}{\partial d^P_s} \left[ \Delta ( 1 - \Delta ) ]^{1/2} ( D R^T ) \right]

    where :math:`l_s` are the elements of the decomposition of :math:`\Lambda = \sum_s l_s h_s`
    and :math:`d^P_s` of :math:`\Delta_p = \sum_s d_s h_s^T`.

    Here, extracting the :math:`l_s`s from :math:`\Lambda` is done using the inverse_realHcombination function.
    The derivative with respect to :math:`d^P_s` is done using the dF function.
    :math:`\Lambda_c` is reconstructed using the realHcombinaison function.

    """
    l = inverse_realHcombination(Lambda, H_list)
    lc = np.copy(l) * 0.0
    DR = np.dot(D, np.transpose(R))
    for k in range(len(H_list)):
        AA = Delta_p
        HH = H_list[k].T
        derivative = dF(AA, HH, denRm1, ddenRm1)
        tt = np.trace(np.dot(DR ,derivative))
        lc[k] = -l[k] - (tt + np.conjugate(tt)).real
    Lambda_c = realHcombination(lc, H_list)
    return Lambda_c

def calc_Lambda_c_new(R, Lambda, Delta_p, D, H_list):
    r"""Compute Lambda_c matrix from R, D, Lambda and Delta. It is given by

    .. math::
        \Lambda_c = \sum_s l^c_s h_s

    where :math:`h_s` are a set of matrices that forms a basis for the space of Hermitian matrices
    with dimension :math:`\nu`. The elements :math:`l^c_s` are obtained by calculating

    .. math::
        l^c_s = -l_s - \sum_{\alpha} \text{Tr} \left[ \frac{\partial}{\partial d^P_s} \left[ \Delta ( 1 - \Delta ) ]^{1/2} ( D R^T ) \right]

    where :math:`l_s` are the elements of the decomposition of :math:`\Lambda = \sum_s l_s h_s`
    and :math:`d^P_s` of :math:`\Delta_p = \sum_s d_s h_s^T`.

    Here, extracting the :math:`l_s`s from :math:`\Lambda` is done using the inverse_realHcombination function.
    The derivative with respect to :math:`d^P_s` is done using the dF function.
    :math:`\Lambda_c` is reconstructed using the realHcombinaison function.

    """
    d, dvecs = np.linalg.eigh(Delta_p.T)
    left_d= dvecs @ np.diag(np.sqrt(d/(1-d))) @ dvecs.T.conj()
    right_d= dvecs @ np.diag(np.sqrt((1-d)/d)) @ dvecs.T.conj()
    Ltmp = 0.5* left_d @ Lambda @ right_d
    Ltmp += Ltmp.T.conj()
    l = inverse_realHcombination(Ltmp, H_list)
    lc = np.copy(l) * 0.0
    DR = np.dot(D, np.transpose(R))
    for k in range(len(H_list)):
        AA = Delta_p
        HH = H_list[k].T
        derivative = dF(AA, HH, denRm1, ddenRm1)
        tt = np.trace(np.dot(DR ,derivative))
        lc[k] = -l[k] - (tt + np.conjugate(tt)).real
    Lambda_c = realHcombination(lc, H_list)
    return Lambda_c

