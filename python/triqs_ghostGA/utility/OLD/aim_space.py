import numpy as np
from triqs_ghostGA.utility.utilities import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, \
     Hermitian_list,funcMat, calc_nf, dF


def calc_Lambda(R, Lambda_c, Delta_p, D, H_list):
    """ Compute Lambda matrix from R, D, Lambda_c and Delta. It comes from the same equation as for Lambda_c. Lambda is expressed as

    .. math::
        \Lambda = \sum_s l_s h_s

    where :math:`h_s` are a set of matrices that forms a basis for the space of Hermitian matrices
    with dimension :math:`\nu`. They are given in H_list. The elements :math:`l_s` are obtained by calculating

    .. math::
        l_s = -l^c_s - \sum_{cb\alpha} \frac{\partial}{\partial d^P_s} \left[ \Delta ( 1 - \Delta ) ]^{1/2}_{cb} ( D R^T)_{bc}

    where :math:`l^c_s` are the elements of the decomposition of :math:`\Lambda_c = \sum_s l^c_s h_s`
    and :math:`d^P_s` of :math:`\Delta_p = \sum_s d_s h_s^T`.

    Here, extracting the :math:`l^c_s`s from :math:`\Lambda_c` is done using the inverse_realHcombination function.
    The derivative with respect to :math:`d^P_s` is done using the dF function.
    :math:`\Lambda` is reconstructed using the realHcombinaison function.

    """
    lc = inverse_realHcombination(Lambda_c, H_list)
    l = np.copy(lc) * 0.0
    DR = np.dot(D, np.transpose(R))
    for k in range(len(H_list)):
        AA = Delta_p
        HH = H_list[k].T
        derivative = dF(AA,HH, denRm1, ddenRm1)
        tt = np.trace(np.dot(DR, derivative))
        l[k] = -lc[k] - (tt + np.conjugate(tt)).real
    Lambda = realHcombination(l, H_list)
    return Lambda

