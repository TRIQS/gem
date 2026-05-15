#######################################################
# Example for the degenerate two-orbital Hubbard model
# Author: Tsung-Han Lee
# Email: henhans74716@gmail.com
#######################################################
import unittest
import numpy as np
import h5py
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular

class TestGrisb(unittest.TestCase):
    def runTest(self):
        np.set_printoptions(suppress=True,precision=10)
        ntot = 16
        nimp = 2
        nbath= 14

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=5000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)
        R0 = np.random.rand(nbath//2,nimp//2)
        R0 = np.kron(R0,np.eye(2))
        Lambda0 = np.zeros((nbath//2,nbath//2))
        Lambda0[0,0] = 1.0
        Lambda0[1,1] = 0.8
        Lambda0[2,2] = 0.6
        Lambda0[3,3] = 0.0
        Lambda0[4,4] =-0.6
        Lambda0[5,5] =-0.8
        Lambda0[6,6] =-1.0
        Lambda0 = np.kron(Lambda0,np.eye(2))

        U = 2.4
        J = U/4.
        eloc = np.zeros((nimp,nimp))
        Utensor = np.zeros((nimp,nimp,nimp,nimp))
        eloc[0,0] =-U/2.
        eloc[1,1] =-U/2.
        Utensor[0,0,1,1] = U
        Utensor[1,1,0,0] = U

        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'ci', 'use_Sz': True, 'use_Ntot': True})
        grisb.run(itmax=1000, mix=0.5, tol=1e-5, beta=500, silence=True, spin_pen=0.05)

        docc0 = grisb.docc[0]
        Z = grisb.R.conj().T.dot(grisb.R)
        docc = grisb.docc
        R0 = grisb.R
        Lambda0 = grisb.Lambda
        print(Z[0,0])

        self.assertAlmostEqual(Z[0,0].real , 0.9946752769303452, 4, 'incorrect spectral weight')

if __name__ == '__main__':
    unittest.main()
