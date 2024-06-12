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
        nimp = 4
        nbath= 12

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=5000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e, 0.0  ],
                               [0.0  , 1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)
        np.random.seed(1234)
        R0 = np.random.rand(nbath//2,nimp//2)
        R0 = np.kron(R0,np.eye(2))
        Lambda0 = np.zeros((nbath//2,nbath//2))
        Lambda0[0,0] = 0.1
        Lambda0[1,1] = 0.1
        Lambda0[2,2] = 0.0
        Lambda0[3,3] = 0.0
        Lambda0[4,4] =-0.1
        Lambda0[5,5] =-0.1
        Lambda0 = np.kron(Lambda0,np.eye(2))

        U = 1.2
        J = U/4.
        eloc = np.zeros((nimp,nimp))
        nnom = 2.0
        eloc[0,0] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[1,1] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[2,2] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[3,3] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[0,2] = 0.0
        eloc[2,0] = 0.0
        eloc[1,3] = 0.0
        eloc[3,1] = 0.0
        Utensor = U_matrix_kanamori(2, U, J)
        #print(Utensor.shape)
        from triqs_ghostGA.ci import CI
        edsolver=CI(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, edsolver=edsolver)
        grisb.run(itmax=1, mix=0.5, tol=1e-6, beta=500, silence=True, spin_pen=0.00)

        docc0 = grisb.docc[0]
        docc1 = grisb.docc[1]
        Z = grisb.R.conj().T.dot(grisb.R)
        docc = grisb.docc
        R0 = grisb.R
        Lambda0 = grisb.Lambda

        self.assertAlmostEqual(docc0.real , 0.1442486503727918, 4, 'incorrect double occupancy')

if __name__ == '__main__':
    unittest.main()
