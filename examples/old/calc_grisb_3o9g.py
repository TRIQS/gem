#######################################################
# Example for the degenerate two-orbital Hubbard model
# Author: Tsung-Han Lee
# Email: henhans74716@gmail.com
#######################################################
import unittest
import numpy as np
import h5py
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utils_TH import get_semicircle_e_list,U_matrix_kanamori

class TestGrisb(unittest.TestCase):
    def runTest(self):
        np.set_printoptions(suppress=True,precision=10)
        ntot = 24
        nimp = 6
        nbath= 18
        mu=5

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=5000).e_list
        eks = []
        for e in e_list:
            tmp = e*np.eye(nimp//2,dtype=np.complex_)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)
        np.random.seed(1234)
        R0 = np.random.rand(nbath//2,nimp//2)
        R0 = np.kron(R0,np.eye(2))
        Lambda0 = np.zeros((nbath//2,nbath//2))
        Lambda0[0, 0], Lambda0[1, 1], Lambda0[2, 2] = -3.0, -3.0, -3.0
        Lambda0[3, 3], Lambda0[4, 4], Lambda0[5, 5] = 0.0, 0.0, 0.0
        Lambda0[6, 6], Lambda0[7, 7], Lambda0[8, 8] = 3.0, 3.0, 3.0
        Lambda0 = np.kron(Lambda0,np.eye(2))

        U = 12.0
        J = U/4.
        #eloc = np.zeros((nimp,nimp))
        eloc = np.diag(np.ones(nimp)*(-mu))
        nnom = 2.0
        #eloc[0,0] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        #eloc[1,1] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        #eloc[2,2] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        #eloc[3,3] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        #eloc[0,2] = 0.0
        #eloc[2,0] = 0.0
        #eloc[1,3] = 0.0
        #eloc[3,1] = 0.0
        Utensor = U_matrix_kanamori(2, U, J)
        #print(Utensor.shape)
        from triqs_ghostGA.solvers.simple_ed import SimpleED
        edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                            N_sector=ntot//2, Sz_sector=0, dtype=np.complex128)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, edsolver=edsolver)
        grisb.run(itmax=100, mix=0.5, tol=1e-6, beta=500, silence=False, spin_pen=0.05)

        docc0 = grisb.docc[0]
        docc1 = grisb.docc[1]
        Z = grisb.R.conj().T.dot(grisb.R)
        docc = grisb.docc
        R0 = grisb.R
        Lambda0 = grisb.Lambda

        #self.assertAlmostEqual(docc0.real , 0.1442486503727918, 4, 'incorrect double occupancy')

if __name__ == '__main__':
    unittest.main()
