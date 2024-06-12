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
        ntot = 28
        nimp = 2
        nbath= 26

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=5000).e_list
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)
        np.random.seed(1234)
        try:
            print('--- reading R and Lambda from checkpoint ---')
            fh5 = h5py.File('checkpoint.h5','r')
            R0 = fh5['R'][...]
            Lambda0 = fh5['Lambda'][...]
            fh5.close()
            assert(R0.shape == (nbath,nimp))
        except:
            print('--- no checkpoint exist or R.shape mismatch with the number of orbitals: initialize R and Lambda ---')
            R0 = np.random.rand(nbath//2,nimp//2)
            R0 = np.kron(R0,np.eye(2))
            Lambda0 = np.zeros((nbath//2,nbath//2))
            Lambda0[0,0] = 3.0
            Lambda0[1,1] = 2.5
            Lambda0[2,2] = 2.0
            Lambda0[3,3] = 1.5
            Lambda0[4,4] = 1.0
            Lambda0[5,5] = 0.5
            Lambda0[6,6] = 0.0
            Lambda0[7,7] =-0.5
            Lambda0[8,8] =-1.0
            Lambda0[9,9] =-1.5
            Lambda0[10,10] =-2.0
            Lambda0[11,11] =-2.5
            Lambda0[12,12] =-3.0
            Lambda0 = np.kron(Lambda0,np.eye(2))

        U = 2.4
        eloc = np.zeros((nimp,nimp))
        Utensor = np.zeros((nimp,nimp,nimp,nimp))
        eloc[0,0] =-U/2.
        eloc[1,1] =-U/2.
        Utensor[0,0,1,1] = U
        Utensor[1,1,0,0] = U

        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'ftps', 'maxM': 100})
        grisb.run(itmax=1000, mix=0.5, tol=5e-5, beta=10000, silence=True, spin_pen=0.05, diis=False)

        docc0 = grisb.docc[0]
        Z = grisb.R.conj().T.dot(grisb.R)
        docc = grisb.docc
        R0 = grisb.R
        Lambda0 = grisb.Lambda

        #self.assertAlmostEqual(Z[0,0].real , 0.999035886155107, 4, 'incorrect spectral weight')

if __name__ == '__main__':
    unittest.main()
