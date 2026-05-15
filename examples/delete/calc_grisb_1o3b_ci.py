#######################################################
# Example for the degenerate two-orbital Hubbard model
# Author: Tsung-Han Lee
# Email: henhans74716@gmail.com
#######################################################
import unittest
import time
import numpy as np
import h5py
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utility.utils_TH import U_matrix_kanamori
from triqs_ghostGA.utility.e_list import EList_SemiCircular
from triqs_ghostGA.utility.delta_fit import *
from triqs_ghostGA.solvers.simple_ed import SimpleED

class TestGrisb(unittest.TestCase):
    def runTest(self):
        np.set_printoptions(suppress=True,precision=10)
        nimp  = 2
        nbath = 6
        ntot  = nimp+nbath

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
        Lambda0[0,0] = 0.0
        Lambda0[1,1] = 1.1
        Lambda0[2,2] =-1.1
        Lambda0[0,1] = 0.1
        Lambda0[1,0] = 0.1
        Lambda0[0,2] = 0.1
        Lambda0[2,0] = 0.1
        
        Lambda0 = np.kron(Lambda0,np.eye(2))

        U = 1.0
        J = U/4.
        eloc = np.zeros((nimp,nimp))
        Utensor = np.zeros((nimp,nimp,nimp,nimp))
        eloc[0,0] =-U/2.
        eloc[1,1] =-U/2.
        Utensor[0,0,1,1] = U
        Utensor[1,1,0,0] = U
        
        Solver = SimpleED(ntot, use_Ntot=True, spin_pen=10)
        t1_i=time.time()
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, edsolver=Solver, spin_sym=True,write=True)
        grisb.run(itmax=1000, mix=0.5, tol=1e-5, beta=500, silence=True)
        t1_f=time.time()

        docc0 = grisb.docc[0]
        Z = grisb.R.conj().T.dot(grisb.R)
        docc = grisb.docc
        R0 = grisb.R
        Lambda0 = grisb.Lambda
        print(Z[0,0])

        print("R",grisb.R)
        print("L",grisb.Lambda)
        print("D",grisb.D)
        print("Lc",grisb.Lambda_c)

        #raise ValueError("STOP HERE")


        t2_i=time.time()
        grisb_2 = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=grisb.R, Lambda=grisb.Lambda, D=grisb.D, Lambda_c=grisb.Lambda_c, edsolver=Solver, spin_sym=True,write=True)
        grisb_2.run_gdmet(itmax=1000, mix=0.5, tol=1e-5, beta=500, silence=True)
        t2_f=time.time()

        docc0_2 = grisb_2.docc[0]
        Z_2 = grisb_2.R.conj().T.dot(grisb_2.R)
        docc_2 = grisb_2.docc
        R0_2 = grisb_2.R
        Lambda0_2 = grisb_2.Lambda
        print(Z_2[0,0])


        print(f"time standard cycle:{t1_f-t1_i}s")
        print(f"time new cycle:{t2_f-t2_i}s")
        self.AssertAlmostEqual( Z[0,0].real , Z_2[0,0].real, 4, 'incorrect QP weight')

if __name__ == '__main__':
    unittest.main()
