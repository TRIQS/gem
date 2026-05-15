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
#from triqs_ghostGA.pyscf_ccsd import *
from triqs_ghostGA.pyscf_solvers import *

class TestGrisb(unittest.TestCase):
    def runTest(self):
        np.set_printoptions(suppress=True,precision=10)
        ntot = 40
        nimp = 10
        nbath= 30

        # construct ek with semicircular DOS
        e_list = EList_SemiCircular(nmesh=100).e_list
        eks = []
        for e in e_list:
            tmp = numpy.array([[ 1.0*e, 0.0  , 0.0  , 0.0  , 0.0  ],
                               [ 0.0  , 1.0*e, 0.0  , 0.0  , 0.0  ],
                               [ 0.0  , 0.0  , 0.5*e, 0.0  , 0.0  ],
                               [ 0.0  , 0.0  , 0.0  , 0.5*e, 0.0  ],
                               [ 0.0  , 0.0  , 0.0  , 0.0  , 0.5*e]],dtype=numpy.complex128)
            tmp = numpy.kron(tmp,numpy.eye(2))
            eks.append(tmp)
        eks = numpy.array(eks)
        Us = numpy.arange(0.1,3.55,0.1)#[::-1]
        #R0 = numpy.loadtxt("data/R_U3.5.dat",dtype=numpy.complex128)[:,2:4] #+ numpy.random.rand(nbath,nimp)*0.01
        #Lambda0 = numpy.loadtxt("data/LAMBDA_U3.5.dat",dtype=numpy.complex128)
        #R0 = numpy.ones((nbath//2,nimp//2))*0.5
        numpy.random.seed(1234)
        R0 = numpy.random.rand(nbath//2,nimp//2)
        R0 = numpy.kron(R0,numpy.eye(2))
        Lambda0 = numpy.zeros((nbath//2,nbath//2))
        Lambda0[0,0] = 2.0
        Lambda0[1,1] = 2.0
        Lambda0[2,2] = 2.0
        Lambda0[3,3] = 2.0
        Lambda0[4,4] = 2.0
        Lambda0[5,5] = 0.0
        Lambda0[6,6] = 0.0
        Lambda0[7,7] = 0.0
        Lambda0[8,8] = 0.0
        Lambda0[9,9] = 0.0
        Lambda0[10,10] =-2.0
        Lambda0[11,11] =-2.0
        Lambda0[12,12] =-2.0
        Lambda0[13,13] =-2.0
        Lambda0[14,14] =-2.0
        #Lambda0 = numpy.random.rand(nbath//2,nbath//2)
        #Lambda0 = (Lambda0 + Lambda0.T)/2
        Lambda0 = numpy.kron(Lambda0,numpy.eye(2))
        #fh5i = h5py.File('sols_backup.h5','r')
        #R0 = fh5i['U1.50/R'][...]
        #Lambda0 = fh5i['U1.50/Lambda'][...]
        #fh5i.close()

        U = 0.4
        J = U/4.
        eloc = np.zeros((nimp,nimp))
        nnom = 4.0 #nominal occupancy
        mu0 = (U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        Utensor = U_matrix_kanamori(nimp//2, U, J)
        #print(Utensor.shape)
        edsolver=Pyscf_ccsd(ntot, nimp, nbath)
        # remove the history files
        import os
        try:
            os.remove('hf.chk')
            os.remove('ccdiis.h5')
        except:
            print('no history files')
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, edsolver=edsolver)
        grisb.run(mu=mu0, itmax=100, mix=0.5, tol=2e-3, beta=500, silence=True, spin_pen=0.00)

        Z = grisb.R.conj().T.dot(grisb.R)
        docc = grisb.docc
        R0 = grisb.R
        Lambda0 = grisb.Lambda
        dm = grisb.denMat

        # compute Gf and Sig
        Nom = 20
        oms = numpy.linspace(-0.05,0.05,Nom)
        eta = 0.0005
        grisb.compute_Gf_Sig(mu0, grisb.eks, oms, eta)
        Z1 = 1/(1-(grisb.Sig[Nom//2+1,0,0]-grisb.Sig[Nom//2,0,0]).real/(oms[Nom//2+1]-oms[Nom//2]))
        Z2 = 1/(1-(grisb.Sig[Nom//2+1,2,2]-grisb.Sig[Nom//2,2,2]).real/(oms[Nom//2+1]-oms[Nom//2]))
        Z3 = 1/(1-(grisb.Sig[Nom//2+1,4,4]-grisb.Sig[Nom//2,4,4]).real/(oms[Nom//2+1]-oms[Nom//2]))
        Z4 = 1/(1-(grisb.Sig[Nom//2+1,6,6]-grisb.Sig[Nom//2,6,6]).real/(oms[Nom//2+1]-oms[Nom//2]))
        Z5 = 1/(1-(grisb.Sig[Nom//2+1,8,8]-grisb.Sig[Nom//2,8,8]).real/(oms[Nom//2+1]-oms[Nom//2]))

        print('occupancy:', 2*dm[0,0], 2*dm[2,2], 2*dm[4,4], 2*dm[6,6], 2*dm[8,8])
        print('total occupancy:', 2*(dm[0,0]+dm[2,2]+dm[4,4]+dm[6,6]+dm[8,8]) )
        print('QP weight:', Z1, Z2, Z3, Z4, Z5)

if __name__ == '__main__':
    unittest.main()
