from pyscf import fci, gto, scf, ao2mo, ci, cc, lib, dmrgscf
from pyscf.scf import diis
import numpy
import os

#SAMUELE'S COMMENT
# - implement docc

class Pyscf_ccsd(object):
    """ Wrapper for pyscf ccsd solvers
    """
    def __init__(self, ntot, nimp, nbath, solver_params=None):
        """Constructor method
        """
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.solver_params = solver_params if solver_params is not None else {}
        self.hsize = 2**ntot
        self.type= 'PySCFCCSD'
        # initialize pyscf solvers

#MANDATORY FUNCTIONS
    def build_Hemb(self, D, H1E, LAMBDA, V2E, spin_pen=0.0,beta=500.0):
        tmat = numpy.zeros((self.ntot,self.ntot))
        tmat[:self.nimp,:self.nimp] = H1E
        tmat[:self.nimp,self.nimp:] = D.T
        tmat[self.nimp:,self.nimp:] = -LAMBDA
        tmat[self.nimp:,:self.nimp] = D.conj()
        self.h1 = tmat[::2,::2]
        self.h2 = numpy.zeros((self.ntot//2,self.ntot//2,self.ntot//2,self.ntot//2))
        self.h2[:self.nimp//2,:self.nimp//2,:self.nimp//2,:self.nimp//2] = V2E[::2,::2,1::2,1::2] # spin symmetric

    def solve_Hemb(self, num_eig=10, verbose=0, restrict=None, T=0.0):
        self.restrict = self.solver_params.get('restrict', True) if restrict is None else restrict
        if restrict:
            mol = gto.M()
            mol.nelectron = self.ntot//2
            mol.incore_anyway = True
            mf = scf.RHF(mol)
            mf.max_cycle = 1000
            mf.conv_tol = 1e-8#1e-12
            mf.diis_space = 20
            mf.diis_start_cycle = 5
            mf.chkfile = 'hf.chk'
            if os.path.isfile('hf.chk'):
                mol = lib.chkfile.load_mol('hf.chk')
                #mf = scf.HF(mol)
                mf.__dict__.update(lib.chkfile.load('hf.chk', 'scf'))
            mf.get_hcore = lambda *args: self.h1
            mf.get_ovlp = lambda *args: numpy.eye(self.ntot//2)
            mf._eri = ao2mo.restore(8, self.h2, self.ntot//2) # 8-fold symmetry
            mf.init_guess = '1e'
            mf.diis = diis.EDIIS
            mf = mf.run()
            self.C = mf.mo_coeff
            self.mycc = cc.RCCSD(mf)
            self.mycc.conv_tol = 1e-8
            self.mycc.conv_tol_normt = 1e-5
            self.mycc.max_cycle = 10000
            self.mycc.diis_space = 20#15
            self.mycc.diis_start_cycle = 5
            self.mycc.diis = diis.EDIIS
            self.mycc.iterative_damping = 0.05
            #self.mycc.diis = False
            self.mycc.diis_file = 'ccdiis.h5'
            if os.path.isfile('ccdiis.h5'):
                self.mycc.restore_from_diis_('ccdiis.h5')
            #self.eccsd, t1, t2 = self.mycc.kernel()
            self.eccsd, t1, t2 = self.mycc.ccsd()
            #self.mycc.nroots = 3
            #e, v = self.mycc.ipccsd()
            #self.mycc.solve_lambda()
            self.e0 = self.eccsd + mf.energy_tot()
        else:
            mol = gto.M()
            mol.nelectron = self.ntot//2
            mol.incore_anyway = True
            mf = scf.UHF(mol)
            mf.max_cycle = 1000
            mf.conv_tol = 1e-8
            mf.diis_space = 20
            mf.init_guess_breaksym = True
            mf.get_hcore = lambda *args: self.h1
            mf.get_ovlp = lambda *args: numpy.eye(self.ntot//2)
            mf._eri = ao2mo.restore(8, self.h2, self.ntot//2) # 8-fold symmetry
            mf.init_guess = 'minao'
            mf = mf.run()
            self.Cup, self.Cdn = mf.mo_coeff
            self.mycc = cc.UCCSD(mf)
            self.mycc.conv_tol = 1e-8
            self.mycc.conv_tol_normt = 1e-5
            self.mycc.max_cycle = 500
            self.mycc.diis_space = 20
            self.mycc.diis_start_cycle = 4
            #self.mycc.diis = False
            self.mycc.iterative_damping = 0.001
            self.mycc.diis = diis.ADIIS
            #self.eccsd, t1, t2 = self.mycc.kernel()
            self.eccsd, t1, t2 = self.mycc.ccsd()
            #self.mycc.nroots = 3
            #e, v = self.mycc.ipccsd()
            #self.mycc.solve_lambda()
            self.e0 = self.eccsd + mf.energy_tot()

    def calc_density_matrix(self):
        if self.restrict:
            dmup = self.mycc.make_rdm1()/2.
            dmup = numpy.einsum('ai,bj,ij->ab',self.C, self.C, dmup)
            dm = numpy.kron(dmup,numpy.eye(2))
            self.dm = dm
            return dm
        else:
            dmup, dmdn = self.mycc.make_rdm1()
            dmup = numpy.einsum('ai,bj,ij->ab',self.Cup, self.Cup, dmup)
            dmdn = numpy.einsum('ai,bj,ij->ab',self.Cdn, self.Cdn, dmdn)
            dm = (dmup + dmdn)/2. # average over spin to recover spin symmetry. Don't use it with magnetism
            dm = numpy.kron(dm,numpy.eye(2))
            self.dm = dm
            return dm

    def compute_E1loc(self, nimp):
        '''
        Compute local energy including local one and two-body term from a given wavefunction.
        Input:
        Return:
          Eloc: float. Total local energy.
        '''
        return numpy.trace(self.h1[:nimp,:nimp].dot(self.dm[:nimp,:nimp].T))

    def compute_E2loc(self):
        eone = 2*numpy.einsum('ij,ij',self.h1,self.dm[::2,::2])
        etwo = self.e0 - eone
        return etwo

#AUXILIARY FUNCTIONS
    def calc_double_occ(self,idx):
        print('double occupancy not implemented')
        return 0.25


class Pyscf_dmrg(Pyscf_ccsd):
    """ Wrapper to pyscf dmrg
    """
    def __init__(self, ntot, nimp, nbath, maxM, solver_params=None):
        """Constructor method
        """
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.solver_params = solver_params if solver_params is not None else {}
        self.hsize = 2**ntot
        self.maxM = self.solver_params.get('maxM', maxM)
        print('maxM=', self.maxM)
        # initialize pyscf solvers

#MANDATORY FUNCTIONS
    def solve_Hemb(self, num_eig=10, verbose=0, sweep_iter = [0,10,20], sweep_epsilon = [5e-3,1e-3,5e-4], maxM=500):
        mol = gto.M()
        mol.nelectron = self.ntot//2
        mol.incore_anyway = True
        mf = scf.RHF(mol)
        mf.max_cycle = 1000
        mf.conv_tol = 1e-8#1e-12
        #mf.diis_space = 5#10
        #mf.diis_start_cycle = 5
        mf.diis = False
        mf.DIIS = diis.ADIIS
        #self.h1, self.u_trans = self.gauge_transform(self.h1, self.ntot//2, self.nimp//2)
        mf.get_hcore = lambda *args: self.h1
        mf.get_ovlp = lambda *args: numpy.eye(self.ntot//2)
        mf._eri = ao2mo.restore(8, self.h2, self.ntot//2) # 8-fold symmetry
        #mf._eri = self.h2
        #mf.init_guess = '1e'
        mf = mf.run()
        self.C = mf.mo_coeff

        self.nmo = mf.mo_coeff.shape[1]
        self.nelec = mol.nelec # this part report error when nelectron is odd
        self.cisolver = dmrgscf.DMRGSCF(mf, self.nmo, self.nelec, maxM=self.maxM, tol=1e-8)
        self.cisolver.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 4))
        print('using threads:',self.cisolver.fcisolver.threads)
        self.cisolver.fcisolver.memory = 50 #int(mol.max_memory / 1000) # mem in GB
        self.e0 = self.cisolver.kernel()[0]

    def calc_density_matrix(self):
        dm1 = self.cisolver.fcisolver.make_rdm1(0, self.nmo, self.nelec)/2.
        dmup = numpy.einsum('ai,bj,ij->ab', self.C, self.C, dm1)
        dm = numpy.kron(dmup,numpy.eye(2))
        self.dm = dm
        return dm

    def compute_E2loc(self):
        eone = 2*numpy.einsum('ij,ij',self.h1,self.dm[::2,::2])
        etwo = self.e0 - eone
        return etwo

#AUXILIARY FUNCTIONS
    @staticmethod
    def gauge_transform(h1e, ntot, nimp):
        from scipy.linalg import eig, eigh
        h1e_trans = numpy.zeros((ntot,ntot), dtype=numpy.float64)
        u_trans = numpy.eye(ntot, dtype=numpy.float64)
        h1e_trans[:nimp,:nimp] = h1e[:nimp,:nimp]
        # enforce spin symmtery
        evals, tmp = eigh(h1e[nimp:,nimp:])
        u_trans[nimp:,nimp:] = tmp
        h1e_trans = u_trans.conj().T.dot(h1e).dot(u_trans)
        print('h1e_trans=')
        print(h1e_trans)
        return h1e_trans, u_trans

    def calc_double_occ(self,idx):
        return 0.25
