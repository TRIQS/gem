from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import block2 as b2
import numpy

#SAMUELE's COMMENTS
# - implement docc

class Pyblock2_N(object):
    """ Wrapper for pyblock2 solvers with N symmetry
    """
    def __init__(self, ntot, nimp, nbath, maxM, spin_pen=0, solver_params=None):
        """Constructor method
        """
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.solver_params = solver_params if solver_params is not None else {}
        self.maxM = self.solver_params.get('maxM', maxM)
        print('maxM=', self.maxM)
        self.type= 'Block2N'
        self.spin_pen = 0
        # initialize pyscf solvers

# MANDATORY FUNCTIONS
    def build_Hemb(self, D, H1E, LAMBDA, V2E):
        tmat = numpy.zeros((self.ntot,self.ntot),dtype=numpy.complex128)
        tmat[:self.nimp,:self.nimp] = H1E
        tmat[:self.nimp,self.nimp:] = D.T
        tmat[self.nimp:,self.nimp:] = -LAMBDA
        tmat[self.nimp:,:self.nimp] = D.conj()
        self.h1 = tmat
        self.Utensor = V2E
        self.Utensor_full = numpy.zeros((self.ntot,self.ntot,self.ntot,self.ntot),dtype=numpy.complex128)
        self.Utensor_full[:self.nimp,:self.nimp,:self.nimp,:self.nimp] = V2E


    def solve_Hemb(self, num_eig=10, verbose=0, sweep_iter = [0,10,20], sweep_epsilon = [5e-3,1e-3,5e-4], maxM=1000,T=0.0):
        
        self.driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SGFCPX, stack_mem=50<<30)#, n_threads=6)

        self.driver.initialize_system(n_sites=self.ntot, n_elec=self.ntot//2)#, spin=0)

        # no order
        #h1e = self.h1
        #Utensor_full = self.Utensor_full
        # lambda_c diagonal gauge
        #h1e, self.u_trans = self.gauge_transform(self.h1, self.ntot, self.nimp)
        #Utensor_full = self.Utensor_full
        #ordering fiedler
        #self.idx = self.driver.orbital_reordering(self.h1, self.Utensor_full)
        #h1e = self.h1[self.idx][:,self.idx]
        #Utensor_full = self.Utensor_full[self.idx][:,self.idx][:,:,self.idx][:,:,:,self.idx]
        #print(idx)
        #ordering gaopt
        h1e, g2e = numpy.copy(self.h1), numpy.copy(self.Utensor_full)

        xmat = numpy.abs(numpy.einsum("ijji->ij", g2e, optimize=True))
        kmat = numpy.abs(h1e) * 1e-7 + xmat
        kmat = b2.VectorDouble(kmat.ravel())
        idx = b2.OrbitalOrdering.fiedler(len(h1e), kmat)

        opts = dict(n_generations=10000, n_configs=len(h1e) * 2, n_elite=8, clone_rate=0.1, mutate_rate=0.1)
        n_tasks = 64
        idxs = []
        for i_task in range(0, n_tasks):
            b2.Random.rand_seed(1234 + i_task)
            idx = b2.OrbitalOrdering.ga_opt(len(h1e), kmat, **opts)
            f = b2.OrbitalOrdering.evaluate(len(h1e), kmat, idx)
            idx = tuple(idx)
            idxs.append(idx)
        idx = sorted(list(set(idxs)))[0]
        self.idx = numpy.array(idx)
        #print(self.idx)
        h1e = h1e[self.idx][:, self.idx]
        Utensor_full = self.Utensor_full[self.idx][:, self.idx][:, :, self.idx][:, :, :, self.idx]

        b = self.driver.expr_builder()

        for i in range(self.ntot):
            for j in range(self.ntot):
                if abs(h1e[i,j])>1e-6:
                    b.add_term("CD", [i,j], h1e[i,j])

        for i in range(self.ntot):
            for j in range(self.ntot):
                for k in range(self.ntot):
                    for l in range(self.ntot):
                        if abs(Utensor_full[i,k,j,l])>1e-6:
                            b.add_term("CCDD", [i,j,l,k], 0.5*Utensor_full[i,k,j,l])

        mpo = self.driver.get_mpo(b.finalize(), iprint=0)

        self.ket = self.driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)

        bond_dims = self.solver_params.get('bond_dims', [200, 300, 400, 500] + [500, 500, 800, 800, 800])
        noises    = self.solver_params.get('noises',    [1e-5]*4 + [1e-6]*4 + [0])
        thrds     = self.solver_params.get('thrds',     [1e-10]*9)
        n_sweeps  = self.solver_params.get('n_sweeps',  20)
        dmrg_tol  = self.solver_params.get('dmrg_tol', 1e-8)

        self.e0 = self.driver.dmrg(mpo, self.ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds, tol=dmrg_tol, cutoff=0, iprint=1)

    def calc_density_matrix(self):
        dm = self.driver.get_1pdm(self.ket)
        # lambdac gauge to original gauge
        #dm = self.u_trans.conj().dot(dm).dot(self.u_trans.T)
        # fildler or gaopt back to original ordering
        idx_rev = []
        for i in range(len(self.idx)):
            for j in range(len(self.idx)):
                if self.idx[j]==i:
                    idx_rev.append(j)
        dm = dm[idx_rev][:,idx_rev]
        self.dm = dm
        return dm

    def compute_E2loc(self):
        eone = numpy.einsum('ij,ij',self.h1,self.dm)
        etwo = self.e0 - eone
        return etwo

#AUXILIARY FUNCTIONS
    @staticmethod
    def gauge_transform(h1e, ntot, nimp):
        from scipy.linalg import eig, eigh
        h1e_trans = numpy.zeros((ntot,ntot), dtype=numpy.complex128)
        u_trans = numpy.eye(ntot, dtype=numpy.complex128)
        h1e_trans[:nimp,:nimp] = h1e[:nimp,:nimp]
        # enforce spin symmtery
        #evals, tmp = eigh(h1e[nimp::2,nimp::2])
        #u_trans[nimp:,nimp:] = numpy.kron(tmp,numpy.eye(2))
        # no spin symmtery
        evals, tmp = eigh(h1e[nimp:,nimp:])
        u_trans[nimp:,nimp:] = tmp
        h1e_trans = u_trans.conj().T.dot(h1e).dot(u_trans)
        #print('h1e_trans=')
        #print(h1e_trans)
        return h1e_trans, u_trans

    def calc_double_occ(self,idx):
        ''' not implemented (return arbitrary value 0.25) '''
        return 0.25



class Pyblock2_N_SZ(Pyblock2_N):
    def __init__(self, ntot, nimp, nbath, maxM, spin_pen=0, solver_params=None):
        """Constructor method
        """
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.solver_params = solver_params if solver_params is not None else {}
        self.maxM = self.solver_params.get('maxM', maxM)
        print('maxM=', self.maxM)
        self.type= 'Block2NSZ'
        # initialize pyscf solvers

#MANDATORY FUNCTIONS
    def solve_Hemb(self, num_eig=10, verbose=0, sweep_iter = [0,10,20], sweep_epsilon = [5e-3,1e-3,5e-4], maxM=1000):
        self.driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ | SymmetryTypes.CPX, stack_mem=50<<30)#, n_threads=6)

        self.driver.initialize_system(n_sites=self.ntot//2, n_elec=self.ntot//2)#, spin=0)

        # no order
        #h1e = self.h1
        #Utensor_full = self.Utensor_full
        # lambda_c diagonal gauge
        h1e, self.u_trans = self.gauge_transform(self.h1, self.ntot, self.nimp)
        Utensor_full = self.Utensor_full

        b = self.driver.expr_builder()

        for i in range(self.ntot):
            for j in range(self.ntot):
                if abs(h1e[i,j])>1e-8:
                    if i%2 == 0:
                        b.add_term("cd", [i//2,j//2], h1e[i,j])
                    elif i%2 == 1:
                        b.add_term("CD", [i//2,j//2], h1e[i,j])

        for i in range(self.ntot):
            for j in range(self.ntot):
                for k in range(self.ntot):
                    for l in range(self.ntot):
                        if abs(Utensor_full[i,k,j,l])>1e-8:
                            if i%2 ==1 and j%2 ==0 and l%2==1 and k%2==0:
                               b.add_term("CcDd", [i//2,j//2,l//2,k//2], 0.5*Utensor_full[i,k,j,l])
                            elif i%2 ==1 and j%2 ==0 and l%2==0 and k%2==1:
                               b.add_term("CcdD", [i//2,j//2,l//2,k//2], 0.5*Utensor_full[i,k,j,l])
                            elif i%2 ==0 and j%2 ==1 and l%2==1 and k%2==0:
                               b.add_term("cCDd", [i//2,j//2,l//2,k//2], 0.5*Utensor_full[i,k,j,l])
                            elif i%2 ==0 and j%2 ==1 and l%2==0 and k%2==1:
                               b.add_term("cCdD", [i//2,j//2,l//2,k//2], 0.5*Utensor_full[i,k,j,l])
                            elif i%2 ==0 and j%2 ==0 and l%2==0 and k%2==0:
                               b.add_term("ccdd", [i//2,j//2,l//2,k//2], 0.5*Utensor_full[i,k,j,l])
                            elif i%2 ==1 and j%2 ==1 and l%2==1 and k%2==1:
                               b.add_term("CCDD", [i//2,j//2,l//2,k//2], 0.5*Utensor_full[i,k,j,l])

        mpo = self.driver.get_mpo(b.finalize(), iprint=0)

        self.ket = self.driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)

        bond_dims = self.solver_params.get('bond_dims', [200, 300, 400, 500] + [500, 500, self.maxM, self.maxM, self.maxM])
        noises    = self.solver_params.get('noises',    [1e-5]*4 + [1e-6]*4 + [0])
        thrds     = self.solver_params.get('thrds',     [1e-10]*9)
        n_sweeps  = self.solver_params.get('n_sweeps',  25)
        dmrg_tol  = self.solver_params.get('dmrg_tol', 1e-10)

        self.e0 = self.driver.dmrg(mpo, self.ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds, tol=dmrg_tol, cutoff=0, iprint=1)

    def calc_density_matrix(self):
        dm = self.driver.get_1pdm(self.ket)
        # lambdac gauge to original gauge
        dm = self.u_trans.conj().dot( numpy.kron( (dm[0] + dm[1])*0.5, numpy.eye(2)) ).dot(self.u_trans.T)
        self.dm = dm
        return dm

#AUXILIARY FUNCTIONS
    @staticmethod
    def gauge_transform(h1e, ntot, nimp):
        from scipy.linalg import eig, eigh
        h1e_trans = numpy.zeros((ntot,ntot), dtype=numpy.complex128)
        u_trans = numpy.eye(ntot, dtype=numpy.complex128)
        h1e_trans[:nimp,:nimp] = h1e[:nimp,:nimp]
        # enforce spin symmtery
        evals, tmp = eigh(h1e[nimp::2,nimp::2])
        u_trans[nimp:,nimp:] = numpy.kron(tmp,numpy.eye(2))
        # no spin symmtery
        #evals, tmp = eigh(h1e[nimp:,nimp:])
        #u_trans[nimp:,nimp:] = tmp
        h1e_trans = u_trans.conj().T.dot(h1e).dot(u_trans)
        #print('h1e_trans=')
        #print(h1e_trans)
        return h1e_trans, u_trans



