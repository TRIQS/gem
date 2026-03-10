"""
Exact diagonalization code to solve the embeding Hamiltonian as described in
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.011008.

Note: Implemented spin penalty. No symmetry implementation yet.

Author: Tsung-Han Lee (2017), henhans74716@gmail.com
"""
import h5py
import numpy
from scipy import sparse
from scipy.sparse.linalg import eigsh
#import primme

class Fed(object):
    """This is a class representation of a full operatorial based
    exact-diagonalization object. 

    :param ntot: Total number of orbital in the exact-diagonalization porblem.
    :type ntot: int

    :param nimp: Total number of imp orbital in the exact-diagonalization porblem.
    :type ntot: int

    :param nbath: Total number of bath orbital in the exact-diagonalization porblem.
    :type ntot: int

    """
    def __init__(self, ntot, nimp, nbath):
        """Constructor method
        """
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.hsize = 2**ntot
        self.build_fermion_op()
        self.build_operators()
        self.type = 'Fed'
   
    def build_fermion_op(self):
        '''
        Build fermionic operators for each spin+orbitals, alpha, with size 
        of Hilberspace A and B.
        Input:
            no: number of orbital
        Output:
            FH_list: a list of fermionic operator <A|[f^dagger_alpha]|B>
        '''
        strb = '{0:0'+str(self.ntot)+'b}' # string to save binary configuration
        self.FH_list = []
        print('Hibert space size:', self.hsize)
        for o in range(self.ntot): #calculate FH for each orbital
            row = []
            col = []
            data = []
            for b in range(self.hsize): # col |B>
                config_b = strb.format(b) # cofiguration correspond to B
                if config_b[o] == '0':
                    col.append(b)
                    config_a = ''
                    #calculate the corresponding <A|
                    for i in range(self.ntot):
                        if i == o:
                            config_a += '1'
                        else:
                            config_a += config_b[i]
                    a = int(config_a,2)
                    row.append(a)
                    #calculate the exponent for minus sign
                    expo = 0
                    for i in range(o):
                        expo += int(config_b[i])
                    data.append((-1.)**(expo))
            #assign values into sparse matrix
            FH = sparse.csr_matrix((data, (row, col)), shape=(self.hsize, self.hsize), dtype=numpy.float64)
            self.FH_list.append(FH)

    def build_operators(self):
        #build spin operators:
        #build total spin operators S+
        Sp = sparse.csr_matrix((self.hsize,self.hsize), dtype=numpy.complex128)
        for i in range(self.ntot//2):
            Sp += self.FH_list[2*i].dot(self.FH_list[2*i+1].getH())
        #build total spin operators S-
        Sm = sparse.csr_matrix((self.hsize,self.hsize), dtype=numpy.complex128)
        for i in range(self.ntot//2):
            Sm += self.FH_list[2*i+1].dot(self.FH_list[2*i].getH())
        #build total Sz and N
        self.Sz = sparse.csr_matrix((self.hsize,self.hsize), dtype=numpy.complex128)
        self.Ntot = sparse.csr_matrix((self.hsize,self.hsize), dtype=numpy.complex128)
        for i in range(self.ntot//2):
            self.Sz += (0.5*self.FH_list[2*i].dot(self.FH_list[2*i].getH()) - 0.5*self.FH_list[2*i+1].dot(self.FH_list[2*i+1].getH()) )
            self.Ntot += (self.FH_list[2*i].dot(self.FH_list[2*i].getH()) + self.FH_list[2*i+1].dot(self.FH_list[2*i+1].getH()) )
        #build S^2 operator
        self.S2 = Sm.dot(Sp)+self.Sz.dot(self.Sz)+self.Sz

    def build_Hemb(self, D, H1E, LAMBDA, V2E, spin_pen=0.0):
        '''
        Build Hemb matrix.
        Input:
            D: hybridization matrix
            H1E: local one-body
            LAMBDA: bath one-body
            V2E: local two-body interaction
        Return:
            Hemb: embedding Hamiltonian
        '''
        self.Hemb = sparse.csr_matrix((self.hsize,self.hsize), dtype=numpy.complex128)
        self.H2e = sparse.csr_matrix((self.hsize,self.hsize), dtype=numpy.complex128)
        #build local one-body part
        for i in range(self.nimp):
            for j in range(self.nimp):
                self.Hemb += H1E[i,j]*self.FH_list[i].dot(self.FH_list[j].getH())
        #build hybridization part
        for i in range(self.nbath):
            for j in range(self.nimp):
                self.Hemb += D[i,j]*self.FH_list[j].dot(self.FH_list[self.nimp+i].getH() )
                self.Hemb += D[i,j].conj()*self.FH_list[self.nimp+i].dot(self.FH_list[j].getH() )
        #build bath part
        for i in range(self.nbath):
            for j in range(self.nbath):
                self.Hemb += LAMBDA[i,j]*self.FH_list[j+self.nimp].getH().dot(self.FH_list[i+self.nimp] )
        #build local two-body part
        for i in range(self.nimp):
            for j in range(self.nimp):
                for k in range(self.nimp):
                    for l in range(self.nimp):
                        if numpy.abs(V2E[i,k,j,l]) > 1e-6:
                            self.Hemb += 0.5*V2E[i,k,j,l]*self.FH_list[i].dot(self.FH_list[j]).dot(self.FH_list[l].getH()).dot(self.FH_list[k].getH())
                            self.H2e += 0.5*V2E[i,k,j,l]*self.FH_list[i].dot(self.FH_list[j]).dot(self.FH_list[l].getH()).dot(self.FH_list[k].getH())
        #add spin penalty:
        self.Hemb += spin_pen*self.S2
        assert(abs( (self.Hemb - self.Hemb.getH()).max() ) < 1e-12), 'Hamiltonian is not Hermitian! H.getH()-H='+str(abs( (self.Hemb - self.Hemb.getH()).max() ))
        return self.Hemb

    def calc_density_matrix(self, T=0.001):
        #'''
        #calculate density matrix.
        #Input:
        #    FH_list: fermion operator list
        #    vec: ground state eigenvector
        #Return:
        #    dm: density matrix
        #'''
        #dm = numpy.zeros((self.ntot,self.ntot),dtype=numpy.complex128)
        #for i in range(self.ntot):
        #    for j in range(self.ntot):
        #        dm[i,j]=numpy.trace(self.evecs[:,:self.deg].conj().T.dot(self.FH_list[i].dot(self.FH_list[j].getH().dot(self.evecs[:,:self.deg]))))/self.deg
        #return dm
        '''
        calculate density matrix using thermal Boltzmann factor
        '''
        exp_bE = numpy.diag(numpy.exp(-self.evals_e0/T))
        Z = numpy.trace(exp_bE)
    
        dm = numpy.zeros((self.ntot,self.ntot),dtype=complex)
        for i in range(self.ntot):
            for j in range(self.ntot):
                dm[i,j] = numpy.trace( self.evecs.dot(exp_bE).dot(self.evecs.conj().T).dot( (self.FH_list[i].dot( self.FH_list[j].getH() )).todense() ) )/Z
        return dm

    def calc_density_matrix_from_wf(self, wf):
        '''
        calculate density matrix.
        Input:
            FH_list: fermion operator list
            vec: ground state eigenvector
        Return:
            dm: density matrix
        '''
        dm = numpy.zeros((self.ntot,self.ntot),dtype=numpy.complex128)
        for i in range(self.ntot):
            for j in range(self.ntot):
                dm[i,j] = wf[:].conj().T.dot(self.FH_list[i].dot(self.FH_list[j].getH().dot(wf[:])))
        return dm

    def reduced_rho(self, NA, NB):
        '''
        Compute local reduce density matrix
        Input:
            NA: number of the retained orbitals
            NB: number of the traced out orbitals
        Return:
            rho: reduced density matrix
        '''
        hsize_imp = 2**NA
        hsize_bath = 2**NB
        self.rho = numpy.zeros((hsize_imp,hsize_imp))
        for i in range(hsize_imp):
            for j in range(hsize_imp):
                for k in range(hsize_bath):
                    for s in range(self.deg):
                        self.rho[i,j] += self.evecs[i*hsize_bath+k,s]*self.evecs[j*hsize_bath+k,s].conj()/self.deg
        return self.rho

    def calc_double_occ(self,idx,T=0.001):
        '''
        calculate double occupancy at orbital idx.
        Input:
            idx: index for the orbital (also idx+1) where the double occupancy is calculated.
            FH_list: fermion operator list.
            vec: ground state eigenvector.
        Return:
            double occupancy
        '''
        #return numpy.trace(self.evecs[:,:self.deg].conj().T.dot(self.FH_list[idx].dot(self.FH_list[idx].getH().dot(
        #    self.FH_list[idx+1].dot(self.FH_list[idx+1].getH().dot(self.evecs[:,:self.deg]))))))/self.deg
        exp_bE = numpy.diag(numpy.exp(-self.evals_e0/T))
        Z = numpy.trace(exp_bE)
        docc = numpy.trace( self.evecs.dot(exp_bE).dot(self.evecs.conj().T).dot( (self.FH_list[idx].dot(self.FH_list[idx].getH().dot(
            self.FH_list[idx+1].dot(self.FH_list[idx+1].getH()))) ).todense() ) )/Z
        return docc

    def calc_double_occ_from_wf(self,idx, wf):
        '''
        calculate double occupancy at orbital idx.
        Input:
            idx: index for the orbital (also idx+1) where the double occupancy is calculated.
            FH_list: fermion operator list.
            vec: ground state eigenvector.
        Return:
            double occupancy
        '''
        return wf[:].conj().T.dot(self.FH_list[idx].dot(self.FH_list[idx].getH().dot(
            self.FH_list[idx+1].dot(self.FH_list[idx+1].getH().dot(wf[:])))))

    def compute_E2loc(self,T=0.001):
        '''
        calculate interacting energy.
        '''
        #return numpy.trace(self.evecs[:,:self.deg].conj().T.dot(self.H2e.dot(self.evecs[:,:self.deg])))/self.deg
        exp_bE = numpy.diag(numpy.exp(-self.evals_e0/T))
        Z = numpy.trace(exp_bE)
        E2loc = numpy.trace( self.evecs.dot(exp_bE).dot(self.evecs.conj().T).dot( self.H2e.todense() ) )/Z
        return E2loc


    def solve_Hemb(self,num_eig=10, verbose=0):
        '''
        solve embeding Hamiltonian.
        Input:
            D: hybridization matrix
            H1E: local one-body
            LAMBDA: bath one-body
            V2E: local two-body interaction
            FH_list: fermion operator list
            idx: index for the orbital (also idx+1) where the double occupancy is calculated.
            spin_pen: prefactor for the S^2 penalty term.
        Return:
            density matrix
            double occupancy
        '''
        # print number of eigenvalues
        # using scipy.eigsh
        #self.evals, self.evecs = eigsh(self.Hemb, k=num_eig, which='SA')
        # using primme.eigsh
        #self.evals, self.evecs = primme.eigsh(self.Hemb, num_eig, tol=1e-12, which='SA')
        # use eigh for full eigenstates
        self.evals, self.evecs = numpy.linalg.eigh(self.Hemb.todense())
        it = 1
        self.deg = 1
        self.e0 = self.evals[0]
        while numpy.abs(self.e0 - self.evals[it]) < 5e-3 and it<(num_eig-1):
            self.deg += 1
            it += 1
        self.evals_e0 = self.evals - self.e0
        #if verbose > 0:
        print('eigenvalues=',self.evals[:num_eig])
        print('# Energy            S^2            Sz            Ntot')
        for i in range(num_eig):
            print(self.evals[i], self.evecs[:,i].conj().T.dot( self.S2.dot( self.evecs[:,i] ) ), self.evecs[:,i].conj().T.dot( self.Sz.dot( self.evecs[:,i] ) ), self.evecs[:,i].conj().T.dot( self.Ntot.dot( self.evecs[:,i] ) ))
            print('deg=',self.deg)

