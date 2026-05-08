#######################################################
# Simple Exact Diagonalization solver
# Author: Tsung-Han Lee
# Email:  henhans74716@gmail.com
#######################################################

import warnings
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.linalg import block_diag
import numpy as np
from numba import jit
import h5py

from math import factorial
from itertools import combinations

#SAMUELE'S QUESTION
# Global change of debug with verbose options for printing


#DEFINITION
class SimpleED(object):
    '''
    Simple exact diagonalization class aim to solve general impurity Hamiltonian.
    '''
    def __init__(self, norb, use_Ntot=False, use_Sz=False,
                 thermal=False, dtype=np.float64, Nparticle=None,
                 solver_params=None,
                 ):
        '''
        Constructor.
        Input:
          norb: int. number of orbitals including spin
          use_Ntot: bool. If using particle number symmetry
          use_Sz: bool. If using Sz symmetry
          thermal: bool. If performing thermal calculation.
          dtype: dtype. data type of the Hamiltonian
        '''
        self.type = "SimpleED"
        self.solver_params = solver_params if solver_params is not None else {}
        self.norb = norb # number of orbitals
        self.use_Ntot = use_Ntot # use Ntot symmetry
        self.use_Sz = use_Sz # use Sz symmtery
        self.thermal = thermal # use thermal ensemble
        if( self.thermal and ( self.use_Ntot or self.use_Sz ) ):
            warnings.warn("Thermal calculations should be performed without symmetries, otherwise the partition function is not correctly computed.")

        self.strb = '{0:0'+str(norb)+'b}' # string to convert integer to binary string
        self.strb_red = '{0:0'+str(norb//2)+'b}' # string to convert integer to binary string in reudced Hilbert space
        self.data_type = dtype # data type of the Hamiltonian
        self.Hone = None # initialize None for one-body part
        self.Htwo = None # initialize None for two-body part

        # create basis in the ground space half-filled and optionally Sz=0.
        if use_Ntot == True and use_Sz == False: # Ntot symmetry
            if Nparticle == None:
                self.basis = table_ep(norb,norb//2)
            else:
                self.basis = table_ep(norb,Nparticle)
        if use_Ntot == True and use_Sz == True: # Ntot and Sz symmetry
            if Nparticle == None:
                self.basis = table_es(norb,norb//2,0)
            else:
                self.basis = table_es(norb,Nparticle,0)
        if use_Ntot == False and use_Sz == False: # no symmetry
            self.basis = np.arange(2**(norb))

        self.hsize = len(self.basis) # hilbert space size
        print('size of basis= {:d}'.format( len(self.basis) ))#, 'data type of basis=', self.basis.dtype)

        print('build denmat_op')
        self.build_denmat_op()
        print('build S2_op')
        self.build_S2_op()


#MANDATORY FUNCTIONS
    def build_Hemb(self, D, eloc, Lambdac, V2E, debug=False, verbose=0,
                   spin_pen=None, sz_pen=None, sx_pen=None, sy_pen=None):
        '''
        build the Hamiltonian and return Hamiltonian
        '''
        spin_pen = self.solver_params.get('spin_pen', 0) if spin_pen is None else spin_pen
        sz_pen   = self.solver_params.get('sz_pen',   0) if sz_pen   is None else sz_pen
        sx_pen   = self.solver_params.get('sx_pen',   0) if sx_pen   is None else sx_pen
        sy_pen   = self.solver_params.get('sy_pen',   0) if sy_pen   is None else sy_pen
        print('build one-body')
        self.build_h1e(eloc, D, Lambdac, 0, verbose=verbose)
        self.build_one_body(self.h1e)
        print('build two-body')
        if self.Htwo is None:
            self.build_two_body(V2E)
        print('one-body + two-body')
        self.M = {"up": self.h1e[::2, ::2], "dn": self.h1e[1::2, 1::2]}
        self.Ham = self.Hone + self.Htwo + spin_pen*self.S2
        self.Ham += sz_pen*self.Sz.dot(self.Sz) + sx_pen*self.Sx.dot(self.Sx) + sy_pen*self.Sy.dot(self.Sy)
        print('done')
        if debug:
            return self.Ham

    def solve_Hemb(self, num_eig=1, which=None, tol=None, verbose=0, T=0.0):
        '''
        diagonalize the Hamiltonian
        '''
        which = self.solver_params.get('which', 'SA') if which is None else which
        tol   = self.solver_params.get('tol',   1e-8) if tol   is None else tol
        print('diagonalizing num_eig= {:d}'.format(num_eig))
        if(self.hsize < 4000):
            print("Doing FULL diagonalization")
            vals, vecs = eigh(self.Ham.toarray())
        else:
            vals, vecs = eigsh(self.Ham,k=num_eig,which=which,tol=tol)
        so = vals.argsort()
        vals = vals[so]
        vecs = vecs[:, so]
        self.gs_wf = vecs[:,0]
        self.gs_ene = vals[0]
        self.evals = vals
        self.evecs = vecs
        it = 1
        self.deg = 1
        self.e0 = self.evals[0]
        self.Zpart = 1 #partition function for thermal and degeneracies, will replace deg
        self.Tstates = 1 #number of thermal states
        self.bw_list = [1] #list of boltzmann weights
        if(T>0.0):
            beta=1/T
            print('Building thermal partition function')
            for eit in vals[1:]:
                boltz_weight = np.exp(-beta*(eit-self.gs_ene))
                if(boltz_weight>1e-8):
                    self.bw_list.append(boltz_weight*1.0)
                    self.Zpart += boltz_weight
                    self.Tstates += 1
                else:
                    break
        else:
            print('Building GS partition function')
            if num_eig > 1:
                for it in range(1,num_eig):
                    if np.abs(self.e0 - self.evals[it]) < 1e-4:#1e-5:
                        self.deg += 1
                        self.Zpart += 1
                        self.bw_list.append(1.0)
                        self.Tstates += 1
                        it += 1
        if True:
            print('# Energy\t\tS2\t\t\tSz\t\t\tSx\t\t\tSy\t\t\tSz2\t\t\tSx2\t\t\tSy2')
            if(verbose>1):
              for i in range(int(self.Tstates)):
                S2 = vecs[:,i].conj().T.dot(self.S2.dot(vecs[:,i]))
                Sz = vecs[:,i].conj().T.dot(self.Sz.dot(vecs[:,i]))
                Sz2 = vecs[:,i].conj().T.dot(self.Sz.dot(self.Sz).dot(vecs[:,i]))
                Sx = vecs[:,i].conj().T.dot(self.Sx.dot(vecs[:,i]))
                Sx2 = vecs[:,i].conj().T.dot(self.Sx.dot(self.Sx).dot(vecs[:,i]))
                Sy = vecs[:,i].conj().T.dot(self.Sy.dot(vecs[:,i]))
                Sy2 = vecs[:,i].conj().T.dot(self.Sy.dot(self.Sy).dot(vecs[:,i]))
                print("%.12e  \t%.1e+%.1ej\t%.1e+%.1ej\t%.1e+%.1ej\t%.1e+%.1ej\t%.1e+%.1ej\t%.1e+%.1ej\t%.1e+%.1ej" %
                      (vals[i], S2.real, S2.imag, Sz.real, Sz.imag, Sx.real, Sx.imag, Sy.real, Sy.imag, Sz2.real, Sz2.imag, Sx2.real, Sx2.imag, Sy2.real, Sy2.imag))
                print('deg=',self.deg,' - Boltzmann weight=',self.bw_list[i])
                print('')
        #CHECK THAN LENGTHS ARE CORRECTS FOR BW_LIST VALS AND SO ON
        self.evals=self.evals[:self.Tstates]
        self.evecs=self.evecs[:,:self.Tstates]
        
        return self.gs_wf, self.gs_ene

    def calc_density_matrix(self):
        '''
        Compute denstiy matrix.
        Return:
          denmat: numpy.array. Densty matrix, <c^\dagger_i c_j>, of the system.
        '''
        dm = np.zeros((self.norb, self.norb), dtype=self.data_type)

        bw = np.asarray(self.bw_list)
        Z = np.sum(bw)
        
        U = self.evecs    # shape (dim, self.Tstates)
        W = np.diag(bw)   # Boltzmann weights
        
        for i in range(self.norb):
            for j in range(self.norb):
                dm[i, j] = np.trace(
                    U.conj().T @ self.denmat_op[(i, j)] @ U @ W
                ) / Z

        self.dm = dm
        return dm



    def compute_E1loc(self, nimp):
        '''
        Compute local energy including local one and two-body term from a given set od thermal states
        Works also at zero Temperature
        Input:
        Return:
          Eloc: float. Total local energy.
        '''
        U = self.evecs
        bw = np.asarray(self.bw_list)
        Z = np.sum(bw)

        # dm_imp = sum_n bw[n] |psi_n><psi_n|  restricted to imp block, divided by Z
        Uimp = U[:nimp, :]  # (nimp, deg)
        dm_imp = (Uimp * bw) @ Uimp.conj().T / Z  # weights columns

        return np.trace(self.h1e[:nimp, :nimp] @ dm_imp.T)

    def compute_E2loc(self):
        '''
        Compute local energy including local one and two-body term from a given set od thermal states
        Works also at zero Temperature
        Input:
        Return:
          Eloc: float. Total local energy.
        '''
        U = self.evecs
        bw = np.asarray(self.bw_list)
        Z = np.sum(bw)

        return np.trace(U.conj().T @ self.Htwo @ U @ np.diag(bw)) / Z



#AUXILIARY FUNCTIONS
    def build_two_body(self,Umatrix, debug=False):
        '''
        build two-body interaction, which usually does not change during the self-consistent iteration
        and can be build once and for all.
        Input:
          Umatrix: numpy array, with index (i,j,k,l) representing 0.5*V_ijkl*CH_i*CH_k*C_l*C_j
          dtype: data type of the matrix
        '''
        bit_max = 2**(self.norb-1)
        self.Htwo = csc_matrix((self.hsize,self.hsize), dtype=self.data_type)
        for i in range(Umatrix.shape[0]):
            for j in range(Umatrix.shape[1]):
                for k in range(Umatrix.shape[2]):
                    for l in range(Umatrix.shape[3]):
                        # check if l==j or i==k or U=0, if true it has 0 contribution
                        if l==j or i==k or abs(Umatrix[i,j,k,l])<1e-8:
                            continue # 0 contribution
                        else:
                            row_ind, col_ind, data = build_two_body_ijkl_csc_2(i, j, k, l, self.basis, bit_max, self.norb)
                            self.Htwo +=  0.5*Umatrix[i,j,k,l]*csc_matrix( (data, (row_ind, col_ind)), shape=(self.hsize,self.hsize),dtype=self.data_type)

    def build_h1e(self, eloc, D, Lambdac, mu, verbose=0):
        self.h1e = np.zeros((self.norb,self.norb), dtype=np.complex128)
        nimp = eloc.shape[0]
        self.h1e[:nimp,:nimp] = eloc - mu*np.eye(nimp)
        self.h1e[:nimp,nimp:] = D.T
        self.h1e[nimp:,nimp:] = -Lambdac
        self.h1e[nimp:,:nimp] = D.conj()

        if(verbose>3):
          print("h1e")
          print(self.h1e)

    def build_one_body(self, H1E):
        '''
        build one body part of Hamiltonian using denmat operators.
        Input:
          H1E: numpy array, with index (i,j). local part of one-body hamiltonian
          dtype: data type of the matrix
          debug: bool. print out debug message
        '''
        self.Hone = csc_matrix((self.hsize,self.hsize),dtype=self.data_type)

        for i in range(0,self.norb):
            for j in range(0,self.norb):
                if np.abs(H1E[i,j])<1e-8:
                    continue # 0 contribution
                else:
                    self.Hone += H1E[i,j]*self.denmat_op[(i,j)]


    def build_denmat_op(self,debug=False):
        '''
        build the density matrix operators into a dictionary.
        denmat_op: key: tuple (i,j) indicating the orbital i and j.
                   element: scipy.sparse.csc_matrix storing the operator C^\dagger_i C_j
        '''
        self.denmat_op = {}
        bit_max = 2**(self.norb-1)
        use_numba = True
        for i in range(self.norb):
            for j in range(self.norb):
                row_ind, col_ind, data = build_cid_cj_csc(i, j, self.basis, bit_max, self.norb, debug=False)
                if len(row_ind) > 0:
                    if np.max(row_ind) >= self.hsize:
                        print(self.basis)
                        print(row_ind)
                self.denmat_op[(i,j)] = csc_matrix( (data, (row_ind, col_ind)), shape=(self.hsize,self.hsize),dtype=self.data_type)

    def build_S2_op(self,debug=False):
        '''
        build total S2 operator
        '''
        #build total spin operator S+
        Sp = csc_matrix((self.hsize,self.hsize),dtype=self.data_type)
        for i in range(self.norb//2):
            #Sp += FH_list[2*i].dot(FH_list[2*i+1].getH())
            Sp += self.denmat_op[(2*i,2*i+1)]
        #build total spin operator S-
        Sm = csc_matrix((self.hsize,self.hsize),dtype=self.data_type)
        for i in range(self.norb//2):
            #Sm += FH_list[2*i+1].dot(FH_list[2*i].getH())
            Sm += self.denmat_op[(2*i+1,2*i)]
        #build total spin opertor Sz
        Sz = csc_matrix((self.hsize,self.hsize),dtype=self.data_type)
        for i in range(self.norb//2):
            #Sz += ( 0.5*FH_list[2*i].dot(FH_list[2*i].getH()) - 0.5*FH_list[2*i+1].dot(FH_list[2*i+1].getH()) )
            Sz += ( 0.5*self.denmat_op[(2*i,2*i)] - 0.5*self.denmat_op[(2*i+1,2*i+1)] )
        #build S^2 operator
        self.S2 = Sm.dot(Sp)+Sz.dot(Sz)+Sz
        self.Sz = Sz
        self.Sx = 0.5*(Sp + Sm)
        self.Sy = 0.5*(Sp - Sm)/1j


    def build_docc_op(self,i,debug=False):
        '''
        build the double occupancy operators into a dictionary.
        denmat_op: key: int i indicating the orbital i (even number).
                   element: scipy.sparse.csc_matrix storing the operator C^\dagger_{i}C_{i}C^\dagger_{i+1}C_{i+1}
        '''
        bit_max = 2**(self.norb-1)
        row_ind = []
        col_ind = []
        data = []
        for bsrid,bsr in enumerate(self.basis):
            tmp_bit1 = bit_max>>(i+1) #2**(self.norb-1-(i+1))
            tmp_bit2 = bit_max>>(i+1) #2**(self.norb-1-(i+1))
            tmp_bit3 = bit_max>>i #2**(self.norb-1-i)
            tmp_bit4 = bit_max>>i #2**(self.norb-1-i)
            if ((tmp_bit1&bsr)!=tmp_bit1 or (tmp_bit3&bsr)!=tmp_bit3): # if i is 0 or if i+1 is 0 continue
                #print('continue')
                continue
            # annhilate particles on l and j
            #bsltmp = bsr ^ tmp_bit1
            # create particles on i and k
            #bsl = bsltmp | tmp_bit2
            #annhilate particle i
            bsltmp1 = bsr ^ tmp_bit1
            #create particle i+1
            bsltmp2 = bsltmp1 | tmp_bit2
            #annhilate particle i
            bsltmp3 = bsltmp2 ^ tmp_bit3
            #create particle k
            bsl = bsltmp3 | tmp_bit4
            #check if bsl is in basis
            id_bsl = np.searchsorted(self.basis, bsl)
            #print( id_bsl, bsl, bsr, strb.format(bsl), strb.format(bsr), basis[id_bsl], len(basis) )
            if bsl != self.basis[id_bsl]: # The c_i^\dagger c_j may lead to the state that is not in the symmetry constrained states.
                continue
            else:
                bslid = id_bsl
            # compute the sign
            count = 0
            bit_tmp = ( ((1 << i+1) - 1)  &  (bsr >> (self.norb-i+1) ) )
            while ( bit_tmp ):
                count += bit_tmp &1
                bit_tmp >>=1
            # extract the first i bits from bsltmp and count the 1s
            bit_tmp = ( ((1 << i+1) - 1)  &  (bsltmp1 >> (self.norb-i+1) ) )
            #print 'bsltmp', i, self.strb.format(bsltmp), self.strb.format(bit_tmp), self.strb.format(bsltmp)[:i]
            while ( bit_tmp ):
                count += bit_tmp &1
                bit_tmp >>=1
            bit_tmp = ( ((1 << i) - 1)  &  (bsltmp2 >> (self.norb-i) ) )
            while ( bit_tmp ):
                count += bit_tmp &1
                bit_tmp >>=1
            # extract the first i bits from bsltmp and count the 1s
            bit_tmp = ( ((1 << i) - 1)  &  (bsltmp3 >> (self.norb-i) ) )
            #print 'bsltmp', i, self.strb.format(bsltmp), self.strb.format(bit_tmp), self.strb.format(bsltmp)[:i]
            while ( bit_tmp ):
                count += bit_tmp &1
                bit_tmp >>=1
            sign = 1
            #if count%2 == 1:
            if count&1 ==1: # equivlaent
                sign = -1
            #if debug:
            #  print i, j, k, l, 'bsrid=',bsrid,'bsr=',bsr, self.strb.format(bsr),'bslid=',bslid,'bsl=',bsl, self.strb.format(bsl), 'cumu=',cumu
            #construct csc matrix index, pointer, and data
            row_ind.append(bslid)
            col_ind.append(bsrid)
            data.append(sign)
        #print(row_ind, col_ind, data)
        return csc_matrix( (data, (row_ind, col_ind)), shape=(self.hsize,self.hsize),dtype=self.data_type)


    def calc_double_occ(self, i):
        '''
        Compute thermal double occupancy on orbital i and i+1.
        '''
        docc_op = self.build_docc_op(i)
        
        U = self.evecs
        bw = np.asarray(self.bw_list)
        Z = np.sum(bw)
        
        return np.trace(
            U.conj().T @ docc_op @ U @ np.diag(bw)
        ) / Z




#UNSURE
    def h5write_gs(self,filename,group_path,name):
        #check that group at group_path exists
        with h5py.File(filename,"a") as f:
            if group_path not in f:
                f.create_group(group_path)
            f[group_path][name] = self.gs_wf
        return

    def h5read_state(self,filename,group_path,name):
        #check that group at group_path exists
        with h5py.File(filename,"r") as f:
            return f[group_path][name][:]

    def inner(self,bra,ket,operator=None):
        if operator is not None:
            return np.vdot(bra,operator.dot(ket))
        else:
            return np.vdot(bra,ket)


# Binary basis utilities


def table_ep(nstate,nparticle,dtype=np.int64):
    '''
    This function generates the table of binary representations of a particle-conserved and spin-non-conserved basis.
    '''
    result=np.zeros(factorial(nstate)//factorial(nparticle)//factorial(nstate-nparticle),dtype=dtype)
    buff=combinations(range(nstate),nparticle)
    for i,v in enumerate(buff):
        basis=0
        for num in v:
            basis+=(1<<num)
        result[i]=basis
    result.sort()
    return result

def table_es(nstate,nparticle,spinz,dtype=np.int64):
    '''
    This function generates the table of binary representations of a particle-conserved and spin-conserved basis.
    '''
    n,nup,ndw=nstate//2,(nparticle+int(2*spinz))//2,(nparticle-int(2*spinz))//2
    result=np.zeros(factorial(n)//factorial(nup)//factorial(n-nup)*factorial(n)//factorial(ndw)//factorial(n-ndw),dtype=dtype)
    buff_up=list(combinations(range(1,2*n,2),nup))
    buff_dw=list(combinations(range(0,2*n,2),ndw))
    count=0
    for vup in buff_up:
        buff=0
        for num in vup:
            buff+=(1<<num)
        for vdw in buff_dw:
            basis=buff
            for num in vdw:
                basis+=(1<<num)
            result[count]=basis
            count+=1
    result.sort()
    return result


@jit(nopython=True)#,cache=True)
def residues(norb,determinant):
    ''' Returns list of residues, which is all possible ways to remove two
        electrons from a given determinant with number of orbitals norb
    '''
    residue_list = []
    nonzero = determinant.bit_count() 
    for i in range(norb):
        mask1 = (1 << i)
        for j in range(i):
            mask2 = (1 << j)
            mask = mask1 ^ mask2
            if (determinant & ~mask).bit_count()  == (nonzero - 2):
                residue_list.append(determinant & ~mask)
    return residue_list

@jit(nopython=True)#,cache=True)
def add_particles(norb,residue_list):
    ''' Returns list of determinants, which is all possible ways to add two
        electrons from a given residue_list with number of orbitals norb
    '''
    determinants = []
    for residue in residue_list:
        determinant = residue
        for i in range(norb):
            mask1 = (1 << i)
            if not bool(determinant & mask1):
                one_particle = determinant | mask1
                for j in range(i):
                    mask2 = (1 << j)
                    if not bool(one_particle & mask2):
                        two_particle = one_particle | mask2
                        determinants.append(two_particle)
    #return [format(det,'#0'+str(n_orbitals+2)+'b') for det in list(set(determinants))]
    return list(set(determinants))


@jit(nopython=True)
def find_count(i,j,bsr,norb,bsltmp):
  """
  Count occupied orbitals (1-bits) before j in bsr and before i in bsltmp.
  Used to compute the fermionic sign: (-1)**count for c_i^† c_j.
  """
  # extract the first j bits from bsr and count the 1s
  bit_tmp = ( ((1 << j) - 1)  &  (bsr >> (norb-j) ) )
  count = 0
  while ( bit_tmp ):
    count += bit_tmp &1
    bit_tmp >>=1
  # extract the first i bits from bsltmp and count the 1s
  bit_tmp = ( ((1 << i) - 1)  &  (bsltmp >> (norb-i) ) )
  while ( bit_tmp ):
    count += bit_tmp &1
    bit_tmp >>=1
  return count

@jit(nopython=True)
def build_cid_cj_csc(i, j, basis, bit_max, norb, debug=False):
    #print 'i=',i, 'j=', j, 'bit_max=', bit_max
    row_ind = []
    col_ind = []
    data = []

    for bsrid,bsr in enumerate(basis):
        # temporary bit for fliping the bit on j and i.
        tmp_bit1 = bit_max>>j
        tmp_bit2 = bit_max>>i
        # check if bit on j is 1 and if bit on i is 0 and i!=j
        if (tmp_bit1&bsr)!=tmp_bit1 or (tmp_bit2&bsr)==tmp_bit2 and i!=j:
            continue
        # annhilate particle on j
        bsltmp = bsr ^ tmp_bit1
        # create particles on i
        bsl = bsltmp | tmp_bit2
        # binary search the index for the final state bsl
        id_bsl = np.searchsorted(basis, bsl)
        if bsl != basis[id_bsl] or id_bsl>=len(basis): # The c_i^\dagger c_j may lead to a state that is not in the symmetry constrained states.
            continue
        else:
            bslid = id_bsl
        #determine sign
        count = find_count(i,j,bsr,norb,bsltmp)
        sign = 1
        if count&1 ==1: # equivlaent
            sign = -1
        #construct csc matrix index, pointer, and data
        row_ind.append(bslid)
        col_ind.append(bsrid)
        data.append(sign)
    return row_ind, col_ind, data

@jit(nopython=True)
def build_two_body_ijkl_csc_2(i, j, k, l, basis, bit_max, norb, debug=False):
    row_ind = []
    col_ind = []
    data = []
    #build <bsl|Htwo|bsr>
    for bsrid,bsr in enumerate(basis):
        # see if l and j orbital are occupied if not continue
        tmp_bit1 = bit_max>>j #2**(self.norb-1-j)
        tmp_bit2 = bit_max>>l #2**(self.norb-1-l)
        tmp_bit3 = bit_max>>k #2**(self.norb-1-k)
        tmp_bit4 = bit_max>>i #2**(self.norb-1-i)
        if ((tmp_bit1&bsr)!=tmp_bit1 or (tmp_bit2&bsr)!=tmp_bit2): # if j is 0 or if l is 0 continue
            #print('continue')
            continue
        #annhilate particle j
        bsltmp1 = bsr ^ tmp_bit1
        #annhilate particle l
        bsltmp2 = bsltmp1 ^ tmp_bit2
        #create particle k
        bsltmp3 = bsltmp2 | tmp_bit3
        #create particle i
        bsl = bsltmp3 | tmp_bit4
        #check if bsl is in basis
        id_bsl = np.searchsorted(basis, bsl)
        if bsl != basis[id_bsl] or id_bsl>=len(basis): # The c_i^\dagger c_j may lead to the state that is not in the symmetry constrained states.
            continue
        else:
            bslid = id_bsl
        # compute the sign
        count = 0
        bit_tmp = ( ((1 << j) - 1)  &  (bsr >> (norb-j) ) )
        while ( bit_tmp ):
            count += bit_tmp &1
            bit_tmp >>=1
        # extract the first i bits from bsltmp and count the 1s
        bit_tmp = ( ((1 << l) - 1)  &  (bsltmp1 >> (norb-l) ) )
        while ( bit_tmp ):
            count += bit_tmp &1
            bit_tmp >>=1
        bit_tmp = ( ((1 << k) - 1)  &  (bsltmp2 >> (norb-k) ) )
        while ( bit_tmp ):
            count += bit_tmp &1
            bit_tmp >>=1
        # extract the first i bits from bsltmp and count the 1s
        bit_tmp = ( ((1 << i) - 1)  &  (bsltmp3 >> (norb-i) ) )
        while ( bit_tmp ):
            count += bit_tmp &1
            bit_tmp >>=1
        sign = 1
        #if count%2 == 1:
        if count&1 ==1: # equivlaent
            sign = -1
        #construct csc matrix index, pointer, and data
        row_ind.append(bslid)
        col_ind.append(bsrid)
        data.append(sign)
    return row_ind, col_ind, data

@jit(nopython=True)
def build_rholoc_onfly(basis,gs_wf,rholoc,bipart_smap):
    '''
    Compute local reduced many-body density matrix onfly (without storing |\phi><\phi|)
    '''
    for i in range(len(basis)):
        for j in range(len(basis)):
            if bipart_smap[i,2] == bipart_smap[j,2] and abs(gs_wf[i]*gs_wf[j]) > 1e-12:
                rholoc[bipart_smap[i,1],bipart_smap[j,1]] += gs_wf[i]*gs_wf[j]
    return rholoc