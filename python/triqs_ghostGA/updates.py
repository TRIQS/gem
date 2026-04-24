import numpy as np
from .core import Fragment
from utility.utilities import funcMat, denR, calc_Lambda, calc_Lambda_c
from utility.delta_fit import update_self_energy_thermal


def update_self_energy(self,move_pen=1e-8):
    '''
    This function update the self-energy parameters Lambda and R
    '''

    self.denMat = self.solver.calc_density_matrix()
    cdagc = self.denMat[:self.nimp,:self.nimp]
    cdagf = self.denMat[:self.nimp,self.nimp:]
    fdagf = self.denMat[self.nimp:,self.nimp:]
    self.Delta_p = np.eye(self.nbath) - fdagf

    if(self.thermal):
        #Here one would use move_pen
        raise ValueError('Not present now, to extract proper function from branch Temperature')
    else:
        R_new = np.transpose( cdagf.dot( funcMat(self.Delta_p, denR)) )
        L_new = calc_Lambda( R_new, self.Lambda_c, self.Delta_p, self.D, self.H_list )
    
    if(self.spin_sym):
        R_new=np.kron( R_new[::2,::2], np.eye(2) )
        L_new=np.kron( L_new[::2,::2], np.eye(2) )

    self.R = R_new.copy()
    self.Lambda = L_new.copy()

    return self.R, self.Lambda

def update_hybridization(self, move_pen=1e-8):
    '''
    This function update the hybridization parameters Lambda_c and D
    '''

    #Here lattice should have been called before, self.Delta_p and self.ERD
    #should have the new values.

    if(self.thermal):
        #Here one would use move_pen
        raise ValueError('Not present now, to extract proper function from branch Temperature')
    else:
        D_new = np.dot(funcMat(self.Delta_p, denR),np.transpose(self.ERD))
        Lc_new = calc_Lambda_c(self.R, self.Lambda, self.Delta_p, self.D, self.H_list)
            
    if(self.spin_sym):
        D_new=np.kron( D_new[::2,::2], np.eye(2) )
        Lc_new=np.kron( Lc_new[::2,::2], np.eye(2) )

    self.D = D_new.copy()
    self.Lambda_c = Lc_new.copy()

    return self.D, self.Lambda_c