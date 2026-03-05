'''
    PyDMET: a python implementation of density matrix embedding theory
    Copyright (C) 2014, 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import numpy as np
from scipy.optimize import fmin_slsqp

class DIIS(object):
    def __init__( self, numVecs ):
        self.errors  = []
        self.states  = []
        self.numVecs = numVecs
        
    def append( self, error, state ):
        self.errors.append(error)
        self.states.append(state)
        
        if len( self.errors ) > self.numVecs:
            self.errors.pop(0)
            self.states.pop(0)

    def flush( self ):
        self.errors = []
        self.states = []
        
    def Solve( self ):
        nStates = len( self.errors )
        mat = np.zeros([ nStates+1 , nStates+1 ], dtype=complex)#dtype=float)
        for cnt in range(0, nStates):
            mat[ cnt, nStates ] = 1.0
            mat[ nStates, cnt ] = 1.0
            for cnt2 in range(0, nStates):
                mat[ cnt, cnt2 ] = np.sum( np.multiply( self.errors[cnt].conj() , self.errors[cnt2] ) )
            #for cnt2 in range(cnt, nStates):
            #    mat[ cnt, cnt2 ] = np.sum( np.multiply( self.errors[cnt] , self.errors[cnt2] ) )
            #    mat[ cnt2, cnt ] = mat[ cnt, cnt2 ]
        #print('errors=')
        #print(self.errors)
        #print('mat=')
        #print(mat)
        vec = np.zeros([ nStates+1 ], dtype=complex)#dtype=float)
        vec[ nStates ] = 1.0
        coeff = np.linalg.tensorsolve(mat, vec)
        coeff = coeff[:-1]
        thestate = coeff[0] * self.states[0]
        for cnt in range(1, len(coeff)):
            thestate += coeff[cnt] * self.states[cnt]
        print("   DIIS :: Coefficients (latest first) = ",coeff[::-1])
        return thestate
        
    def CDIIS_eq( self, coeff ):
    
        return np.array( [ np.sum(coeff) - 1.0 ] )
        
    def CDIIS_ineq( self, coeff ):
    
        return coeff
        
    def CDIIS_cost( self, coeff ):
    
        errorvec = self.errors[0] * coeff[0]
        for cnt in range(1, len(self.errors)):
            errorvec += self.errors[cnt] * coeff[cnt]
        return np.linalg.norm( errorvec )**2
        
    def SolveCDIIS( self ):
    
        coeff = np.zeros([ len(self.errors) ], dtype=complex)#dtype=float)
        coeff[ len(self.errors) - 1 ] = 1.0
        result = fmin_slsqp( self.CDIIS_cost, coeff, f_eqcons=self.CDIIS_eq, f_ieqcons=self.CDIIS_ineq, iprint=0 )
        thestate = result[0] * self.states[0]
        for cnt in range(1, len(result)):
            thestate += result[cnt] * self.states[cnt]
        print("   CDIIS :: Coefficients (latest first) = ",result[::-1])
        return thestate
        
