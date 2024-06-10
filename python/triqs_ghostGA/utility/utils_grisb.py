import numpy as np
from triqs_ghostGA.utility.utils_TH import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, \
     Hermitian_list, get_blocks, funcMat, calc_nf, dF

def calc_rhoks(R, Lambda, eks, T):
    return [calc_nf( np.dot(R, np.dot(x, R.conj().T ) ) + Lambda ,T).T for x in eks]

def calc_Delta_p(rhok_list):
    return sum(rhok_list)/len(rhok_list)

def calc_D(R, Lambda, Delta_p, eks, rhoks):
    """ Compute D matrix
    """
    Left=[np.dot( np.dot(eks[x], R.conj().T ), rhoks[x].T ) for x in range(len(rhoks))]
    Left=sum(Left)/float(len(rhoks))
    Right=funcMat(Delta_p, denR)
    return np.dot(Right,np.transpose(Left))

def calc_Lambda_c(R, Lambda, Delta_p, D, H_list):
    """ Compute Lambda_c matrix
    """
    no = Lambda.shape[0]
    print("In calc_Lambda_c")
    print(Lambda)
    l=inverse_realHcombination(Lambda,H_list)
    print(l)
    lc=np.copy(l)*0.0
    MM=np.dot(D,np.transpose(R))
    print(MM)
    for k in range(len(H_list)):
        AA=Delta_p
        HH=H_list[k].T
        derivative=dF(AA,HH, denRm1, ddenRm1)
        tt=np.trace(np.dot(MM,derivative))
        lc[k]=-l[k]-(tt+np.conjugate(tt)).real
    print(tt)
    Lambda_c=realHcombination(lc,H_list)
    print(Lambda_c)
    return Lambda_c

def calc_Lambda(R, Lambda_c, Delta_p, D, H_list):
    """ Compute Lambda_c matrix
    """
    no = Lambda_c.shape[0]
    lc=inverse_realHcombination(Lambda_c,H_list)
    l=np.copy(lc)*0.0
    MM=np.dot(D,np.transpose(R))
    for k in range(len(H_list)):
        AA=Delta_p
        HH=H_list[k].T
        derivative=dF(AA,HH, denRm1, ddenRm1)
        tt=np.trace(np.dot(MM,derivative))
        l[k]=-lc[k]-(tt+np.conjugate(tt)).real
    Lambda=realHcombination(l,H_list)
    return Lambda

def cost_function(x, *args):
    ''' Cost function for find Lambda
    '''
    R, ffdagger, eks, Hspin_list, beta = args
    Lambda_spin = realHcombination(x, Hspin_list)
    Lambda = np.kron(Lambda_spin,np.eye(2))
    rhok_list=calc_rhoks(R, Lambda, eks, 1./beta)
    Delta_p=calc_Delta_p(rhok_list)
    diff = np.linalg.norm(Delta_p[::2,::2]-ffdagger.T[::2,::2])
    #diff = inverse_realHcombination(Delta_p[::2,::2]-ffdagger.T[::2,::2], Hspin_list)
    #print('diff=',diff)
    return diff

def find_Lambda(Lambda0, R, ffdagger, eks, Hspin_list, beta):
    """ Find Lambda for given ffdagger
    """
    Lambda0_spin = Lambda0[::2,::2]
    x = inverse_realHcombination(Lambda0_spin, Hspin_list)
    args = (R, ffdagger, eks, Hspin_list, beta)
    result = scipy.optimize.minimize( cost_function, x, args=args, tol=1e-12, method='L-BFGS-B', options={'eps':1e-12} )
    #result = scipy.optimize.root( cost_function, x, args=args, method='lm', options={'eps':1e-10} )
    if ( result.success==False ):
        print("   Minimize mesage ::",result.message)
    print("   Minimize :: Cost function after convergence =",result.fun)
    Lambda = np.kron(realHcombination(result.x, Hspin_list),np.eye(2))
    return Lambda

def svd_truncate_R(R, eps=0.5):
    "perform SVD truncation for the singular value of R greater than 1 and smaller than a threshold eps"

    from scipy.linalg import svd
    try:
        u, s, vh = svd(R)
    except ValueError:
        print("R")
        print(R)
        raise

    print('singular values of R:', s)
    sp = np.zeros(R.shape, dtype=s.dtype)
    for i,si in enumerate(s):
        if si > 1.0:
            sp[i,i] = 1.0
        elif si < (1.0 - eps):
            sp[i,i] = (1.0 - eps)
        else:
            sp[i,i] = si
    Rp = u @ sp @ vh
    return Rp
