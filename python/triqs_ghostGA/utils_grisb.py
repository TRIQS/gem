import numpy as np
import scipy.optimize as scop
from triqs_ghostGA.utils_TH import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, \
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
    l=inverse_realHcombination(Lambda,H_list)
    lc=np.copy(l)*0.0
    MM=np.dot(D,np.transpose(R))
    for k in range(len(H_list)):
        AA=Delta_p
        HH=H_list[k].T
        derivative=dF(AA,HH, denRm1, ddenRm1)
        tt=np.trace(np.dot(MM,derivative))
        lc[k]=-l[k]-(tt+np.conjugate(tt)).real
    Lambda_c=realHcombination(lc,H_list)
    return Lambda_c

def calc_Lambda(R, Lambda_c, Delta_p, D, H_list):
    """ Compute Lambda_c matrix
    """
    print("In calc_Lambda")
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

# Routines to fit R and Lambda from the second embedding problem
def fit_R_and_Lambda(R, Lambda, D, Lambda_c, Dbath_aim, Dhyb_aim):
    """ Fit R and Lambda 
    """
    params = RL_to_params(R,Lambda)
    nbath,nimp = R.shape
    args   = (D, Lambda_c, Dbath_aim, Dhyb_aim, nbath,nimp)
    print("Initial cost function =",cost_function_R_and_Lambda(params, *args))
    result = scop.minimize( cost_function_R_and_Lambda, params, args=args, tol=1e-12 ) # method='L-BFGS-B',  options={'eps':1e-12}
    R, Lambda = RL_from_params(result.x,nbath,nimp)
    print("converged after",result.nit,"iterations")
    print("cost function after convergence =",result.fun)
    return R, Lambda

def RL_to_params(R, Lambda):
    """ Convert R and Lambda to a 1D array
    """
    return np.concatenate((R.real.flatten(), R.imag.flatten(), flatten_hermitian(Lambda)))

def RL_from_params(params,nbath,nimp):
    """ Convert a 1D array to R and Lambda
    """
    R = params[:nbath*nimp].reshape((nbath,nimp)) + 1j*params[nbath*nimp:2*nbath*nimp].reshape((nbath,nimp))
    Lambda = unflatten_hermitian(params[2*nbath*nimp:])
    return R, Lambda

def cost_function_R_and_Lambda(params, *args):
    """ Cost function for fitting R and Lambda
    """
    D, Lambda_c, Dbath_aim, Dhyb_aim, nbath,nimp = args
    R, Lambda = RL_from_params(params,nbath,nimp)
    H0emb = np.zeros((2*nbath,2*nbath),dtype=np.float64)
    H0emb[:nbath,:nbath] = Lambda.real
    H0emb[nbath:,nbath:] = -Lambda_c.real
    H0emb[:nbath,nbath:] = (R@D.T).real
    H0emb[nbath:,:nbath] = (np.conjugate(R@(D.T)).T).real
    evals, evecs = np.linalg.eigh(H0emb)
    onebdm = np.zeros((2*nbath,2*nbath))
    for i in range(2*nbath):
        weight  = fermi_dist(1000*evals[i])
        onebdm += weight*np.outer(evecs[:,i],np.conjugate(evecs[:,i]))
    Delta1 = np.eye(nbath)-onebdm[nbath:,nbath:]-Dbath_aim
    Delta2 = (onebdm[:nbath,nbath:] @ R).T - Dhyb_aim
    cost_RnL = np.linalg.norm(Delta1)+np.linalg.norm(Delta2)
    return cost_RnL


# Routines to fit D and Lambda from the second embedding problem
def fit_D_and_Lambda_c(D, Lambda_c, R, Lambda, Dbath_aim, Dhyb_aim):
    """ Fit D and Lambda _c
    """
    params = DL_to_params(D,Lambda_c)
    nbath,nimp = R.shape
    args   = (R, Lambda, Dbath_aim, Dhyb_aim, nbath,nimp)
    print("Initial cost function =",cost_function_D_and_Lambda_c(params, *args))
    result = scop.minimize( cost_function_D_and_Lambda_c, params, args=args, tol=1e-12 ) # method='L-BFGS-B',  options={'eps':1e-12}
    D, Lambda_c = DL_from_params(result.x,nbath,nimp)
    print("converged after",result.nit,"iterations")
    print("cost function after convergence =",result.fun)
    return D, Lambda_c


def DL_to_params(D, Lambda_c):
    """ Convert D and Lambda to a 1D array
    """
    return np.concatenate((D.real.flatten(), D.imag.flatten(), flatten_hermitian(Lambda_c)))

def DL_from_params(params,nbath,nimp):
    """ Convert a 1D array to D and Lambda
    """
    D = params[:nbath*nimp].reshape((nbath,nimp)) + 1j*params[nbath*nimp:2*nbath*nimp].reshape((nbath,nimp))
    Lambda_c = unflatten_hermitian(params[2*nbath*nimp:])
    return D, Lambda_c

#@jit(nopython=True)
def cost_function_D_and_Lambda_c(params, *args):
    """ Cost function for fitting D and Lambda
    """
    R, Lambda, Dbath_aim, Dhyb_aim, nbath,nimp = args
    D, Lambda_c = DL_from_params(params,nbath,nimp)
    H0emb = np.zeros((2*nbath,2*nbath),dtype=np.float64)
    H0emb[:nbath,:nbath] = Lambda.real
    H0emb[nbath:,nbath:] = -Lambda_c.real
    H0emb[:nbath,nbath:] = (R@D.T).real
    H0emb[nbath:,:nbath] = (np.conjugate(R@(D.T)).T).real
    evals, evecs = np.linalg.eigh(H0emb)
    onebdm = np.zeros((2*nbath,2*nbath))
    for i in range(2*nbath):
        weight  = fermi_dist(1000*evals[i])
        onebdm += weight*np.outer(evecs[:,i],np.conjugate(evecs[:,i]))
    Delta1 = np.eye(nbath)-onebdm[nbath:,nbath:]-Dbath_aim
    Delta2 = (onebdm[:nbath,nbath:] @ R).T - Dhyb_aim
    cost_DnL = np.linalg.norm(Delta1)+np.linalg.norm(Delta2)
    return cost_DnL

def get_new_R(D,Delta_p,rhok_list,eks):
    squareD=funcMat(Delta_p, denR)
    B = (squareD @ D.T).flatten
    N_i, N_j = eks[0].shape
    N_I, N_J = rhok_list[0].shape
    # indexes i,j,J,I
    Right= sum([ np.multiply.outer(eks[i],rhok_list[i]) for i in range(len(eks))])/len(rhok_list)
    # i,j,I,J
    Right = np.swapaxes(Right,aixs1=2,axis2=3)
    # i,I,j,J
    Right = np.swapaxes(Right,aixs1=1,axis2=2)
    A = np.reshape(Right,newshape=(N_i*N_I,N_j*N_J))
    Rdagflat = np.linalg.solve(A,B)
    new_Rdag = np.reshape(Rdagflat,newshape=(N_j,N_J))
    return np.conjugate(new_Rdag.T)



#@jit(nopython=True)
def fermi_dist(x):
    """ Fermi-Dirac distribution for x=beta*E
    """
    return 1.0/(1.0+np.exp(x))


def flatten_hermitian(H):
    """ routine to flatten an hermitian matrix """
    Hsize=H.shape[0]
    if( np.any(H.shape!=(Hsize,Hsize))): raise ValueError("Passing wrong shape array to flatten_hermitian")
    H2flat=np.triu(H.real,k=0)+np.tril(H.imag,k=-1)
    return H2flat.flatten()

def unflatten_hermitian(Hflat):
    """ routine to unflatten an hermitian matrix """
    Hsize=int(np.sqrt(len(Hflat)))
    if( len(Hflat)!=Hsize**2 ): raise ValueError("Passing wrong number of parameters to unflatten_hermitian")
    H2flat=Hflat.reshape((Hsize,Hsize))
    H = np.triu(H2flat)+1j*np.tril(H2flat,k=-1)
    H = H+np.conjugate(H.T)-np.diag(np.diag(H))
    return H
