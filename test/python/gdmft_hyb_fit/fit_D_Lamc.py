import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize
np.set_printoptions(suppress=True)

def get_G0_aim(nimp, oms, V, eb, eloc, mu):
    """ Generalize to multiorbital but still assume diagonal eloc
    """ 
    G0_aim = np.zeros((len(oms),nimp),dtype=np.complex128)
    for iom, om in enumerate(oms):
        for ii in range(nimp):
            Delta = 0.0
            for ib in range(eb.shape[0]):
                Delta += (V[ii,ib]*np.conj(V[ii,ib])).real/(om-eb[ib,ib])
            G0_aim[iom,ii] = 1./(om - eloc[ii,ii] + mu - Delta)
    return G0_aim 
        
def get_hyb_aim(nimp, oms, V, eb, eloc, mu):
    """ Generalize to multiorbital but still assume diagonal eloc
    """ 
    hyb_aim = np.zeros((len(oms),nimp),dtype=np.complex128)
    for iom, om in enumerate(oms):
        for ii in range(nimp):
            Delta = 0.0 
            for ib in range(eb.shape[0]):
                Delta += (V[ii,ib]*np.conj(V[ii,ib])).real/(om-eb[ib,ib])
            hyb_aim[iom,ii] = Delta
    return hyb_aim
    
        
def cost_func(x, *args):
    """ Generalize to multiorbital
    """ 
    nimp, nv, ne, omFs, eloc, mu, G0, Nmax, Hv_list, He_list, fit_scheme, power = args
    Vs = x[:nv] + 1j*x[nv:2*nv]
    ebs = x[2*nv:2*nv+ne]
    V = sum([v*Hv_list[i] for i,v in enumerate(Vs)])
    eb = sum([e*He_list[i] for i,e in enumerate(ebs)])
    #print('V=')
    #print(V)
    #print('eb=')
    #print(eb)
    G0_aim = get_G0_aim(nimp, 1j*omFs, V, eb, eloc, mu)
    #print(G0_aim)
    cost = 0.0
    for iomF in range(Nmax):
        for ii in range(nimp):
            if fit_scheme == 'hyb':
                diff = 1./G0_aim[iomF,ii] - 1./G0[iomF,ii]
            elif fit_scheme == 'G0':
                diff = G0_aim[iomF,ii] - G0[iomF,ii]
            cost += (np.conj(diff)*diff).real/omFs[iomF]**power
    return (cost/nimp).real

fh5 = h5py.File('B7/sols.h5','r')
R = fh5['R'][...]
Lam = fh5['Lambda'][...]
D = fh5['D'][...]
Lamc = fh5['Lambda_c'][...]
fh5.close()

assert(np.allclose(Lamc[::2,::2], Lamc[1::2,1::2]))
evals, u = np.linalg.eigh(Lamc[::2,::2])
u = np.kron(u,np.eye(2))
Lamc_gauge = u.conj().T @ Lamc @ u
D_gauge = u.T @ D

print('D_gauge=')
print(D_gauge)
print('Lambda_c original=')
print(Lamc)
print('Lambda_c diagonal gauge=')
print(Lamc_gauge)
print()

nimp = 2

# old basis
#Hv_list = [np.array([[1,0,0]]),
#           np.array([[0,1,0]]),
#           np.array([[0,0,1]]),
#          ]
#He_list = [np.array([[1,0,0],
#                      [0,0,0],
#                      [0,0,0]]),
#           np.array([[0,0,0],
#                      [0,1,0],
#                      [0,0,0]]),
#           np.array([[0,0,0],
#                      [0,0,0],
#                      [0,0,1]]),
#          ]
NomF = 50
Nom = 500
beta = 50.
eloc = np.zeros((nimp,nimp))
mu = 0.0
omFs = (2*np.arange(NomF)+1)*np.pi/beta
oms = np.linspace(-5,5,Nom)
G0 = get_G0_aim(nimp//2, 1j*omFs, D_gauge.conj().T, Lamc_gauge, eloc[::2,::2], mu)
G0w = get_G0_aim(nimp//2, oms+1j*0.05, D_gauge.conj().T, Lamc_gauge, eloc[::2,::2], mu)
#print(G0)

# new basis to be fitted
#Hv_list = [np.array([[1,0,0,0,0]]),
#           np.array([[0,1,0,0,0]]),
#           np.array([[0,0,1,0,0]]),
#           np.array([[0,0,0,1,0]]),
#           np.array([[0,0,0,0,1]]),
#          ]
#He_list = [np.array([[1,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0]]),
#           np.array([[0,0,0,0,0],
#                        [0,1,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0]]),
#           np.array([[0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,1,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0]]),
#           np.array([[0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,1,0],
#                        [0,0,0,0,0]]),
#           np.array([[0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,1]]),
#          ]
nbath = 19
Hv_list = []
for i in range(nbath):
    tmp = np.zeros((1,nbath))
    tmp[0,i] = 1.0
    Hv_list.append(tmp)
He_list = []
for i in range(nbath):
    tmp = np.zeros((nbath,nbath))
    tmp[i,i] = 1.0
    He_list.append(tmp)
ne = len(He_list)
nv = len(Hv_list)

Nmax = NomF
fit_method = 'SLSQP'#'BFGS'#'L-BFGS-B'
fit_scheme = 'hyb'#'G0'
power = 0
#x0 = np.hstack((D_gauge.T[::2,::2][0].real, D_gauge.T[::2,::2][0].imag,Lamc_gauge[::2,::2].diagonal()))
x0 = np.hstack((np.ones((nv))*0.1,np.zeros((nv)),np.linspace(-3,3,ne)))
print('x0=',x0)

args = nimp//2, nv, ne, omFs, eloc[::2,::2], mu, G0, Nmax, Hv_list, He_list, fit_scheme, power
#check cost function
#print(cost_func(x0, *args))
result = minimize(cost_func,x0,args=args, method=fit_method, options={'gtol': 1e-5, 'eps': 1e-8})
#result = minimize(cost_func,x0,args=args, method='CG', options={'gtol': 1e-5, 'eps': 1e-8})#, 'maxiter': 500})
#result = minimize(cost_func,x0,args=args, method='L-BFGS-B', options={'gtol': 1e-5, 'eps': 1e-8})
#result = minimize(cost_func,x0,args=args, method='BFGS', options={'gtol': 1e-5, 'eps': 1e-8})
print("GA root convergence message---------------------------------")
print("sucess=",result.success)
print(result.message)
print("f(x)=",result.fun)
Vs = result.x[:nv] + 1j*result.x[nv:2*nv]
ebs = result.x[2*nv:2*nv+ne]
V = sum([v*Hv_list[i] for i,v in enumerate(Vs)])
eb = sum([e*He_list[i] for i,e in enumerate(ebs)])
x0 = result.x
print('V=',Vs)
print('eb=', ebs)

G0_fit = get_G0_aim(nimp//2, 1j*omFs, V, eb, eloc[::2,::2], mu)
G0w_fit = get_G0_aim(nimp//2, oms+1j*0.05, V, eb, eloc[::2,::2], mu)

fh5o = h5py.File('D_Lamc_fit_B%d.h5'%(nbath),'w')
fh5o['D'] = V.conj().T
fh5o['Lambda_c'] = eb
fh5o.close()

plt.plot(omFs, G0.imag, 'b-', label='imag')
plt.plot(omFs, G0.real, 'r-', label='real')
plt.plot(omFs, G0_fit.imag, 'bo', label='imag fit')
plt.plot(omFs, G0_fit.real, 'rv', label='real fit')
plt.show()

plt.plot(oms, -G0w.imag, 'b-', label='imag')
#plt.plot(oms, G0w.real, 'r-', label='real')
plt.plot(oms, -G0w_fit.imag, 'bo', label='imag fit')
#plt.plot(oms, G0w_fit.real, 'rv', label='real fit')
plt.show()
