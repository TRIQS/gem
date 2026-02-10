import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize
np.set_printoptions(suppress=True)

def get_Delta(nimp, oms, V, eb, eloc, mu):
    """ Generalize to multiorbital but still assume diagonal eloc
    """ 
    Delta = np.zeros((len(oms),nimp),dtype=np.complex128)
    for iom, om in enumerate(oms):
        for ii in range(nimp):
            tmp = 0.0
            for ib in range(eb.shape[0]):
                tmp += (V[ii,ib]*np.conj(V[ii,ib])).real/(om-eb[ib,ib])
            Delta[iom,ii] = tmp
    return Delta 
        
def cost_func(x, *args):
    """ Generalize to multiorbital
    """ 
    nimp, nv, ne, omFs, eloc, mu, Delta0, Nmax, Hv_list, He_list, power = args
    Vs = x[:nv] + 1j*x[nv:2*nv]
    ebs = x[2*nv:2*nv+ne]
    V = sum([v*Hv_list[i] for i,v in enumerate(Vs)])
    eb = sum([e*He_list[i] for i,e in enumerate(ebs)])
    #print('V=')
    #print(V)
    #print('eb=')
    #print(eb)
    Delta = get_Delta(nimp, 1j*omFs, V, eb, eloc, mu)
    #print(G0_aim)
    cost = 0.0
    for iomF in range(Nmax):
        for ii in range(nimp):
            diff = Delta0[iomF] - Delta[iomF,ii]
            cost += (np.conj(diff)*diff).real/omFs[iomF]**power
    return (cost/nimp).real

nimp = 2

NomF = 200
Nom = 500
beta = 200.
eloc = np.zeros((nimp,nimp))
mu = 0.0
omFs = (2*np.arange(NomF)+1)*np.pi/beta
oms = np.linspace(-5,5,Nom)
folder_path = 'B5/'
Delta0 = np.loadtxt(folder_path+'Deltaiw_U2.00.dat').T
Delta0w = np.loadtxt(folder_path+'Delta_U2.00.dat').T
#print(G0)

# new basis to be fitted
nbath = 25 #19
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
fit_method = 'SLSQP'#'SLSQP','BFGS','L-BFGS-B'
power = 0
#x0 = np.hstack((D_gauge.T[::2,::2][0].real, D_gauge.T[::2,::2][0].imag,Lamc_gauge[::2,::2].diagonal()))
x0 = np.hstack((np.ones((nv))*0.1,np.zeros((nv)),np.linspace(-5,5,ne)))
print('x0=',x0)

args = nimp//2, nv, ne, omFs, eloc[::2,::2], mu, Delta0[1]+1j*Delta0[2], Nmax, Hv_list, He_list, power
#check cost function
#print(cost_func(x0, *args))
result = minimize(cost_func,x0,args=args, method=fit_method, options={'gtol': 1e-5, 'eps': 1e-8, 'maxiter': 500})
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

Delta0_fit = get_Delta(nimp//2, 1j*omFs, V, eb, eloc[::2,::2], mu)
Delta0w_fit = get_Delta(nimp//2, oms+1j*0.05, V, eb, eloc[::2,::2], mu)

fh5o = h5py.File('D_Lamc_fit_hyb_B%d.h5'%(nbath),'w')
fh5o['D'] = V.conj().T
fh5o['Lambda_c'] = eb
fh5o.close()

plt.plot(Delta0[0], Delta0[2], 'b-', label='imag')
plt.plot(Delta0[0], Delta0[1], 'r-', label='real')
plt.plot(omFs, Delta0_fit.imag, 'bo', label='imag fit')
plt.plot(omFs, Delta0_fit.real, 'rv', label='real fit')
plt.show()

plt.plot(Delta0w[0], -Delta0w[2], 'b-', label='imag')
#plt.plot(oms, Delta0w.real, 'r-', label='real')
plt.plot(oms, -Delta0w_fit.imag, 'bo', label='imag fit')
#plt.plot(oms, Delta0w_fit.real, 'rv', label='real fit')
plt.show()
