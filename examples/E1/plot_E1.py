import numpy as np
import matplotlib.pyplot as plt

T=np.loadtxt('Tlist.dat')
docc=np.loadtxt('docclist.dat',dtype=np.complex128)
E=np.loadtxt('Elist.dat')
S=np.loadtxt('Slist.dat')

cp='blue'
cl='red'

plt.figure(figsize=(4,4))
plt.plot(T[1:],E[1:],marker='.',color=cp,label='E(T>0)')
plt.axhline(E[0],color=cl,linestyle=':',label='E(T=0)')
plt.xlabel('T/D')
plt.ylabel('E(T)/D')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('Eplot.svg')
plt.show()


plt.figure(figsize=(4,4))
plt.plot(T[1:],S[1:],marker='.',color=cp,label='S(T>0)')
plt.axhline(S[0],color=cl,linestyle=':',label='S(T=0)')
plt.xlabel('T/D')
plt.ylabel('S(T)')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('Splot.svg')
plt.show()


plt.figure(figsize=(4,4))
plt.plot(T[1:],docc[1:],marker='.',color=cp,label='docc(T>0)')
plt.axhline(docc[0],color=cl,linestyle=':',label='docc(T=0)')
plt.xlabel('T/D')
plt.ylabel(r'$\langle n_{\uparrow} n_{\downarrow}\rangle$')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('doccplot.svg')
plt.show()





