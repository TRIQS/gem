import matplotlib.pyplot as plt
import numpy as np

G_nb3_ged = np.loadtxt('B3/G_U2.00_ED.dat').T
G_nb3_gcc = np.loadtxt('B3/G_U2.00_CCSD.dat').T
G_nb3_gdm = np.loadtxt('B3/G_U2.00_DMRG.dat').T

G_nb5_ged = np.loadtxt('B5/G_U2.00_ED.dat').T
G_nb5_gcc = np.loadtxt('B5/G_U2.00_CCSD.dat').T
G_nb5_gdm = np.loadtxt('B5/G_U2.00_DMRG.dat').T

G_nb7_ged = np.loadtxt('B7/G_U2.00_ED.dat').T
G_nb7_gcc = np.loadtxt('B7/G_U2.00_CCSD.dat').T
G_nb7_gdm = np.loadtxt('B7/G_U2.00_DMRG.dat').T

G_nb19_gcc = np.loadtxt('B19/G_U2.00_CCSD.dat').T
G_nb19_gdm = np.loadtxt('B19/G_U2.00_DMRG.dat').T

G_nb25_gcc = np.loadtxt('B25/G_U2.00_CCSD.dat').T
G_nb25_gdm = np.loadtxt('B25/G_U2.00_DMRG.dat').T

plt.figure(figsize=(8,6))

plt.subplot(2,2,1)
plt.title('B=3')
plt.plot(G_nb3_ged[0], -G_nb3_ged[2]/np.pi, 'b-', label='ED')
plt.plot(G_nb3_gcc[0], -G_nb3_gcc[2]/np.pi, 'r--', label='CCSD')
plt.plot(G_nb3_gdm[0], -G_nb3_gdm[2]/np.pi, 'g-.', label='DMRG M=100')
plt.xlim(-3,3)
plt.ylim(0,)
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.title('B=5')
plt.plot(G_nb5_ged[0], -G_nb5_ged[2]/np.pi, 'b-', label='ED')
plt.plot(G_nb5_gcc[0], -G_nb5_gcc[2]/np.pi, 'r--', label='CCSD')
plt.plot(G_nb5_gdm[0], -G_nb5_gdm[2]/np.pi, 'g-.', label='DMRG M=100')
plt.xlim(-3,3)
plt.ylim(0,)

#plt.subplot(2,2,3)
#plt.title('B=7')
#plt.plot(G_nb7_ged[0], -G_nb7_ged[2]/np.pi, 'b-', label='ED')
#plt.plot(G_nb7_gcc[0], -G_nb7_gcc[2]/np.pi, 'r--', label='CCSD')
#plt.plot(G_nb7_gdm[0], -G_nb7_gdm[2]/np.pi, 'g-.', label='DMRG M=100')
#plt.xlim(-3,3)
#plt.ylim(0,)

plt.subplot(2,2,3)
plt.title('B=19')
plt.plot(G_nb19_gcc[0], -G_nb19_gcc[2]/np.pi, 'r--', label='CCSD')
plt.plot(G_nb19_gdm[0], -G_nb19_gdm[2]/np.pi, 'g-.', label='DMRG M=100')
plt.xlim(-3,3)
plt.ylim(0,)

plt.subplot(2,2,4)
plt.title('B=25')
plt.plot(G_nb25_gcc[0], -G_nb25_gcc[2]/np.pi, 'r--', label='CCSD')
plt.plot(G_nb25_gdm[0], -G_nb25_gdm[2]/np.pi, 'g-.', label='DMRG M=100')
plt.xlim(-3,3)
plt.ylim(0,)

plt.tight_layout()
plt.show()
