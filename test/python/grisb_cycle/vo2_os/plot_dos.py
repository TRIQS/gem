import numpy as np
import matplotlib.pyplot as plt
import triqs.utility.mpi as mpi
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from triqs.gf.tools import inverse
from triqs_ghostGA.sumk_grisb import SumkGRISB
from triqs_ghostGA.utility.utils_TH import calc_nf
from triqs.plot.mpl_interface import oplot
from h5 import HDFArchive
from copy import deepcopy
import time
np.set_printoptions(suppress=True,precision=6)

t_start = time.time()

beta = 200.

# first we have to determine the mesh
sumk_mesh = MeshReFreq(window=[-20,20], n_w=2000)

sumk = SumkGRISB(hdf_file='vo2.h5',
                mesh=sumk_mesh, use_dft_blocks=False, beta=beta, h_field=0.0, nbaths=[5,5])

Gloc = sumk.extract_G_loc(broadening=0.1)

mesh = Gloc[0].mesh
mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))

plt.plot(mesh_values,-Gloc[0]['up'].data[:,0,0].imag/np.pi,'b-',label='V1_dz2')
plt.plot(mesh_values,-Gloc[0]['up'].data[:,1,1].imag/np.pi,'r-',label='V1_dxz')
plt.plot(mesh_values,-Gloc[0]['up'].data[:,2,2].imag/np.pi,'g-',label='V1_dyz')
plt.plot(mesh_values,-Gloc[0]['up'].data[:,3,3].imag/np.pi,'c-',label='V1_dx2-y2')
plt.plot(mesh_values,-Gloc[0]['up'].data[:,4,4].imag/np.pi,'m-',label='V1_dxy')
plt.plot(mesh_values,-Gloc[1]['up'].data[:,0,0].imag/np.pi,'b--',label='V2_dz2')
plt.plot(mesh_values,-Gloc[1]['up'].data[:,1,1].imag/np.pi,'r--',label='V2_dxz')
plt.plot(mesh_values,-Gloc[1]['up'].data[:,2,2].imag/np.pi,'g--',label='V2_dyz')
plt.plot(mesh_values,-Gloc[1]['up'].data[:,3,3].imag/np.pi,'c--',label='V2_dx2-y2')
plt.plot(mesh_values,-Gloc[1]['up'].data[:,4,4].imag/np.pi,'m--',label='V2_dxy')
plt.axvline(0)
plt.xlim(-10,10)
plt.legend()
plt.show()

