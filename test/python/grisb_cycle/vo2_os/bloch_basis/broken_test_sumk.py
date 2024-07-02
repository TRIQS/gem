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
#sumk_mesh = MeshReFreq(window=[-20,20], n_w=2000)
sumk_mesh = None

sumk = SumkGRISB(hdf_file='quantum_espresso_files/vo2.h5',
                mesh=sumk_mesh, use_dft_blocks=False, beta=beta, h_field=0.0, nbaths=5)
sumk.chemical_potential = 11.209421#11.2631
icrsh = 0
eloc = [{}]
eloc_full = {}
enloc_full = {}
denmat = {}
denmat_test = {}
for sp, isp in sumk.spin_names_to_ind[sumk.SO].items():
    ind = sumk.spin_names_to_ind[
                        sumk.corr_shells[icrsh]['SO']][sp]
    eloc[icrsh][sp] = np.zeros((5,5),dtype=complex)
    eloc_full[sp] = np.zeros((8,8),dtype=complex)
    enloc_full[sp] = np.zeros((8,8),dtype=complex)
    denmat[sp] = np.zeros((22,22),dtype=complex)
    denmat_test[sp] = np.zeros((22,22),dtype=complex)
    for ik in range(sumk.n_k):
        eloc[icrsh][sp] += sumk.bz_weights[ik] * sumk.rot_mat[icrsh].conjugate().transpose().dot(sumk.hopping[ik,ind,:5,:5]).dot(sumk.rot_mat[icrsh])
        eloc_full[sp] += sumk.bz_weights[ik] * sumk.hopping[ik,ind,:8,:8]
        enloc_full[sp] += sumk.bz_weights[ik] * sumk.hopping_nloc[ik,ind,:8,:8]
        denmat[sp] += sumk.bz_weights[ik] * calc_nf(sumk.hopping[ik,ind,:,:]-sumk.chemical_potential*np.eye(22), 1/beta).T
        #tmp = sumk.hopping_nloc[ik,isp,:,:].copy()
        #tmp[0:5,0:5] += sumk.Hsumk[0][sp][:,:]
        #tmp[5:10,5:10] += sumk.Hsumk[1][sp][:,:]
        #denmat_test[sp] += sumk.bz_weights[ik] * calc_nf(tmp-sumk.chemical_potential*np.eye(22), 1/beta).T

assert(np.allclose(sumk.eloc_orig[0]['up'],sumk.Hsumk[0]['up']))
assert(np.allclose(sumk.eloc_orig[0]['down'],sumk.Hsumk[0]['down']))
assert(np.allclose(sumk.eloc_orig[1]['up'],sumk.Hsumk[1]['up']))
assert(np.allclose(sumk.eloc_orig[1]['down'],sumk.Hsumk[1]['down']))

print('eloc=')
print(eloc)
print('Hsumk=')
print(sumk.Hsumk)
print('eloc_full=')
print(eloc_full)
print('enloc_full=')
print(enloc_full)
print('denmat=')
print(denmat)
#print('denmat_test=')
#print(denmat_test)
#quit()

mu = sumk.calc_mu(precision=0.001,beta=beta)
dm_test = sumk.density_matrix(method='using_gf')
#print(sumk.gf_struct_sumk)
#print(sumk.gf_struct_sumk)
#print(sumk.hopping.shape)
if mpi.is_master_node():
    print('mu=',mu)
    print('dm_test=')
    print(dm_test)
#quit()

#sumk.eff_atomic_levels()
#print('Hsumk=')
#print(sumk.Hsumk)
#print('rot_mat=')
#print(sumk.rot_mat)
R = [{"up":np.eye(5,dtype=complex),"down":np.eye(5,dtype=complex)},{"up":np.eye(5,dtype=complex),"down":np.eye(5,dtype=complex)}]
Lambda = sumk.Hsumk#np.zeros((3,3),dtype=complex) - mu*np.eye(3)
if mpi.is_master_node():
    print('Hsumk=')
    print(sumk.Hsumk)
    print('R=')
    print(R)
    print('Lambda=')
    print(Lambda)
T = 1/beta
sumk.calc_rhoks( R,Lambda,T)
#print(sumk.rhoks['up'][0,:,:])
#print(sumk.rhoks['down'][0,:,:])
sumk.calc_Delta()
if mpi.is_master_node():
    print(sumk.Delta[0]['up'])
    print(sumk.Delta[0]['down'])
    print(sumk.Delta[1]['up'])
    print(sumk.Delta[1]['down'])

#dm_full = np.zeros((22,22),dtype=complex)
#for ik in range(125):
#    dm_full += sumk.rhoks_full['up'][ik,:,:]
#dm_full/=125
#print('dm_full=')
#print(dm_full)
#print(np.dot( np.dot(np.conjugate(sumk.rot_mat[icrsh]).transpose(), sumk.Delta[icrsh]['up']), sumk.rot_mat[icrsh] ) )
#print(np.dot( np.dot(np.conjugate(sumk.rot_mat[icrsh]).transpose(), sumk.Delta[icrsh]['down']), sumk.rot_mat[icrsh] ) )
sumk.calc_D(R, Lambda)
print(sumk.D[0]['up'])
print(sumk.D[0]['down'])
print(sumk.D[1]['up'])
print(sumk.D[1]['down'])

##ikarray = np.array(list(range(sumk.n_k)))
##print(len(sumk.spin_names_to_ind[1]))#sumk.SO])
##
##for icrsh in range(sumk.n_corr_shells):
##    dim = sumk.corr_shells[icrsh]['dim']
##    print('icrsh=',icrsh, 'dim=',dim)
##    for sp, isp in sumk.spin_names_to_ind[sumk.SO].items():
##        for ik in mpi.slice_array(ikarray):
##            print('ik=', ik, 'sp=', sp, 'isp=', isp)
##            print(sumk.hopping[ik,isp])
###    G_latt_w = sumk.lattice_gf(ik=ik, mu=sumk.chemical_potential)
###    print(G_latt_w["up"].data[:,:])
###    print(G_latt_w["down"].data[:,:])
#
#t_end = time.time()
#t_elapse = t_end - t_start
#
#print('time elapsed=', t_elapse)
