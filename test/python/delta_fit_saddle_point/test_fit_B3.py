from delta_fit import *
import numpy as np
import h5py
import time


#TESTS START HERE
size=1
B=3
Bsize=int(B*size)
beta=500
noise=0.0#0.2
fold_data=f"input_data/"

#READING SOLUTIONS OF A B=3 Norb=1 gRISB calculation

fh5 = h5py.File(fold_data+'sols.h5','r')
Lambda_target = fh5['U1.50/Lambda'][...][::2,::2]
R_target = fh5['U1.50/R'][...][::2,::2]
x_target = pack_params(Lambda_target, R_target)


Lambda_c = fh5['U1.50/Lambdac'][...][::2,::2]
D = fh5['U1.50/D'][...][::2,::2]

Delta_target = fh5['U1.50/denMat'][...][::2,::2]
fh5.close()

D11_target=Delta_target[:1,:1]
D22_target=Delta_target[1:,1:]
D12_target=Delta_target[:1,1:]
print("D11_target:",D11_target.real)
print("D22_target:",D22_target.real)

#in principle zero:
residual0 = residual(x_target, beta, Lambda_c, D, D22_target, D12_target)
jacobian0 = jacobian(x_target, beta, Lambda_c, D, D22_target, D12_target)
tot_res0 = np.sum(np.abs(residual0))
if(tot_res0>1e-10):
    raise ValueError(f"The residual of the starting point should be zero while it is:{tot_res0}")

Lambda_0 = 2.0*(-0.5+np.random.rand(Bsize,Bsize)) + 2j*(-0.5+np.random.rand(Bsize,Bsize))
Lambda_0=0.5*(Lambda_0 + Lambda_0.T.conj() )
R_0 = 2.0*(np.random.rand(Bsize,size)-0.5 +1j*np.random.rand(Bsize,size)-0.5*1j)

Lambda_0 = Lambda_target +noise*Lambda_0
R_0      = R_target +noise*R_0

print("Starting from:")
print(Lambda_0)
print(R_0)
start_x = pack_params(Lambda_0, R_0)
start_residual = residual(start_x, beta, Lambda_c, D, D22_target, D12_target)
print("Starting residual:",np.sum(np.abs(start_residual)))


print(" --- TESTING ROOT WITHOUT DERIVATIVES ---")
in_time=time.time()
res, Lam_sol, R_sol = solve_F_only(beta, Lambda_c, D, Lambda_0, R_0, D22_target, D12_target)
fin_time=time.time()
noder_time=fin_time-in_time
x_fonly=pack_params(Lam_sol,R_sol)
Fonly_residual=residual(x_fonly,beta,Lambda_c,D,D22_target,D12_target)
print("F only residual:",np.sum(np.abs(Fonly_residual)))

H_sol = build_H(Lam_sol, Lambda_c, D, R_sol)
Delta_sol = F_of_H(H_sol, beta)
D11_sol=Delta_sol[:Bsize,:Bsize]
D12_sol=Delta_sol[:Bsize,Bsize:]
D22_sol=Delta_sol[Bsize:,Bsize:]

print("Final flling:",np.sum(np.diag(Delta_sol)))
print("")
print("Without derivatives in ",noder_time,"s")
print("Distance in R :", np.sum(np.abs(R_target-R_sol)))
print("Distance in Lambda:", np.sum(np.abs(Lambda_target-Lam_sol)))
Lg_trg, Ut = np.linalg.eigh(Lambda_target)
Lg_sol, Us = np.linalg.eigh(Lam_sol)
Rg_trg=np.abs(Ut.T.conj()@R_target)
Rg_sol=np.abs(Us.T.conj()@R_sol)
print("Distance in gauge invariant R:",np.sum(np.abs(Rg_trg-Rg_sol)))
print("Distance in gauge invariant Lambda:", np.sum(np.abs(Lg_sol-Lg_trg)))
noder_error = np.sum(np.abs(Rg_trg-Rg_sol))+np.sum(np.abs(Lg_sol-Lg_trg))
print("")




print(" --- TESTING ROOT WITH DERIVATIVES FITTING <bdagb> ---")
in_time=time.time()
res, Lam_sol, R_sol = solve_F_dF(beta, Lambda_c, D, Lambda_0, R_0, D22_target, D12_target)
fin_time=time.time()
yeder_time=fin_time-in_time
x_fdf=pack_params(Lam_sol,R_sol)
Fdf_residual=residual(x_fdf,beta,Lambda_c,D,D22_target,D12_target)
print("F dF residual:",np.sum(np.abs(Fdf_residual)))

H_sol = build_H(Lam_sol, Lambda_c, D, R_sol)
Delta_sol = F_of_H(H_sol, beta)
D11_sol = Delta_sol[:Bsize,:Bsize]
D12_sol = Delta_sol[:Bsize,Bsize:]
D22_sol = Delta_sol[Bsize:,Bsize:]

#print(np.diag(Delta_sol))
print("Final filling:",np.sum(np.diag(Delta_sol)))
print("")
print("With derivatives in ",yeder_time,"s")
print("Distance in R :", np.sum(np.abs(R_target-R_sol)))
print("Distance in Lambda:", np.sum(np.abs(Lambda_target-Lam_sol)))
Lg_trg, Ut = np.linalg.eigh(Lambda_target)
Lg_sol, Us = np.linalg.eigh(Lam_sol)
Rg_trg=np.abs(Ut.T.conj()@R_target)
Rg_sol=np.abs(Us.T.conj()@R_sol)
print("Distance in gauge invariant R:",np.sum(np.abs(Rg_trg-Rg_sol)))
print("Distance in gauge invariant Lambda:", np.sum(np.abs(Lg_sol-Lg_trg)))
yeder_error = np.sum(np.abs(Rg_trg-Rg_sol))+np.sum(np.abs(Lg_sol-Lg_trg))
print("")
print(" --- OVERALL ---")
print("Time without derivatives:",noder_time,"s - and error:",noder_error)
print("Time with derivatives:",yeder_time,"s - and error:",yeder_error)
