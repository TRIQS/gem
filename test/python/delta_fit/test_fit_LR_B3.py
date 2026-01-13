from delta_fit import *
import numpy as np

import time


#TESTS START HERE
size=1
B=3
Bsize=int(B*size)
beta=100
noise=0.2
fold_data=f"input_data/B3"

#READING SOLUTIONS OF A B=3 Norb=1 gRISB calculation

Lambda_target = np.loadtxt(f"{fold_data}/lambda.real")
Lambda_target=0.5*(Lambda_target + Lambda_target.T.conj() )
R_target = np.loadtxt(f"{fold_data}/R.real").reshape((3,1))
x_target = pack_params(Lambda_target, R_target)


Lambda_c = np.loadtxt(f"{fold_data}/lambdac.real")
Lambda_c = 0.5*(Lambda_c + Lambda_c.T.conj() )
D = np.loadtxt(f"{fold_data}/V.real").reshape((3,1))

H = build_H(Lambda_target, Lambda_c, D, R_target)
Delta_target = F_of_H(H, beta)


D11_target=Delta_target[:Bsize,:Bsize]
D22_target=Delta_target[Bsize:,Bsize:]
D12_target=Delta_target[:Bsize,Bsize:]
RTD12_target = R_target.T@D12_target

print("D11_target:",D11_target.real)
print("D22_target:",D22_target.real)

#in principle zero:
residual0 = residual_LR(x_target, beta, Lambda_c, D, D22_target, RTD12_target)
jacobian0 = jacobian_LR(x_target, beta, Lambda_c, D, D22_target, RTD12_target)
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
start_residual = residual_LR(start_x, beta, Lambda_c, D, D22_target, RTD12_target)
print("Starting residual:",np.sum(np.abs(start_residual)))


print(" --- TESTING ROOT WITHOUT DERIVATIVES ---")
in_time=time.time()
Lam_sol, R_sol = new_self_energy( Lambda_0,R_0, Lambda_c,D, D22_target,RTD12_target, beta=beta, method="F")
fin_time=time.time()
noder_time=fin_time-in_time
x_fonly=pack_params(Lam_sol,R_sol)
Fonly_residual=residual_LR(x_fonly,beta,Lambda_c,D,D22_target,RTD12_target)
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
Lam_sol, R_sol = new_self_energy( Lambda_0,R_0, Lambda_c,D, D22_target,RTD12_target, beta=beta, method="dF")
fin_time=time.time()
yeder_time=fin_time-in_time
x_fdf=pack_params(Lam_sol,R_sol)
Fdf_residual=residual_LR(x_fdf,beta,Lambda_c,D,D22_target,RTD12_target)
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
