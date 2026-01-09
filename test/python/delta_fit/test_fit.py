from delta_fit import solve_F_only, solve_F_dF, build_H, F_of_H
import numpy as np

import time


#TESTS START HERE
size=1
B=3
Bsize=int(B*size)
beta=100

#READING SOLUTIONS OF A B=3 Norb=1 gRISB calculation

Lambda_target = np.loadtxt("input_data/lambda.real")
Lambda_target=0.5*(Lambda_target + Lambda_target.T.conj() )
R_target = np.loadtxt("input_data/R.real").reshape((3,1))

Lambda_c = np.loadtxt("input_data/lambdac.real")
Lambda_c = 0.5*(Lambda_c + Lambda_c.T.conj() )
D = np.loadtxt("input_data/V.real").reshape((3,1))

H = build_H(Lambda_target, Lambda_c, D, R_target)
Delta_target = F_of_H(H, beta)



D11_target=Delta_target[:Bsize,:Bsize]
D12_target=Delta_target[:Bsize,Bsize:]

Lambda_0 = 2.0*(-0.5+np.random.rand(Bsize,Bsize)) + 2j*(-0.5+np.random.rand(Bsize,Bsize))
Lambda_0=0.5*(Lambda_0 + Lambda_0.T.conj() )
R_0 = 2.0*(np.random.rand(Bsize,size)-0.5 +1j*np.random.rand(Bsize,size)-0.5*1j)

Lambda_0 = Lambda_target +0.02*Lambda_0
R_0      = R_target +0.02*R_0

print("Starting from:")
print(Lambda_0)
print(R_0)


in_time=time.time()
res, Lam_sol, R_sol = solve_F_only(beta, Lambda_c, D, Lambda_0, R_0, D11_target, D12_target)
fin_time=time.time()
noder_time=fin_time-in_time

print("")
print("Without derivatives in ",noder_time,"s")
print("Distance between Lambda sol and target without derivatives:",np.linalg.norm(Lam_sol-Lambda_target))
print("Distance between R sol and target without derivatives:",np.linalg.norm(R_sol-R_target))
print("")


in_time=time.time()
res, Lam_sol, R_sol = solve_F_dF(beta, Lambda_c, D, Lambda_0, R_0, D11_target, D12_target)
fin_time=time.time()
yeder_time=fin_time-in_time

print("")
print("With derivatives in ",yeder_time,"s")
print("Distance between Lambda sol and target with derivatives:",np.linalg.norm(Lam_sol-Lambda_target))
print("Distance between R sol and target with derivatives:",np.linalg.norm(R_sol-R_target))
print("")

