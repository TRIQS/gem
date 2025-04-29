###########################################
#      utilities for grisb
#      Author: Samuele Giuli
#      email: samuele.giuli@gmail.com
###########################################
import numpy as np

def space_list(nbath:int):
    Nempt=0; Nfull=0; U_qp=np.eye(nbath,dtype=np.complex128)
    my_list=[Nempt, Nfull, U_qp]
    return my_list

def measure_space(my_list,delta_qp):
    Nempt, Nfull, U_qp = my_list
    nb = delta_qp.shape[0]
    delta_tot = np.conj(U_qp.T) @ delta_qp @ U_qp
    delta_aux = delta_tot[Nempt:nb-Nfull,Nempt:nb-Nfull]
    e_aux, U_aux = np.linalg.eigh(delta_aux)
    i_empt=-1; i_full=len(e_aux)
    for i,ei in enumerate(e_aux):
        if(ei<1e-10):
            i_empt=i
        elif(ei>1.0-1e-10):
            i_full=i
            break
    if(i_empt>-1 or i_full<len(e_aux)):
        U_tot = np.eye( U_qp.shape[0], dtype=np.complex128 )
        U_tot[Nempt:nb-Nfull,Nempt:nb-Nfull] = U_aux
        U_qp = U_qp @ U_tot
        Nempt += (i_empt+1)
        Nfull += (len(e_aux)-i_full)
    return [Nempt, Nfull, U_qp]
    
