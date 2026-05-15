import numpy as np

from triqs_ghostGA.mps import *
import triqs.operators.util
np.set_printoptions(suppress=True, precision=6)

def promote_to_spinful(up,dn):
    assert up.shape == dn.shape
    full=np.zeros((up.shape[0]*2,up.shape[1]*2),dtype=up.dtype)
    full[::2,::2]=up
    full[1::2,1::2]=dn
    return full


if __name__ == "__main__":
    do_ci_calc=False
    import h5py
    # fh5 = h5py.File('data/hemb_test_1orb3bath.h5')
    #fh5 = h5py.File('data/hemb_test.h5')
    #D = fh5['D'][...]
    #Lambda_c = fh5['Lambda_c'][...]
    #Utensor = fh5['Utensor'][...]
    #eloc = fh5['eloc'][...]
    #mu = fh5['mu'][...]
    #fh5.close()
    Nimp=3
    Nbath=3*Nimp
    N=Nimp+Nbath
    U=3.0
    J=0.4
    #Build quartic part
    Utype="Slater"    #"Kanamori" or "Slater"
    if Utype=="Kanamori":
        try:
            Utensor=triqs.operators.util.U_matrix_kanamori(Nimp,U_int=U,J_hund=J,full_Uijkl=True)
        except:
            Utensor=U_matrix_kanamori(Nimp,U_int=U,J_hund=J,full_Uijkl=True)
        finally:
            print("Kanamori UTensor construction breaks. The 4-index functionality is only present in triqs version >= 3.2.")
        #Utensor=U_matrix_kanamori(Nimp,U_int=U,J_hund=)
        #Um,Ump=triqs.operators.util.U_matrix_kanamori(Nimp,U_int=U,J_hund=J)
    elif Utype=="Slater":
        Utensor=triqs.operators.util.U_matrix_slater(Nimp//2,U_int=U,J_hund=J)

    
    #Buid quadratic part 
    np.random.seed(1234)
    eloc_spinless=np.random.rand(Nimp,Nimp)
    eloc_spinless+=0.5*eloc_spinless.conjugate().T
    eloc=promote_to_spinful(eloc_spinless,eloc_spinless)
    
    W=3.0
    bathenergies=np.sort(2*W*(np.random.rand(Nbath)-0.5))
    Lambda_spinless=np.diag(bathenergies)
    Lambda_c=promote_to_spinful(Lambda_spinless,Lambda_spinless)

    D_spinless=np.random.rand(Nbath,Nimp)
    D=promote_to_spinful(D_spinless,D_spinless)
    
    ntot=2*N
    nimp=2*Nimp
    nbath=2*Nbath
    solver = ITensorMPSSolver(ntot, nimp, nbath)
    solver.set_kwargs({"use_Sz":True,"use_Ntot":True,"spin_pen":1.0})
    solver.make_schedule()  ##default schedule, probably overkill for 3-orbital model
    solver.set_tolerances(("E","rho"),(1e-5,5e-3))
    solver.build_Hemb(D, eloc, Lambda_c, Utensor)

    solver.solve_Hemb(outfile="test_data")  #CHECK: is passing the filename as a keyword arg working properly?
    denMat_mps = solver.calc_density_matrix()
    print('denMat_mps=')
    print(denMat_mps)

    if do_ci_calc:
        from simple_ed import *
        edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                            N_sector=ntot//2, Sz_sector=0, dtype=np.complex128)
        
        h1e = np.zeros((ntot//2,ntot//2),dtype=complex)
        h1e[:nimp//2,:nimp//2] = eloc[::2,::2]
        h1e[:nimp//2,nimp//2:] = D[::2,::2].T
        h1e[nimp//2:,:nimp//2] = D[::2,::2].conj()
        h1e[nimp//2:,nimp//2:] = -Lambda_c[::2,::2]
        h1e = np.kron(h1e, np.eye(2))
        print('h1e=')
        print(h1e)
        #Utensor = np.zeros((2,2,2,2),dtype=complex)
        #Utensor[0,0,1,1] = U
        #Utensor[1,1,0,0] = U
        
        edsolver.build_Hemb(h1e, Utensor, spin_pen=0.0, sz_pen=0.0)
        edsolver.solve_Hemb(num_eig=1, verbose=True )
        denMat = edsolver.calc_density_matrix()
        np.set_printoptions(precision=3, threshold=np.inf, linewidth=np.inf)
        print('density matrix CI=')
        print(denMat)
        print('density matrix MPS=')
        print(denMat_mps)
        
        
        print('diff in density matrix=')
        print(denMat-denMat_mps)
    
