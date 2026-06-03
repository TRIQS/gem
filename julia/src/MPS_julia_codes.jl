using ITensors
# activate /home/hh8367/anaconda3/envs/triqs_base/gem.src/julia
# please add this file into your julia project which should be in: path-to-env/gem.src/julia/src
# then recompile

# confirmed to correctly build the EH
function EH_terms(Utensor,H1E,kwargs)
    ## convert python matrices/arrays
    Utensor=GGMPSSolver.PythonCall.pyconvert(Array,Utensor)
    H1Eup=GGMPSSolver.PythonCall.pyconvert(Matrix,H1E["up"])
    H1Edn=GGMPSSolver.PythonCall.pyconvert(Matrix,H1E["dn"])

    ## extract keyword args/parameters
    Nimp=size(Utensor,1)
    N=size(H1Eup,1)

    Nbath=N-Nimp
    kwargs=GGMPSSolver.convert_schedule(kwargs)[]
    conserve_sz=get(kwargs, :use_Sz, true)
    conserve_N=get(kwargs, :use_Ntot,true)
    spin_pen=get(kwargs,:spin_pen,0.0)
    println("params set")

    ## setup Hamiltonian
    perm=collect(1:N)   ###ToDo: implement other arrangements
    os_quadratic=GGMPSSolver.get_H_quadratic(N,H1Eup, H1Edn;perm=perm)
    println("one body terms set")
    sites=GGMPSSolver.ITensors.siteinds("Electron", N; conserve_nf=conserve_N,conserve_sz=conserve_sz)

    if maximum(Utensor) == 0
        println("Utensor is zero. Skipping initialization of os_quartic.")
        os = os_quadratic
        Hint = nothing
        # os_quartic = OpSum()  # Or any default initialization if needed
    else
        os_quartic=GGMPSSolver.get_H_quartic(N,Utensor;perm=perm)
        println("two body terms set")
        os = os_quadratic + os_quartic
        Hint=GGMPSSolver.ITensors.MPO(os_quartic,sites)
    end
    
    os_S2=GGMPSSolver.get_Ssquared(N)

    ## make sites
    println("sites set")
    ## make MPOs
    S2=MPO(os_S2,sites)
    println("s2 set")

    H=GGMPSSolver.ITensors.MPO(os,sites)
    Hnint=GGMPSSolver.ITensors.MPO(os_quadratic,sites)
    println("op sum set")
    # Hint=GGMPSSolver.ITensors.MPO(os_quartic,sites)  ##for <Eimp>        ###FIXME: most likely we'll want to use only the quartic part here
    @show spin_pen
    if !iszero(spin_pen)
       Sz2new=GGMPSSolver.ITensors.MPO(GGMPSSolver.get_Sz_squared(N),sites)
       Spm2new=GGMPSSolver.ITensors.MPO(GGMPSSolver.get_Spm_squared(N),sites)
       Sz2newh=GGMPSSolver.make_hermitian(Sz2new)
       Spm2newh=GGMPSSolver.make_hermitian(Spm2new)
       S2=Spm2newh+Sz2newh
       @show maxlinkdim(H)
       @show maxlinkdim(S2)
       H=H+spin_pen*S2
       @show maxlinkdim(H)
       spincommutator = GGMPSSolver.compute_commutator(H,S2)
       @show spincommutator

        if Hint !== nothing
            spincommutator_int = GGMPSSolver.compute_commutator(Hint, S2)
            @show spincommutator_int
        end

       spincommutator_nint = GGMPSSolver.compute_commutator(Hnint,S2)
       @show spincommutator_nint
       
       if spincommutator_nint > 1e-2
            @show H1Eup
            @show H1Edn
            @show all(H1Eup .== H1Edn)
            #@assert false
       end
       spincommutator>1e-2 && @warn("Spin commutator larger than expected! Likely due to numerical noise in embedding H.")
    end
    return H, perm, sites, N, Nbath
end

# confirmed to correctly solve the EH
function solve_EH(H, perm, sites, N, schedule, tolerances, filling=nothing,magnetization=nothing)
    # ensuring H is a list
    # EH = [H[i] for i in 1:length(H)]
    println("sites:\n", sites)
    ## some checks and threading setup
    # @show outfile
    @show GGMPSSolver.BLAS.get_num_threads()
    @show GGMPSSolver.BLAS.get_config()
    #@show GGMPSSolver.BLAS.set_num_threads(1)
    @show GGMPSSolver.BLAS.get_num_threads()
    @show Threads.nthreads()
    println("before thread handling")
    GGMPSSolver.ITensors.Strided.disable_threads()
    GGMPSSolver.ITensors.enable_threaded_blocksparse()

    println("after thread handling")    ##make starting MPS
    dmrg_params=GGMPSSolver.convert_schedule(schedule)
    tolerances=GGMPSSolver.convert_schedule(tolerances)[]
    filling,magnetization, sector_consistent=check_filling(filling,magnetization, N)
    @assert sector_consistent
    #potentially trigger different behaviour via kwarg
    #assumes that the total system size is even, otherwise not half filled and zero mag
    @assert iseven(length(sites))
    psi=GGMPSSolver.ITensors.randomMPS(sites,init_state_insector(filling,magnetization,N), linkdims=128)

    GS_ene = nothing
    ## setup observers
    internal_obs = GGMPSSolver.Observers.Observer(
        "sweepnumber"=>get_total_sweep,
        "maxdim"=>get_maxdim,
        "energy"=>get_energy,
    )
    obs = GGMPSSolver.MyDMRGObserver(0,internal_obs,perm)

    for (iteration,pars) in enumerate(dmrg_params)
        #we should be passing all these
        #dmrg_kwargs = (nsweeps=Nsweeps[i], reverse_step=false, normalize=true, maxdim=D, cutoff=cutoffs[i], noise=noise[i], outputlevel=1, nsites = 2,)
        #@show typeof(H)
        GS_ene,psi=GGMPSSolver.ITensors.dmrg(H,psi; ishermitian=false, observer=obs,eigsolve_krylovdim=10,pars...)
    end
    return psi, GS_ene
end

function hamiltonian_terms(Utensor_list, M_list,bath_orbs,imp_orbs, kwargs)
    """
    Generates the one-body terms of the embedding Hamiltonian.

    # Arguments
    - `Utensor_list::list`: A list of interaction energy tensors.
    - `M_list::list`: A list of matrices that contain the impurity, hybridization,
        and bath energies between each site.
    - `bath_orbs::int`: An integer specifying the number bath orbitals.
    - `imp_orbs::int`: An array specifying the number of impurity orbitals.
    - `kwargs`: Additional keyword arguments for the construction of the Hamiltonian.

    # Returns
    - `Hr`: The Hamiltonian terms constructed as matrix product operators.

    """

    Nbath = div(bath_orbs, 2)
    Nimp = div(imp_orbs, 2)
    N = Nimp + Nbath
    
    perm=collect(1:N)   ###ToDo: implement other arrangements
    kwargs=GGMPSSolver.convert_schedule(kwargs)[]
    conserve_sz=get(kwargs, :use_Sz, true)
    conserve_N=get(kwargs, :use_Ntot,true)
    spin_pen=get(kwargs,:spin_pen,0.0)
    sites=GGMPSSolver.ITensors.siteinds("Electron", N; conserve_nf=conserve_N,conserve_sz=conserve_sz)
    # println("sites:\n", sites)
    H_list = []
    first_iteration = true
    for (U, M) in zip(Utensor_list, M_list)
        if first_iteration
            # this case if for non zero eloc, utensor, and including spin
            # Convert python matrices/arrays
            Utensor = GGMPSSolver.PythonCall.pyconvert(Array, U)
            H1Eup = GGMPSSolver.PythonCall.pyconvert(Matrix, M["up"])
            H1Edn = GGMPSSolver.PythonCall.pyconvert(Matrix, M["dn"])
            
            ## setup Hamiltonian
            os_quadratic=GGMPSSolver.get_H_quadratic(N,H1Eup, H1Edn;perm=perm)
            println("one body terms set")

            os_quartic=GGMPSSolver.get_H_quartic(N,Utensor;perm=perm)
            println("two body terms set")

            os = os_quadratic + os_quartic
            os_S2=GGMPSSolver.get_Ssquared(N)

            ## make MPOs
            S2=MPO(os_S2,sites)
            H=GGMPSSolver.ITensors.MPO(os,sites)
            Hint=GGMPSSolver.ITensors.MPO(os_quartic,sites)            
            Hnint=GGMPSSolver.ITensors.MPO(os_quadratic,sites)
            println("MPOs set")
            # Hint=GGMPSSolver.ITensors.MPO(os_quartic,sites)  ##for <Eimp>        ###FIXME: most likely we'll want to use only the quartic part here
            @show spin_pen
            if !iszero(spin_pen)
                Sz2new=GGMPSSolver.ITensors.MPO(GGMPSSolver.get_Sz_squared(N),sites)
                Spm2new=GGMPSSolver.ITensors.MPO(GGMPSSolver.get_Spm_squared(N),sites)
                Sz2newh=GGMPSSolver.make_hermitian(Sz2new)
                Spm2newh=GGMPSSolver.make_hermitian(Spm2new)
                S2=Spm2newh+Sz2newh
                # @show maxlinkdim(H)
                # @show maxlinkdim(S2)
                H=H+spin_pen*S2
                # @show H
                # @show maxlinkdim(H)
                spincommutator = GGMPSSolver.compute_commutator(H,S2)
                # @show spincommutator
                
                if Hint !== nothing
                    spincommutator_int = GGMPSSolver.compute_commutator(Hint, S2)
                    @show spincommutator_int
                end
                
                spincommutator_nint = GGMPSSolver.compute_commutator(Hnint,S2)
                # @show spincommutator_nint
                
                if spincommutator_nint > 1e-2
                    @show H1Eup
                    @show H1Edn
                    @show all(H1Eup .== H1Edn)
                    #@assert false
                end
                spincommutator>1e-2 && @warn("Spin commutator larger than expected! Likely due to numerical noise in embedding H.")
            end
            push!(H_list, H)
            first_iteration = false
        else

            # Convert python matrices/arrays
            Utensor = GGMPSSolver.PythonCall.pyconvert(Array, U)
            H1Eup = GGMPSSolver.PythonCall.pyconvert(Matrix, M["up"])
            H1Edn = GGMPSSolver.PythonCall.pyconvert(Matrix, M["dn"])
            
            ## setup Hamiltonian
            os_quadratic=GGMPSSolver.get_H_quadratic(N,H1Eup, H1Edn;perm=perm)
            # println("one body terms set")            
            
            # println("Utensor is zero. Skipping initialization of os_quartic.")
            os = os_quadratic
            H = GGMPSSolver.ITensors.MPO(os,sites)
            push!(H_list, H)
        end
    end
    println("Julia: Length of H_list:", length(H_list))
    return H_list, perm, sites, N, Nbath 
end

function sum_EH_terms(X, H_list, bath_orbs)

    EH = deepcopy(H_list[1])
    for r in 1:bath_orbs
        EH += X[r] * H_list[r+1]
    end
    return EH
end

function store_MPOs(Hr_list, Ht_r_path)
    num_digits = length(string(length(Hr_list)))  # Determine number of digits for padding
    f = h5open(Ht_r_path, "w")  # Open HDF5 file in write mode
    
    try
        for idx in 1:length(Hr_list)
            # Manually format the dataset name with zero-padding
            formatted_index = lpad(string(idx), num_digits, '0')
            name = "H_r_$formatted_index"
            write(f, name, Hr_list[idx])  # Write each MPO to the HDF5 file with the formatted name
        end
        println("MPOs have been stored successfully.")
    finally
        close(f)  # Ensure the file is closed properly
    end
end

function read_MPOs(path::String)
    mpo_list = []

    # Open the HDF5 file for reading
    f = h5open(path, "r")

    # Iterate through each key in the HDF5 file
    for key in keys(f)
        # Read the MPO from the HDF5 file
        mpo = read(f, key, MPO)
        push!(mpo_list, mpo)
    end

    # Close the HDF5 file
    close(f)

    return mpo_list
end

function method_of_snapshots(solver_path::String, samples::Int64)
    """
    Uses the method of snapshots to compute the overlaps if the data is too
    large to be loaded into memory. Instead of loading all data into memory, 
    the Method of snapshots loads 2 states at a time to compute each entry 
    of the overlaps:

        [overlaps]_{i,j} = <state_i | state_j>
    """
    ts =[]
    f = h5open(solver_path, "r")
    
    # Iterate over the groups in the file and append to ts
    for t in keys(f)
        push!(ts, t)
    end

    close(f)
    

    overlap = zeros(ComplexF64, samples, samples)

    for i in 1:samples
        state_i = GGMPSSolver.read_mps_from_file(solver_path, ts[i],"state")

        for j in i:samples
            state_j = GGMPSSolver.read_mps_from_file(solver_path, ts[j],"state")
            overlap[i, j] = GGMPSSolver.ITensors.inner(state_i', state_j)
            # println("overlap: ", overlap[i, j])
            if i != j
                overlap[j, i] = overlap[i, j]
            end
        end

        println("Overlap column $(i) constructed")
    end

    return overlap
end
