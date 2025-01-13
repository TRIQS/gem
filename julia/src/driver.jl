function solve(Utensor,H1E,schedule,tolerances,kwargs;outfile="data",filling=nothing,magnetization=nothing)
    # kwargs
    #   sweep schedule as a list of Dictionaries or zipped key value pairs
    #   flags: permute sites, diagonalize_bath, min_iters etc.
    #   conserve_qns=true
    #
    #outfile="data"

    ## some checks and threading setup
    @show outfile
    @show GGMPSSolver.BLAS.get_num_threads()
    @show GGMPSSolver.BLAS.get_config()
    #@show GGMPSSolver.BLAS.set_num_threads(1)
    @show GGMPSSolver.BLAS.get_num_threads()
    @show Threads.nthreads()
    println("before thread handling")
    GGMPSSolver.ITensors.Strided.disable_threads()
    GGMPSSolver.ITensors.enable_threaded_blocksparse()
    println("after thread handling")

    ## convert python matrices/arrays
    Utensor=GGMPSSolver.PythonCall.pyconvert(Array,Utensor)
    H1Eup=GGMPSSolver.PythonCall.pyconvert(Matrix,H1E["up"])
    H1Edn=GGMPSSolver.PythonCall.pyconvert(Matrix,H1E["dn"])

    ## extract keyword args/parameters
    Nimp=size(Utensor,1)
    N=size(H1Eup,1)
    filling,magnetization, sector_consistent=check_filling(filling,magnetization, N)
    @assert sector_consistent

    Nbath=N-Nimp
    dmrg_params=GGMPSSolver.convert_schedule(schedule)
    kwargs=GGMPSSolver.convert_schedule(kwargs)[]
    tolerances=GGMPSSolver.convert_schedule(tolerances)[]
    conserve_sz=get(kwargs, :use_Sz, true)
    conserve_N=get(kwargs, :use_Ntot,true)
    spin_pen=get(kwargs,:spin_pen,0.0)

    ## setup Hamiltonian
    perm=collect(1:N)   ###ToDo: implement other arrangements
    os_quadratic=GGMPSSolver.get_H_quadratic(N,H1Eup, H1Edn;perm=perm)
    os_quartic=GGMPSSolver.get_H_quartic(N,Utensor;perm=perm)
    os_S2=GGMPSSolver.get_Ssquared(N)

    ## make sites
    sites=GGMPSSolver.ITensors.siteinds("Electron", N; conserve_nf=conserve_N,conserve_sz=conserve_sz)

    ## make MPOs
    S2=MPO(os_S2,sites)
    os=os_quadratic + os_quartic
    H=GGMPSSolver.ITensors.MPO(os,sites)
    Hnint=GGMPSSolver.ITensors.MPO(os_quadratic,sites)
    Hint=GGMPSSolver.ITensors.MPO(os_quartic,sites)  ##for <Eimp>        ###FIXME: most likely we'll want to use only the quartic part here
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
       spincommutator_int = GGMPSSolver.compute_commutator(Hint,S2)
       @show spincommutator_int
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
    @show GGMPSSolver.ITensors.maxlinkdim(H)

    ##make starting MPS
    #potentially trigger different behaviour via kwarg
    #assumes that the total system size is even, otherwise not half filled and zero mag
    @assert iseven(length(sites))
    psi=GGMPSSolver.ITensors.randomMPS(sites,init_state_insector(filling,magnetization,N), linkdims=128)
    #psi=psi+GGMPSSolver.ITensors.randomMPS(sites,x -> isodd(x) ? "Dn" : "Up", linkdims=64)

    ## initialize quantities for which convergence is assessed
    oldCuu=nothing
    oldCdd=nothing
    Eold=nothing
    ## bring exported quantities into this scope
    Cuu=nothing
    Cdd=nothing
    Eint=nothing
    E=nothing
    ## setup observers
    internal_obs = GGMPSSolver.Observers.Observer(
        "sweepnumber"=>get_total_sweep,
        "maxdim"=>get_maxdim,
        "energy"=>get_energy,
        "corr_dn"=>get_corr_dn,
        "corr_up"=>get_corr_up,
    )
    obs = GGMPSSolver.MyDMRGObserver(0,internal_obs,perm)

    ## run dmrg loop, terminate when tolerances are satisfied
    for (iteration,pars) in enumerate(dmrg_params)
        #we should be passing all these
        #dmrg_kwargs = (nsweeps=Nsweeps[i], reverse_step=false, normalize=true, maxdim=D, cutoff=cutoffs[i], noise=noise[i], outputlevel=1, nsites = 2,)
        #@show typeof(H)
        E,psi=GGMPSSolver.ITensors.dmrg(H,psi; ishermitian=false, observer=obs,eigsolve_krylovdim=10,pars...)
        savedata(outfile,obs.the_observer)
        GC.gc()
        Eint=GGMPSSolver.ITensors.inner(psi',Hint,psi)
        S2val=GGMPSSolver.ITensors.inner(psi',S2,psi)
        Cuu = GGMPSSolver.ITensors.correlation_matrix(psi, "Cdagup", "Cup";ishermitian=true)[perm,perm]
        Cdd = GGMPSSolver.ITensors.correlation_matrix(psi, "Cdagdn", "Cdn";ishermitian=true)[perm,perm]
        GC.gc()
        converged=false
        if !isnothing(oldCuu)
            @show E,Eold
            @show maximum(abs.(oldCuu .- Cuu))
            @show maximum(abs.(oldCdd .- Cdd))
            @show S2val
            converged=GGMPSSolver.check_convergence(E,Cuu,Cdd,Eold,oldCuu,oldCdd,tolerances)
        end
        if converged
            return true, psi, Eint, Cuu, Cdd, E
        end
        oldCuu=deepcopy(Cuu)
        oldCdd=deepcopy(Cdd)
        Eold=deepcopy(E)
    end

    return false, psi, Eint, Cuu, Cdd, E
end
    #eventually implement logging via Observers, pass in an iteration id, so we can save separate HDF5 files for every iteration

