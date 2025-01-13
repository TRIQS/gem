function check_convergence(E::Number,Cuu::AbstractMatrix,Cdd::AbstractMatrix,Eold::Number,Cuuold::AbstractMatrix,Cddold::AbstractMatrix,tolerances)
    Etol=tolerances[:E]
    rhotol=tolerances[:rho]
    converged=false
    if maximum(abs.(Cuuold.-Cuu))<=rhotol && maximum(abs.(Cddold.-Cdd))<=rhotol && abs((E-Eold)/E)<=Etol
        converged=true
    end
    return converged
end

function  convert_schedule(schedule)::Vector{NamedTuple}
    param_vec=NamedTuple[]

    for apair in schedule
        keys,values=apair
        push!(param_vec,namedtuple(keys,values))
    end

    return param_vec
end

function get_perm(Nimp,Nbath,es;mu=0.0)
    normal=1:(Nimp+Nbath)
    impnormal=1:Nimp
    bathnormal=Nimp+1:(Nimp+Nbath)
    left=bathnormal[es .< mu]
    right=bathnormal[es .>= mu]
    return sortperm(vcat(left,impnormal,right))
end

function get_perm_bybathabs(Nimp,Nbath,es;mu=0.0)
    return vcat(1:Nimp,sortperm(abs(es)) .+Nimp)
end

function compute_commutator(A,B)
    Bp=prime(siteinds,B)
    Ap=prime(siteinds,A)
    return norm(A*Bp - B*Ap)
end

function make_hermitian(O)
    O=0.5*(O+setprime(siteinds,uniqueinds,prime(dag.(O)),O,0))
    return O
end

function check_hermitian(O)
    O=0.5*(O-setprime(siteinds,uniqueinds,prime(dag.(O)),O,0))
    return norm(O)
end

function symmetrize(A)
    n=length(size(A))
    A .+= conj.(permutedims(A,n:-1:1))
    return A
end

function check_filling(filling::Union{Int,Nothing},magnetization::Union{Int,Nothing}, N::Int)::Tuple{Int,Int,Bool}
    isconsistent=true
    if isnothing(filling)
        filling=N
        magnetization = isnothing(magnetization) ? 0 : magnetization
        isconsistent = iseven(N) && iseven(magnetization)
    else
        @assert !isnothing(magnetization)
        if isodd(filling)
            isconsistent =  isodd(magnetization)
        end
    end
    return filling,magnetization, isconsistent
end

function init_state_insector(filling::Int,magnetization::Int,N::Int)
    Ndown = div(filling-magnetization,2)
    Nup = filling-Ndown
    Nuppos=fill(0,N)
    Nuppos[StatsBase.sample(collect(1:N),Nup;replace=false,ordered=true)] .= 1
    Ndnpos=fill(0,N)
    Ndnpos[StatsBase.sample(collect(1:N),Ndown;replace=false,ordered=true)] .= 2
    stateints=Nuppos + Ndnpos
    int2str=Dict(0=>"Emp",1=>"Up",2=>"Dn",3=>"UpDn")
    statestr=String[]
    for i in 1:N
        push!(statestr,int2str[stateints[i]])
    end
    return statestr
end

function write_mps_to_file(filename,group_path,name,state)
    #assumes that group in group_path already exists
    #and that file is closed! when entering the function
    f=HDF5.h5open(filename,"cw")
    if !(isempty(group_path))
        g=f[group_path]
        HDF5.write(g,name,state)
    else
        HDF5.write(f,name,state)
    end
    close(f)
    return
end

function read_mps_from_file(filename,group_path,name;T=ITensors.MPS::Type)
    #assumes that group in group_path already exists
    #and that file is closed! when entering the function
    f=HDF5.h5open(filename,"r")
    if !(isempty(group_path))
        g=f[group_path]
        state=HDF5.read(g,name,T)
    else
        state=HDF5.read(f,name,T)
    end
    close(f)
    return state
end
