function get_H_quadratic(n::Int,hup::AbstractMatrix,hdn::AbstractMatrix;perm=1:n,cutoff=1e-6)
    os=OpSum()
    #@show cutoff
    for i in 1:size(hup,1)
        for j in 1:size(hup,2)
            #if abs(hup[i,j])>cutoff
		    #@show Jup[i,j], conj(Jup[j,i])
                @assert hup[i,j]==conj(hup[j,i])
                os+=hup[i,j],"Cdagup",perm[i],"Cup",perm[j]
            #end
            #if abs(hdn[i,j])>cutoff
                @assert hdn[i,j]==conj(hdn[j,i])
                os+=hdn[i,j],"Cdagdn",perm[i],"Cdn",perm[j]
            #end
        end
    end
    return os
    #print("new")
end

function get_H_quartic(n::Int,U::AbstractArray;perm=1:n)
    os=OpSum()
    for i in 1:size(U,1), j in 1:size(U,2), k in 1:size(U,3), l in 1:size(U,4)
        if !iszero(U[i,j,k,l])
            pref=0.5

            cU=U[i,j,k,l]
            os+=(pref*cU,"Cdagup",perm[i],"Cdagdn",perm[j],"Cdn",perm[l],"Cup",perm[k])
            os+=(pref*cU,"Cdagdn",perm[i],"Cdagup",perm[j],"Cup",perm[l],"Cdn",perm[k])
            os+=(pref*cU,"Cdagup",perm[i],"Cdagup",perm[j],"Cup",perm[l],"Cup",perm[k])
            os+=(pref*cU,"Cdagdn",perm[i],"Cdagdn",perm[j],"Cdn",perm[l],"Cdn",perm[k])
        end
    end
    @show os
    return os
end

function get_Ssquared(n)
    os=OpSum()
    for i in 1:n
        for j in 1:n
            os+=0.25, "Cdagup", i, "Cup",i , "Cdagup", j, "Cup",j
            os+=0.25, "Cdagdn", i, "Cdn",i , "Cdagdn", j, "Cdn",j
            os-=0.25, "Cdagdn", i, "Cdn",i , "Cdagup", j, "Cup",j
            os-=0.25, "Cdagup", i, "Cup",i , "Cdagdn", j, "Cdn",j


            os+=0.25, "Cup",i,"Cdagdn",i,"Cdn",j,"Cdagup",j
            os+=0.25,"Cup",j,"Cdagdn",j,"Cdn",i,"Cdagup",i
            os+=0.25, "Cdn",j,"Cdagup",j,"Cup",i,"Cdagdn",i
            os+=0.25, "Cdn",i,"Cdagup",i,"Cup",j,"Cdagdn",j
        end
    end
    return os
end

function get_Ssquared_alt(n)
    os=OpSum()
    for i in 1:n
        for j in 1:n
            os += 0.5, "S+",i,"S-",j
            os += 0.5, "S-",i,"S+",j
            os += "Sz",i,"Sz",j
        end
    end
    return os
end

function get_Sz_squared(n)
    os=OpSum()
    for i in 1:n
        for j in 1:n
            os += "Sz",i,"Sz",j
        end
    end
    return os
end


function get_Spm_squared(n)
    os=OpSum()
    for i in 1:n
        for j in 1:n
            os += 0.5, "S+",i,"S-",j
            os += 0.5, "S-",i,"S+",j
        end
    end
    return os
end
