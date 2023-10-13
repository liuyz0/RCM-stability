# Use distributed way to do simulation

using Distributed, MAT, ClusterManagers
addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))
println("Added workers: ", nworkers())
flush(stdout)

@everywhere begin
    using LinearAlgebra
    using SharedArrays
    using DifferentialEquations
    using Random
    using Statistics
    include("./RCMfunction.jl")
end

# input C,G,R,S
loaded = matread("allsamples.mat")
loaded1 = matread("LinLogsamples.mat")

C_span = Float64.(loaded["allC"])
Nr = loaded["Nr"]
Ns_span = loaded["Ns_span"]
num_try = loaded["num_try"]
rho_span = Float64.(loaded["rho_span"])

idxs = [(i,j) for i=1:size(C_span)[1], j=1:size(C_span)[2]]
idxs = reshape(idxs,:)

disturb = 0.1

# build parallel function
@everywhere function parallel_sol(ii, idxs, loaded, loaded1, disturb)
    C_span = Float64.(loaded["allC"])
    G_span = Float64.(loaded["allG"])
    Sstar_span = Float64.(loaded["allSstar"])
    Rstar_span = Float64.(loaded["allRstar"])
    Ns_span = loaded["Ns_span"]
    num_try = loaded["num_try"]
    Nr = loaded["Nr"]

    l_span = Float64.(loaded1["alll"])
    kappa_span = Float64.(loaded1["allkappa"])

    (i,j) = idxs[ii]
    
    num_init = 10
    if Nr == 64
        num_init = 5 # 10 different initial conditions, 5 for NR64
    end

    Ns = Ns_span[i]
    C = C_span[i,j,1:Ns,:]
    G = G_span[i,j,1:Ns,:]
    # initialze output of this file
    Fractions = zeros(num_try,3)


    for k in 1:num_try # for R^* and S^*

        Ss = Sstar_span[i,j,k,1:Ns]
        Rs = Rstar_span[i,j,k,:]

        δ = G * Rs
        #g = Rs.*(C'*Ss)
        l = l_span[i,j,k,:]
        kappa = kappa_span[i,j,k,:]

        #tspan = (0.0, 5000.0) # time for one test solution
        #para = (Ns,Nr,G,C,g,δ)
        para = (Ns,Nr,G,C,l,kappa,δ)

        fstates = zeros(num_init,3)
        ufs = Matrix{Float64}(undef, Ns+Nr, 0)

        for ic in 1:num_init
            fstates[ic,:], uf = solRCM(Ss, Rs, para,ResConLin!, disturb)
            if fstates[ic,3] == 1.0
                ufs = [ufs uf]
            end
        end

        Fractions[k,1] = mean(fstates[:,1]) # surviving
        Fractions[k,2] = mean(fstates[:,2]) # fluc
        Fractions[k,3] = mean(fstates[:,3]) # alternative SS

        if size(ufs)[2] == num_init
            innerprod = ufs' * ufs
            innerprod = diagm(sqrt.(diag(innerprod)).^(-1)) * innerprod * diagm(sqrt.(diag(innerprod)).^(-1))
            PCA = real(eigvals(innerprod))

            if length(PCA[PCA .> 0.05*maximum(PCA)]) <= 1
                Fractions[k,3] = 0.0 # alternative stable states
            end
        end
        
    end # for k

    println("Running to i = ", i, " and j =", j)
    flush(stdout)

    return Fractions

end

@time Fractions_coll = pmap(ii -> parallel_sol(ii, idxs, loaded, loaded1, disturb), 1:length(idxs))
Fractions = zeros(length(Ns_span),length(rho_span),num_try,3)

for ii in eachindex(idxs)
    (i,j) = idxs[ii]
    Fractions[i,j,:,:] = Fractions_coll[ii]
end


# save
myfilename = "Dis"*string(disturb)*"SimuFractionsLin.mat"
file = matopen(myfilename, "w")
write(file, "Fractions", Fractions)
close(file)

# remove all workers
#rmprocs(workers())