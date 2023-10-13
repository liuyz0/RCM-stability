# Dive in dynamics for logistic supply

# distributed
using Distributed, MAT, ClusterManagers
#addprocs(4) # add 4 cores, different on cluster!
addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))
println("Added workers: ", nworkers())
flush(stdout)

@everywhere begin
    using DynamicalSystems
    using DifferentialEquations
    using Statistics
    using LinearAlgebra
    #include("./scr/Functions.jl")
    using .SysInit
    include("./scr/RCMfunction.jl")
end


## CGRS
loaded = matread("./data/GCRSv1.mat")
loaded1 = matread("./data/gKv1.mat")

G_span = loaded["G_span"]
(num_rho,num_samp,Ns,Nr) = size(G_span)

idxs = [(i,j) for i=1:num_rho, j=1:num_samp]
idxs = reshape(idxs,:)

num_init = 50 # important!

# build parallel function
@everywhere function parallel_sol(ii, idxs, loaded, loaded1, num_init)
    (i,j) = idxs[ii]

    G_span = loaded["G_span"]
    (_,_,Ns,Nr) = size(G_span)
    C_span = loaded["C_span"]
    R_span = loaded["R_span"]
    S_span = loaded["S_span"]
    g_span = loaded1["g_span"]
    K_span = loaded1["K_span"]

    G = G_span[i,j,:,:]
    C = C_span[i,j,:,:]
    Rs = R_span[i,j,:]
    Ss = S_span[i,j,:]
    g = g_span[i,j,:]
    K = K_span[i,j,:]

    #  N_s,N_r,G,C,l,kappa,δ
    δ = G * Rs
    para = (Ns,Nr,G,C,g,K,δ)

    fstates = zeros(num_init,5)
    ufs = Matrix{Float64}(undef, Ns+Nr, 0)
    for k in 1:num_init
        fstates[k,:], uf = solRCM(Ss,Rs,para,ResConLog!) # one initial condition
        if fstates[k,4] == 1.0
            ufs = [ufs uf]
        end
    end # end for k 
    if size(ufs)[2] == num_init
        innerprod = ufs' * ufs
        innerprod = diagm(sqrt.(diag(innerprod)).^(-1)) * innerprod * diagm(sqrt.(diag(innerprod)).^(-1))
        PCA = real(eigvals(innerprod))

        if length(PCA[PCA .> 0.05*maximum(PCA)]) <= 1 # single stable
            fstates[:,4] .= 0.0 # alternative stable states
        end
    end

    println("Running to i = ", i, " and j = ",j)
    flush(stdout)

    return fstates
end


@time Fractions_coll = pmap(ii -> parallel_sol(ii, idxs, loaded, loaded1, num_init), 1:length(idxs))
Fractions = zeros(num_rho,num_samp,num_init,5)

for ii in eachindex(idxs)
    (i,j) = idxs[ii]
    Fractions[i,j,:,:] = Fractions_coll[ii]
end

# save
myfilename = "./data/DiveDynv1Log.mat"
file = matopen(myfilename, "w")
write(file, "Fractions", Fractions)
close(file)