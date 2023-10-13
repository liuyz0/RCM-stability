# Dive in dynamics for logistic supply

# jobarray

## packages
using MAT
using DynamicalSystems
using DifferentialEquations
using Statistics
using LinearAlgebra
include("./scr/Functions.jl")
using .SysInit

include("./scr/RCMfunction.jl")

## Grab the argument that is passed in
# This is the index into fnames for this process
#task_id = parse(Int,ARGS[1])
#num_tasks = parse(Int,ARGS[2])
task_id = 0
i_span = [task_id + 1, task_id + 49, task_id + 97]

## CGRS
loaded = matread("./data/GCRS.mat")
loaded1 = matread("./data/lkappa.mat")

G_span = loaded["G_span"]
C_span = loaded["C_span"]
R_span = loaded["R_span"]
S_span = loaded["S_span"]

l_span = loaded1["l_span"]
κ_span = loaded1["kappa_span"]

(_,num_samp,Ns,Nr) = size(G_span)
num_init = 10

# initialze output of this file
Fractions = zeros(3,num_samp,num_init, 5)

for i_num = 1:3
    i = i_span[i_num]
    for j = 1:num_samp
        G = G_span[i,j,:,:]
        C = C_span[i,j,:,:]
        Rs = R_span[i,j,:]
        Ss = S_span[i,j,:]
        #  N_s,N_r,G,C,l,kappa,δ
        l = l_span[i,j,:]
        κ = κ_span[i,j,:]
        δ = getδ(G,Rs)
        para = (Ns,Nr,G,C,l,κ,δ)

        fstates = zeros(num_init,5)
        ufs = Matrix{Float64}(undef, Ns+Nr, 0)
        for k in 1:num_init
            fstates[k,:], uf = solRCM(Ss,Rs,para,ResConLin!) # one initial condition
            if fstates[k,4] == 1.0
                ufs = [ufs uf]
            end
        end # end for k 
        innerprod = ufs' * ufs
        innerprod = diagm(sqrt.(diag(innerprod)).^(-1)) * innerprod * diagm(sqrt.(diag(innerprod)).^(-1))
        PCA = real(eigvals(innerprod))

        if length(PCA[PCA .> 0.1]) <= 1 # single stable
            fstates[:,4] .= 0.0 # alternative stable states
        end
        Fractions[i_num,j,:,:] = fstates

        println("Running to i_num = ", i_num, " and j = ",j)
        flush(stdout)
    end # end for j
end # end of i 

# save
myfilename = "./data/DiveDynLin"*string(task_id)*".mat"
file = matopen(myfilename, "w")
write(file, "Fractions", Fractions)
close(file)