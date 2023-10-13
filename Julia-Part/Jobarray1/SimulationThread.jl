using MAT
using Plots
using LinearAlgebra
using DifferentialEquations
using Random
using Statistics
using LaTeXStrings
using TickTock

# Grab the argument that is passed in
# This is the index into fnames for this process
#task_id = parse(Int,ARGS[1])
#num_tasks = parse(Int,ARGS[2])
#task_id = 0
#num_tasks = 48
#j_span = [task_id + 1, task_id + 49, task_id + 97]

# input C,G,R,S
loaded = matread("allsamples.mat")

C_span = Float64.(loaded["allC"])
G_span = Float64.(loaded["allG"])
Sstar_span = Float64.(loaded["allSstar"])
Rstar_span = Float64.(loaded["allRstar"])

Nr = loaded["Nr"]
Ns_span = loaded["Ns_span"]

num_try = loaded["num_try"]
rho_span = Float64.(loaded["rho_span"])
num_init = 10 # 10 different initial conditions
disturb = 0.1
# Read in file of function
include("./RCMfunction.jl")

# initialze output of this file
Fractions = zeros(length(Ns_span),length(rho_span),num_try,3)

# for loops begin simulation
tick()
for i in eachindex(Ns_span) # for Ns
    Ns = Ns_span[i]

    Threads.@threads for j in eachindex(rho_span)

        C = C_span[i,j,1:Ns,:]
        G = G_span[i,j,1:Ns,:]

        for k in 1:num_try # for R^* and S^*

            Ss = Sstar_span[i,j,k,1:Ns]
            Rs = Rstar_span[i,j,k,:]

            δ = G * Rs
            g = Rs.*(C'*Ss)

            tspan = (0.0, 5000.0) # time for one test solution
            para = (Ns,Nr,G,C,g,δ)

            fstates = zeros(num_init,3)
            ufs = Matrix{Float64}(undef, Ns+Nr, 0)

            for ic in 1:num_init
                fstates[ic,:], uf = solRCM(Ss, Rs, para, ResCon!, disturb)
                if fstates[ic,3] == 1.0
                    ufs = [ufs uf]
                end
            end

            Fractions[i,j,k,1] = mean(fstates[:,1]) # surviving
            Fractions[i,j,k,2] = mean(fstates[:,2]) # fluc
            Fractions[i,j,k,3] = mean(fstates[:,3]) # alternative SS

            if size(ufs)[2] == num_init
                innerprod = ufs' * ufs
                innerprod = diagm(sqrt.(diag(innerprod)).^(-1)) * innerprod * diagm(sqrt.(diag(innerprod)).^(-1))
                PCA = real(eigvals(innerprod))
    
                if length(PCA[PCA .> 0.05*maximum(PCA)]) <= 1
                    Fractions[i,j,k,3] = 0.0 # alternative stable states
                end
            end
            
        end # for k

    end # for loop of j 

    println("Running to Ns = ", Ns)
    flush(stdout)
end # for loop i
tock()

# save
myfilename = "Dis"*string(disturb)*"SimuFractionsThd.mat"
file = matopen(myfilename, "w")
write(file, "Fractions", Fractions)
close(file)