# -*- coding: utf-8 -*-
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
task_id = parse(Int,ARGS[1])
num_tasks = parse(Int,ARGS[2])
#task_id = 0
#num_tasks = 48
j_span = [task_id + 1, task_id + 49, task_id + 97]

# input C,G,R,S
loaded = matread("allsamples.mat")
loaded1 = matread("LinLogsamples.mat")

C_span = Float64.(loaded["allC"])
G_span = Float64.(loaded["allG"])
Sstar_span = Float64.(loaded["allSstar"])
Rstar_span = Float64.(loaded["allRstar"])
g_span = Float64.(loaded1["allg"])
K_span = Float64.(loaded1["allK"])

Nr = loaded["Nr"]
Ns_span = loaded["Ns_span"]

num_try = loaded["num_try"]
rho_span = Float64.(loaded["rho_span"])
batch_size = 10 # 10 different initial conditions

# Read in file of function
include("./RCMfunction.jl")

# initialze output of this file
Fractions = zeros(length(Ns_span),3,num_try,3)

# for loops begin simulation
tick()
for i in eachindex(Ns_span) # for Ns
    Ns = Ns_span[i]

    for j_num = 1:3
        j = j_span[j_num] # rho

        C = C_span[i,j,1:Ns,:]
        G = G_span[i,j,1:Ns,:]

        for k in 1:num_try # for R^* and S^*

            Sstar = Sstar_span[i,j,k,1:Ns]
            Rstar = Rstar_span[i,j,k,:]

            δ = G * Rstar
            g = g_span[i,j,k,:]
            K = K_span[i,j,k,:]

            tspan = (0.0, 5000.0) # time for one test solution
            para = (Ns,Nr,G,C,g,K,δ)

            fstate = zeros(batch_size,Ns+2)
            # fstate[1:Ns]: species abundance; fstate[Ns+1]: flucturate or not; 
            # fstate[Ns+2]: surviving fraction

            for batch in 1:batch_size
            # sample an initial condition
            u0 = zeros(Ns+Nr)
            u0[1:Ns] = Sstar .* (1 .+ .5*(rand(Ns) .- 0.5))
            u0[Ns+1:end] = Rstar .* (1 .+ .5*(rand(Nr) .- 0.5))
            iter = 1

            stop = 0
            while stop == 0
                prob = ODEProblem(ResConLog!,u0,tspan,para)
                #VCABM3() or Tsit5()
                sol = solve(prob, saveat = 10.0, Tsit5())
                meanS = mean(sol.u[end-100:end])[1:Ns]
                global index = findall(meanS .> 1e-5)
                global flagS = mean(abs.((sol.u[end][index] - meanS[index])./meanS[index]))
                if iter == 15 || flagS <= 1e-3
                    stop = 1
                end
                iter = iter + 1
                u0 = sol.u[end]
                u0[u0 .< 1.0e-9] .= 1.0e-9
            end

            fluc = 0.0
            if (flagS > 0.01) & (iter == 16)
                fluc = 1.0
            end

            fstate[batch,1:Ns] = u0[1:Ns] # so we don't need to global sol later!
            fstate[batch,Ns+1] = fluc
            fstate[batch,Ns+2] = length(index)/Ns
            end

            Fractions[i,j_num,k,1] = mean(fstate[:,Ns+2]) # surviving
            Fractions[i,j_num,k,2] = mean(fstate[:,Ns+1]) # fluc

            if mean(fstate[:,Ns+1]) < 1.0
                nonfluc_idx = findall(x -> x == 0.0, fstate[:,Ns+1])
                nonfluc_spe = fstate[nonfluc_idx,1:Ns]
                nonfluc_spe[nonfluc_spe .> 1e-5] .= 1.0
                nonfluc_spe[nonfluc_spe .< 1e-5] .= 0.0
                innerprod = nonfluc_spe * nonfluc_spe'
                #innerprod = fstate[nonfluc_idx,1:Ns] * (fstate[nonfluc_idx,1:Ns])'
                innerprod = diagm(sqrt.(diag(innerprod)).^(-1)) * innerprod * diagm(sqrt.(diag(innerprod)).^(-1))
                PCA = real(eigvals(innerprod))
                if length(PCA[PCA .> 0.1]) > 1
                    Fractions[i,j_num,k,3] = 1 - Fractions[i,j_num,k,2]
                end
            end

        end # for k

    end # for loop of j 

    println("Running to Ns = ", Ns)
    flush(stdout)
end # for loop i
tock()

# save
myfilename = "SimuFractionsLog"*string(task_id)*".mat"
file = matopen(myfilename, "w")
write(file, "Fractions", Fractions)
close(file)
