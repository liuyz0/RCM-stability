# Use julia to solve all the dynamics, output 1. surviving fraction, 2. fluctuation fraction, and 3. fraction of ass

# i: Ns
# j: rho
# k: fixed point

# batch: initial condixtions

# output a tensor (i,j,k,3)

using MAT
using Plots
using LinearAlgebra
using DifferentialEquations
using Random
using Statistics
using LaTeXStrings

# input C,G,R,S
loaded = matread("./data/allsamples.mat")

C_span = Float64.(loaded["allC"])
G_span = Float64.(loaded["allG"])
Sstar_span = Float64.(loaded["allSstar"])
Rstar_span = Float64.(loaded["allRstar"])

Nr = loaded["Nr"]
Ns_span = loaded["Ns_span"]

num_try = loaded["num_try"]
rho_span = Float64.(loaded["rho_span"])
batch_size = 10 # 10 different initial conditions

# functions
function ResCon!(du,u,para,t)
    N_s,N_r,G,C,g,δ = para
    du[1:N_s] = u[1:N_s].*(G*u[N_s+1:N_r+N_s] - δ) + 1e-7*ones(N_s) 
    du[N_s+1:N_r+N_s] = g - u[N_s+1:N_r+N_s].*(C'*u[1:N_s])
end


# initialze output of this file
Fractions = zeros(2,2,2,3)

# for loops begin simulation
i = 30 # Ns
Ns = Ns_span[i]
j = 50 # rho

C = C_span[i,j,1:Ns,:]
G = G_span[i,j,1:Ns,:]

k = 1 # R^* and S^*

Sstar = Sstar_span[i,j,k,1:Ns]
Rstar = Rstar_span[i,j,k,:]

δ = G * Rstar
g = Rstar.*(C'*Sstar)

tspan = (0.0, 5000.0) # time for one test solution
para = (Ns,Nr,G,C,g,δ)

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
    prob = ODEProblem(ResCon!,u0,tspan,para)
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
end

fluc = 0.0
if (flagS > 0.01) & (iter == 16)
    fluc = 1.0
end

fstate[batch,1:Ns] = u0[1:Ns] # so we don't need to global sol later!
fstate[batch,Ns+1] = fluc
fstate[batch,Ns+2] = length(index)/Ns
end

Fractions[1,1,1,1] = mean(fstate[:,Ns+2]) # surviving
Fractions[1,1,1,2] = mean(fstate[:,Ns+1]) # fluc

if mean(fstate[:,Ns+1]) < 1.0
    nonfluc_idx = findall(x -> x == 0.0, fstate[:,Ns+1])
    innerprod = fstate[nonfluc_idx,1:Ns] * (fstate[nonfluc_idx,1:Ns])'
    innerprod = diagm(sqrt.(diag(innerprod)).^(-1)) * innerprod * diagm(sqrt.(diag(innerprod)).^(-1))
    PCA = real(eigvals(innerprod))
    if length(PCA[PCA .> 0.1]) > 1
        Fractions[1,1,1,3] = 1 - Fractions[1,1,1,2]
    end
end

# save
myfilename = "./data/testSimu"*string(0)*".mat"
file = matopen(myfilename, "w")
write(file, "Fractions", Fractions)
close(file)