# Test DynamicalSystem

## packages
using DynamicalSystems
using DifferentialEquations
using Plots
using Statistics
using LinearAlgebra
include("./scr/Functions.jl")
using .SysInit

include("../Jobarray/RCMfunction.jl")

using TickTock

## inputs
ρ = 0.8
Ns = 32
Nr = 32

G, C = sampleGC(Ns,Nr,ρ)
Ss, Rs = sampleSt(Ns,Nr)

δ = getδ(G,Rs)
g, K = getgK(C,Ss,Rs)

para = (Ns,Nr,G,C,g,K,δ)
num_init = 10
fstates = zeros(num_init,5)
ufs = Matrix{Float64}(undef, Ns+Nr, 0)
for ii in 1:num_init
    fstates[ii,:], uf = solRCM(Ss,Rs,para,ResConLog!) # one initial condition
    if fstates[ii,4] == 1.0
        ufs = [ufs uf]
    end
end
innerprod = ufs' * ufs
innerprod = diagm(sqrt.(diag(innerprod)).^(-1)) * innerprod * diagm(sqrt.(diag(innerprod)).^(-1))
PCA = real(eigvals(innerprod))

if length(PCA[PCA .> 0.1]) <= 1 # single stable
    fstates[:,4] .= 0.0 # alternative stable states
end