# sample and save things for Dive in dynamics
using Statistics
using LinearAlgebra
using MAT
include("./scr/Functions.jl")
using .SysInit

## GCRS for all three dynamics

# we have 144 different ρ, 
# for each ρ, sample 10 different GCRS
# for each GCRS, try 10 different initial conditions

#num_ρ = 144
num_ρ = 1440 #v1

ρ_span = zeros(num_ρ)
ρ_span[1:floor(Int64, num_ρ/2)] = LinRange(0, .85, floor(Int64, num_ρ/2))
ρ_span[floor(Int64, num_ρ/2)+1:end] = LinRange(0.87, 1.0, floor(Int64, num_ρ/2))

Ns = 32
Nr = 32

num_samp = 10
"""
G_span = zeros(num_ρ,num_samp,Ns,Nr)
C_span = zeros(num_ρ,num_samp,Ns,Nr)

R_span = zeros(num_ρ,num_samp,Nr)
S_span = zeros(num_ρ,num_samp,Ns)

for i in 1:num_ρ
    ρ = ρ_span[i]
    for j in 1:num_samp
        G_span[i,j,:,:], C_span[i,j,:,:] = sampleGC(Ns,Nr,ρ)
        S_span[i,j,:], R_span[i,j,:] = sampleSt(Ns,Nr)
    end
end

## saving
matwrite("./data/GCRSv1.mat", Dict(
	"G_span" => G_span,
	"C_span" => C_span,
    "R_span" => R_span,
    "S_span" => S_span
))
"""
## load
vars = matread("./data/GCRSv1.mat")
#G_span = vars["G_span"]
C_span = vars["C_span"]
R_span = vars["R_span"]
S_span = vars["S_span"]


g_span = zeros(num_ρ,num_samp,Nr)
K_span = zeros(num_ρ,num_samp,Nr)

for i in 1:num_ρ
    for j in 1:num_samp
        g_span[i,j,:], K_span[i,j,:] = getgK(C_span[i,j,:,:],S_span[i,j,:],R_span[i,j,:])
    end
end

##
matwrite("./data/gKv1.mat", Dict(
	"g_span" => g_span,
	"K_span" => K_span
))


l_span = zeros(num_ρ,num_samp,Nr)
κ_span = zeros(num_ρ,num_samp,Nr)

for i in 1:num_ρ
    for j in 1:num_samp
        l_span[i,j,:], κ_span[i,j,:] = getlkappa(C_span[i,j,:,:],S_span[i,j,:],R_span[i,j,:])
    end
end

##
matwrite("./data/lkappav1.mat", Dict(
	"l_span" => l_span,
	"kappa_span" => κ_span
))