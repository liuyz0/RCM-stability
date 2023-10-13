# Try to draw time seriers for Figure 1.

## packages

using DynamicalSystems
using DifferentialEquations
using Statistics
using LinearAlgebra
using TensorOperations
using Random
using Plots

include("./scr/RCMfunction.jl")

## Inputs

ρ = 0.4
Ns = 32
Nr = 32

sample = rand(Ns,2,Nr)
L = [[1, ρ] [0, sqrt(1-ρ^2)]]

@tensoropt sample[a,b,c] = L[b,e] * sample[a,e,c]

G = sample[:,1,:]
C = sample[:,2,:]

# C = C .* (0.01 .+ 0.99*rand(Nr))'

Ss = ones(Ns)
Rs = 0.01 .+ 0.99*rand(Nr)

δ = G * Rs
gamma = Rs.*(C'*Ss)
#N_s,N_r,G,C,gamma,δ = para

#
tspan = (0.0, 10000.0) # time for one test solution
para = (Ns,Nr,G,C,gamma,δ)

u0 = zeros(Ns+Nr)
u0[1:Ns] = Ss .* (1 .+ 2.0*(rand(Ns) .- 0.5))
u0[Ns+1:end] = Rs .* (1 .+ 2.0*(rand(Nr) .- 0.5))

prob = ODEProblem(ResCon!,u0,tspan,para)
sol = solve(prob, saveat = 10, VCABM3(), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))

diffeq = (alg = TRBDF2(), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
LyExp = lyapunov(ContinuousDynamicalSystem(ResCon!,sol.u[end],para; diffeq), 5000.0) #Lyapnov exponant!!!
println("LyExp is ", LyExp)

# Plots
plotly()
plot(sol,idxs = 1:Ns,legend = false,
    ylabel="Density",
    xlabel="Time",
    grid=false,
    xtickfont=font(12),
    ylims=(0,Inf),
    ytickfont=font(12),
    guidefont=font(12),
    size = (600, 150)
    )
