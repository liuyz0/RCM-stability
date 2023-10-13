# to solve the problem that no limit cycles

## packages

using MAT
using DynamicalSystems
using DifferentialEquations
using Statistics
using LinearAlgebra
using TensorOperations
using Plots

include("./scr/RCMfunction.jl")

## conditions
ρ = 0.7
Ns = 32
Nr = 32

sample = rand(Ns,2,Nr)
L = [[1, ρ] [0, sqrt(1-ρ^2)]]

@tensoropt sample[a,b,c] = L[b,e] * sample[a,e,c]

G = sample[:,1,:]
C = sample[:,2,:]

C = C .* (0.01 .+ 0.99*rand(Nr))'

Ss = 0.01 .+ 0.99*rand(Ns)
Rs = 0.01 .+ 0.99*rand(Nr)

δ = G * Rs
gamma = Rs.*(C'*Ss)
g = 0.1 .+ 0.9*rand(Nr)

K = (C' * Ss)./g + Rs;


tspan = (0.0, 10000.0) # time for one test solution
#para = (Ns,Nr,G,C,gamma,δ)
para = (Ns,Nr,G,C,g,K,δ)

u0 = zeros(Ns+Nr)
u0[1:Ns] = Ss .* (1 .+ 1.0*(rand(Ns) .- 0.5))
u0[Ns+1:end] = Rs .* (1 .+ 1.0*(rand(Nr) .- 0.5))

prob = ODEProblem(ResConLog!,u0,tspan,para)
#VCABM3() or Tsit5() or TRBDF2()
sol = solve(prob, saveat = 10, VCABM3(), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))

diffeq = (alg = TRBDF2(), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
LyExp = lyapunov(ContinuousDynamicalSystem(ResConLog!,sol.u[end],para; diffeq), 5000) #Lyapnov exponant!!!
println("Lyapnov exp is ", LyExp)

plotly()
plot(sol,idxs = 1:Ns,legend = false,
    ylabel="Species abundances",
    xtickfont=font(12),
    ytickfont=font(12),
    guidefont=font(12),
    legendfont=font(12))
