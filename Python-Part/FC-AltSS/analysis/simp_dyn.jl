## packages
using DifferentialEquations
#using Distributions
using DynamicalSystems
using LinearAlgebra
#using ProgressBars
using Random
using MAT

using Plots

## functions

@doc raw"""
ResCon!(du, u, p, t)

Resource consumer model with constant reosurce supply
"""
function ResCon!(du,u,para,t)
    N_s,N_r,G,C,gamma,δ = para
    du[1:N_s] = u[1:N_s].*(G*u[N_s+1:N_r+N_s] - δ)
    du[N_s+1:N_r+N_s] = gamma - u[N_s+1:N_r+N_s].*(C'*u[1:N_s])
end

loaded = matread("./data/forjl.mat")
G = loaded["G"]
C = loaded["C"]
Rs = vec(loaded["Rs"])
Ss = vec(loaded["Ss"])


Ns = size(G)[1]
Nr = size(G)[2]

gamma = Rs.*(C'*Ss)
δ = G * Rs

##
#u0 = [Ss;Rs]
#u0 = zeros(Ns+Nr)
#u0[1:Ns] = [1.1118, 0.0 ,3.0352, 0.0]
#u0[Ns+1:end] = [0.5984, 0.6657, 0.7067, 0.1690]
u0 = 2*rand(Ns+Nr)
para = (Ns,Nr,G,C,gamma,δ)
prob = ODEProblem(ResCon!,u0,(0.0,100000.0),para)
sol = solve(prob, saveat = 10.0, Tsit5(), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))

##
plotly()
plot(sol,
    idxs=1:Ns,
    ylabel = "Species abundances",
    ylims=(0,Inf),
    xtickfont=font(11),
    ytickfont=font(11),
    guidefont=font(11),
    legendfont = font(11),
    size = (400, 300))
##
plot(sol,
    idxs=(1,2,3),
    ylims=(0,Inf),
    xlims=(0,Inf),
    zlims=(0,Inf),
    lw = 2.0)