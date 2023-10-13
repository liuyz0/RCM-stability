# test attractor identifier

## packages
using DifferentialEquations
#using Distributions
using DynamicalSystems
using LinearAlgebra
#using ProgressBars
using Random
using MAT
using TensorOperations

using Plots

## functions

@doc raw"""
GCSR(Ns,Nr,``\rho'')

Get G, C, S^*, and R^*;
"""
function GCSR(Ns, Nr, ρ, low = 0.01, upp = 1.0)
    sample = rand(Ns,2,Nr)
    L = [[1, ρ] [0, sqrt(1-ρ^2)]]

    @tensoropt sample[a,b,c] = L[b,e] * sample[a,e,c]

    G = sample[:,1,:]
    C = sample[:,2,:]

    C = C .* (0.01 .+ 0.99*rand(Nr))'

    # uniform distribution
    Ss = low .+ (upp-low)*rand(Ns)
    Rs = low .+ (upp-low)*rand(Nr)

    return G, C, Ss, Rs
end

@doc raw"""
ResCon!(du, u, p, t)

Resource consumer model with constant reosurce supply
"""
function ResCon!(du,u,para,t)
    N_s,N_r,G,C,gamma,δ = para
    du[1:N_s] = u[1:N_s].*(G*u[N_s+1:N_r+N_s] - δ)
    du[N_s+1:N_r+N_s] = gamma - u[N_s+1:N_r+N_s].*(C'*u[1:N_s])
end

## inputs 
#Nr = 12 # number of resources
#Ns = 12 # number of species

#ρ = 0.98 # corr between G and C, will be a range

#G, C, Ss, Rs = GCSR(Ns,Nr,ρ)
#loaded = matread("./data/att_GCRS.mat")
"""
G_span = loaded["G_span"]
C_span = loaded["C_span"]
R_span = loaded["R_span"]
S_span = loaded["S_span"]

i = 44
j = 2

G = G_span[i,j,:,:]
C = C_span[i,j,:,:]
Rs = R_span[i,j,:]
Ss = S_span[i,j,:]
"""
loaded = matread("./data/mindim_task0.mat")
G = loaded["G"]
C = loaded["C"]
Rs = loaded["Rs"]
Ss = loaded["Ss"]

Ns = size(G)[1]
Nr = size(G)[2]

gamma = Rs.*(C'*Ss)
δ = G * Rs

u0 = zeros(Ns+Nr)
u0[1:Ns] = Ss .* rand(Ns)
u0[Ns+1:end] = Rs .* rand(Nr)
para = (Ns,Nr,G,C,gamma,δ)
prob = ODEProblem(ResCon!,u0,(0.0,5000.0),para)
sol = solve(prob, saveat = 10.0, AutoVern7(Rodas4()), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
gridupper = maximum(maximum.(sol.u))


GRID_SIZE = 200

speg = range(0.0, 2.0*gridupper; length=GRID_SIZE) #grid dimension for speices
resg = range(0.0, 2.0; length=GRID_SIZE) #grid dimension for resources
function gridfun(i::Integer)
    if i <= Ns
        return speg
    else
        return resg
    end
end
grid = ntuple(i->gridfun(i), Ns+Nr) #grid

# samples initial conditions for finding basin fractions
sss, = statespace_sampler(;
min_bounds=Tuple(zeros(Ns+Nr)), max_bounds=Tuple(gridupper*ones(Ns+Nr))
)

diffeq = (alg=Tsit5(), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
# resource consumer model
RCM = CoupledODEs(ResCon!, u0, para; diffeq)

# maps initial conditions to attractors via recurrence
mapper = AttractorsViaRecurrences(RCM, grid; mx_chk_fnd_att=1000, mx_chk_loc_att=1000)
# find attractor basin fractions
fractions = basins_fractions(mapper, sss; show_progress=false)
attractors = extract_attractors(mapper) #extract attractors locations

num_att = length(attractors)

num_stable = 0.0
num_fluc = 0.0
full_coe = 0.0

frac_gs = 0.0
frac_as = 0.0
frac_fl = 0.0

for (key, att) in attractors
    # our initial sample may not able to find all attractors
    fracofthis = 0.0
    if key in keys(fractions)
        fracofthis = fractions[key]
    end
    if length(att) == 1
        global num_stable = num_stable + 1.0
        if num_att == 1
            global frac_gs = frac_gs + fracofthis
        else
            global frac_as = frac_as + fracofthis
        end
        if length(findall(x -> x >= 1e-5 ,att[1][1:Ns])) == Ns
            global full_coe = 1
        end
    else
        global num_fluc = num_fluc + 1.0
        global frac_fl = frac_fl + fracofthis
    end
end

frac_sum = frac_gs + frac_as + frac_fl
frac_gs = frac_gs/frac_sum
frac_as = frac_as/frac_sum
frac_fl = frac_fl/frac_sum

# outputs: whether full coexisting is stable
# number of attractors (stable, fluc)
# basin fractions (global stable, alt stable, fluc)


##
plotly()
plot(sol,idx=1:12,legend=false)
