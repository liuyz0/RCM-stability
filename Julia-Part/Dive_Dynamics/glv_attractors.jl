@time begin

using DifferentialEquations
using Distributions
using DynamicalSystems
using JSON3
using LinearAlgebra
using Printf
using ProgressBars
using Random

RSLT_DIR = "results/glv_attractors/" #directory in project to save results
RNGS = [Xoshiro(i) for i in 1:Threads.nthreads()] #random number generators
GRID_SIZE = 111 #units per grid dimension
DISPERSAL = 1e-6 #scalar lambda
GROWTH_RATE = 1.0 #scalar r
SIGMA = 0.5 #sigma of interaction gamma distribution
MIN_POP = 1e-9 #threshold below which to impose extinction
S_RANGE = range(3, 24; length=4) #species numbers
C_RANGE = range(0.3, 1.2; length=4) #interaction strengths
N_MATRICES = 400 #iterations per parameters

@doc raw"""
interaction_matrix(``\sigma``, s, c)

returns a generalized Lotka-Volterra interaction matrix
with diagonals 1 and off-diagonals distributed as 
``c\cdot\text{Gamma}(1/\sigma^2, \sigma^2)``
using random-number generator rng
"""
function interaction_matrix(rng, σ, s, c)
    # initializes s-by-s matrix with entries ~ Gamma
    α = c*rand(rng, Gamma(1/σ^2,σ^2), (s,s))
    α[diagind(α)] .= 1.0 #sets diagonals to 1
    return α
end

@doc raw"""
GeneralizedLotkaVolterra!(du, u, p, t)

iip dynamical rule for the generalized Lotka-Volterra model

``\frac{du}{dt}=\lambda+ru(1-\alpha*u)``

where p = [``\lambda``, ``r``, ``\alpha``]
"""
function GeneralizedLotkaVolterra!(du, u, p, t)
    λ, r, α = p
    du .= λ+r.*u.*(1 .-α*u)
    return nothing
end

xg = range(0.0, 1.1; length=GRID_SIZE) #grid dimension
# checks if species fall below extinction threshold
condition(u, t, integrator) = any(u.<MIN_POP)
# sets species below extinction threshold to 0
function affect!(integrator)
    integrator.u[integrator.u.<MIN_POP] .= 0.0
end
cb = DiscreteCallback(condition, affect!)
diffeq = (alg=AutoVern7(Rodas4()), reltol=MIN_POP, abstol=MIN_POP, callback=cb)

for s in S_RANGE #iterate over number of species
    s = Integer(s)
    λ = DISPERSAL*ones(s) #vector lambda
    r = GROWTH_RATE*ones(s) #vector r
    grid = ntuple(_->xg, s) #grid
    # samples initial conditions for finding basin fractions
    sss, = statespace_sampler(;
        min_bounds=Tuple(zeros(s)), max_bounds=Tuple(ones(s))
    )
    for c in C_RANGE #iterate over interaction strengths
        println("S = "*string(s)*", C = "*string(c))
        result = Dict{Integer, Dict}()
        # iterate over interaction matrices
        Threads.@threads for matrix_index in ProgressBar(1:N_MATRICES)
            rng = RNGS[Threads.threadid()]
            #initial conditions used to define system
            u0 = rand(rng, s)
            # define interaction matrix
            α = interaction_matrix(rng, SIGMA, s, c)
            p = [λ, r, α] #glv parameters
            # glv dynamical system
            glv = CoupledODEs(GeneralizedLotkaVolterra!, u0, p; diffeq)
            # maps initial conditions to attractors via recurrence
            mapper = AttractorsViaRecurrences(glv, grid;
                    Δt=0.1, mx_chk_fnd_att=800, mx_chk_loc_att=800)
            # find attractor basin fractions
            fractions = basins_fractions(mapper, sss;
                N=10_000, show_progress=false)
            attractors = extract_attractors(mapper) #extract attractors locations
            # attractor data to save in result
            val = Dict("fractions" => fractions,
                "attractors" => attractors,
                "interactions" => α)
            result[matrix_index] = val #save result
        end
        # file name to save result
        fn = RSLT_DIR*"/S"*lpad(s, 2, "0")*
            "C"*replace(string(c), "."=>"")*"e-1.json"
        open(fn, "w") do io #save file
            JSON3.pretty(io, result)
        end
    end
end

end