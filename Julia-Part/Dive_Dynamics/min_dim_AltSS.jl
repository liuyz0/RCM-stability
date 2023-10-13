# find the minimum dimension and case of fully coexisting stable state
# try to answer the question of E = 0.5

# packages
using DifferentialEquations
using DynamicalSystems
using LinearAlgebra
using Random
using MAT
using TensorOperations

@time begin

# functions

@doc raw"""
ResCon!(du, u, p, t)

Resource consumer model with constant reosurce supply
"""
function ResCon!(du,u,para,t)
    N_s,N_r,G,C,gamma,δ = para
    du[1:N_s] = u[1:N_s].*(G*u[N_s+1:N_r+N_s] - δ)
    du[N_s+1:N_r+N_s] = gamma - u[N_s+1:N_r+N_s].*(C'*u[1:N_s])
end

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
    #Rs = low .+ (upp-low)*rand(Nr)
    Rs =  0.1 .+ 0.9*rand(Nr) # metabolic trade-off??

    return G, C, Ss, Rs
end

# inputs 

## Grab the argument that is passed in
# This is the index into fnames for this process
#task_id = parse(Int,ARGS[1])
#num_tasks = parse(Int,ARGS[2])
task_id = 0
num_tasks = 48

diffeq = (alg=AutoTsit5(Rosenbrock23()), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))

#i = task_id + 1
GRID_SIZE = 200

# we have 48 different rho
num_ρ = num_tasks
ρ_span = zeros(num_ρ)
ρ_span[1:floor(Int64, num_ρ/2)] = LinRange(0, .85, floor(Int64, num_ρ/2))
ρ_span[floor(Int64, num_ρ/2)+1:end] = LinRange(0.86, 0.95, floor(Int64, num_ρ/2))

ρ = ρ_span[num_ρ-task_id]

num_N = 10
num_samp = 10 # same para sample 10 diff communities
# iterations over Ns, Nr
find = 0.0
for j in 1:num_N
    Ns = j + 2
    Nr = Ns
    if find == 1.0
        break
    end

    for k in 1:num_samp
        G, C, Ss, Rs = GCSR(Ns, Nr, ρ)
        gamma = Rs.*(C'*Ss)
        δ = G * Rs

        # estimate upper bound
        u0 = zeros(Ns+Nr)
        u0[1:Ns] = Ss .* rand(Ns)
        u0[Ns+1:end] = Rs .* rand(Nr)
        para = (Ns,Nr,G,C,gamma,δ)
        prob = ODEProblem(ResCon!,u0,(0.0,5000.0),para)
        sol = solve(prob, saveat = 10.0, AutoTsit5(Rosenbrock23()), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
        gridupper = maximum(maximum.(sol.u))

        # attractors
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

        # resource consumer model
        RCM = CoupledODEs(ResCon!, u0, para; diffeq)

        # maps initial conditions to attractors via recurrence
        mapper = AttractorsViaRecurrences(RCM, grid; 
        Δt = 0.5, mx_chk_fnd_att=2000, mx_chk_loc_att=2000)
        # find attractor basin fractions
        fractions = basins_fractions(mapper, sss; show_progress=false)
        attractors = extract_attractors(mapper) #extract attractors locations
        num_att = length(attractors)

        full_coe = 0.0
        if (num_att <= 5) & (num_att > 1)
            for (key, att) in attractors
                if length(att) == 1
                    if length(findall(x -> x >= 1e-5 ,att[1][1:Ns])) == Ns
                        full_coe = 1.0
                    end
                end
            end
        end

        if full_coe == 1.0
            global find = 1.0
            println("success, dim = ", Ns)
            flush(stdout)
            matwrite("./data/mindim_task"*string(task_id)*".mat", Dict(
                    "G" => G,
                    "C" => C,
                    "Rs" => Rs,
                    "Ss" => Ss,
                    "num_att" => num_att
                    ))
            break
        end

    end # for k
    println("Going to j = ", j)
    flush(stdout)
end # for j

end