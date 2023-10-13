# find all attractors number, basin fraction, whether fully coexisting!
# job array + multithread
@time begin
# packages
using DifferentialEquations
using DynamicalSystems
using LinearAlgebra
using Random
using MAT

# functions

@doc raw"""
ResCon!(du, u, p, t)

Resource consumer model with constant reosurce supply
"""
function ResCon!(du,u,para,t)
    N_s,N_r,G,C,gamma,δ = para
    du[1:N_s] = u[1:N_s].*(G*u[N_s+1:N_r+N_s] - δ) .+ 1.0e-7
    du[N_s+1:N_r+N_s] = gamma - u[N_s+1:N_r+N_s].*(C'*u[1:N_s])
end

# inputs 

## Grab the argument that is passed in
# This is the index into fnames for this process
#task_id = parse(Int,ARGS[1])
#num_tasks = parse(Int,ARGS[2])
task_id = 43
loaded = matread("./data/att_GCRS.mat")

G_span = loaded["G_span"]
C_span = loaded["C_span"]
R_span = loaded["R_span"]
S_span = loaded["S_span"]
(_, num_samp, Ns, Nr) = size(G_span)
GRID_SIZE = 200
i = task_id + 1 # which correlation to use

diffeq = (alg=AutoVern7(Rodas4()), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))

# initialize output
# outputs: [1] whether full coexisting is stable
# number of attractors ([2] stable, [3] fluc)
# basin fractions ([4] global stable, [5] alt stable, [6] fluc)
outputs = zeros(num_samp, 6)

# iterations
Threads.@threads for j in 1:num_samp
    G = G_span[i,j,:,:]
    C = C_span[i,j,:,:]
    Rs = R_span[i,j,:]
    Ss = S_span[i,j,:]

    gamma = Rs.*(C'*Ss)
    δ = G * Rs

    # estimate upper bound
    u0 = zeros(Ns+Nr)
    u0[1:Ns] = Ss .* rand(Ns)
    u0[Ns+1:end] = Rs .* rand(Nr)
    para = (Ns,Nr,G,C,gamma,δ)
    prob = ODEProblem(ResCon!,u0,(0.0,5000.0),para)
    sol = solve(prob, saveat = 10.0, AutoVern7(Rodas4()), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
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
    Δt = 0.1, mx_chk_fnd_att=800, mx_chk_loc_att=800)
    # find attractor basin fractions
    fractions = basins_fractions(mapper, sss; N=10_000, show_progress=false)
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
            num_stable += 1.0
            if num_att == 1
                frac_gs += fracofthis
            else
                frac_as += fracofthis
            end
            if length(findall(x -> x >= 1e-5 ,att[1][1:Ns])) == Ns
                full_coe = 1.0
            end
        else
            num_fluc += 1.0
            frac_fl += fracofthis
        end

    end

    frac_sum = frac_gs + frac_as + frac_fl
    frac_gs = frac_gs/frac_sum
    frac_as = frac_as/frac_sum
    frac_fl = frac_fl/frac_sum
    # outputs: [1] whether full coexisting is stable
    # number of attractors ([2] stable, [3] fluc)
    # basin fractions ([4] global stable, [5] alt stable, [6] fluc)
    outputs[j,1] = full_coe
    outputs[j,2] = num_stable
    outputs[j,3] = num_fluc
    outputs[j,4] = frac_gs
    outputs[j,5] = frac_as
    outputs[j,6] = frac_fl
    println("Finished ",j, " with totally ",num_samp)
    flush(stdout)
end

# save
myfilename = "./data/att_RCM"*string(task_id)*".mat"
file = matopen(myfilename, "w")
write(file, "outputs", outputs)
close(file)

end