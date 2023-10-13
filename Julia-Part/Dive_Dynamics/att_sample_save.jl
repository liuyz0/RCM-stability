# sample G, C, R, S for attractor identifying

using LinearAlgebra
using Random
using MAT
using TensorOperations

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

Ns = 12
Nr = 12

num_ρ = 48
ρ_span = zeros(num_ρ)
ρ_span[1:floor(Int64, num_ρ/2)] = LinRange(0, .85, floor(Int64, num_ρ/2))
ρ_span[floor(Int64, num_ρ/2)+1:end] = LinRange(0.87, 1.0, floor(Int64, num_ρ/2))
num_samp = 16 # given rho, how many samples

G_span = zeros(num_ρ,num_samp,Ns,Nr)
C_span = zeros(num_ρ,num_samp,Ns,Nr)

R_span = zeros(num_ρ,num_samp,Nr)
S_span = zeros(num_ρ,num_samp,Ns)

for i in 1:num_ρ
    ρ = ρ_span[i]
    for j in 1:num_samp
        G_span[i,j,:,:], C_span[i,j,:,:], S_span[i,j,:], R_span[i,j,:] = GCSR(Ns,Nr,ρ)
    end
end



## saving
matwrite("./data/att_GCRS.mat", Dict(
	"G_span" => G_span,
	"C_span" => C_span,
    "R_span" => R_span,
    "S_span" => S_span
))
