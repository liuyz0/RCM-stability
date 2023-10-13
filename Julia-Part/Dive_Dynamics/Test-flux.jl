# Since we use julia to do optimization here
# need to figure out how to use Flux.jl 

## packages
using MAT
using Plots

using LinearAlgebra
using Random
#using DifferentialEquations
using Statistics

using LaTeXStrings
using TickTock

## functions

include("../Jobarray/RCMfunction.jl")

function NormalG(G)
    G = (sum(G, dims=2)).^(-1) .* G
    # Ns x 1 matrix
    return G
end

function H(G)
    # normaized to the simplex
    G = sum(G, dims=2).^(-1) .* G

    Ns = size(G)[1]
    # G-G distances
    distances = zeros(Ns,Ns)
    for i in 1:Ns-1
        for j in i+1:Ns
            distances[i,j] = norm(G[i,:] - G[j,:])
        end
    end
    distances = distances + distances' + 2*diagm(ones(Ns))
    minDis = minimum(distances, dims = 2)
    H = mean(minDis)
    return H 
end

function Lossfun(NG,CD,H)
    # normaized to the simplex
    CD = sum(CD,dims=2).^(-1) .* CD

    # G-C distances
    I = mean(norm.(eachrow(NG-CD)))

    # G-G distances given by H
    # by definition
    E = I/H
    return E
end

struct Rescale
    D
end

# init
function Rescale(Nr::Integer)
    Rescale(ones(1,Nr))
end

# forward
function (R::Rescale)(C)
    D = R.D
    return (C .* D)
end
@functor Rescale

## inputs
ρ = 0
Ns = 32
Nr = 32


sample = rand(Ns,2,Nr)
L = [[1, ρ] [0, sqrt(1-ρ^2)]]

@tensoropt sample[a,b,c] = L[b,e] * sample[a,e,c]

G = sample[:,1,:]
C = sample[:,2,:]

C = C .* (0.01 .+ 0.99*rand(Nr))'

HG = H(G) 
NorG = NormalG(G)
layer = Rescale(Nr)

lr = 0.1

optim = Flux.setup(Adam(lr), layer)
epochs = 1000
losses = zeros(epochs)
for epoch = 1:epochs
    grads = gradient(layer) do m
        CD = m(C)
        Lossfun(NorG,CD,HG)
    end
    losses[epoch] = Lossfun(NorG,layer(C),HG) # comment later
    Flux.update!(optim, layer, grads[1])
end

#plot(losses)

##
include("./scr/Functions.jl")
using .EGC
using .sysInit

ρ = 0.8
Ns = 32
Nr = 32

G, C = sampGC(Ns,Nr,ρ)
E = trainI(G,C)