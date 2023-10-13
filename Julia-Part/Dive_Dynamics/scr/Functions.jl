
module SysInit
    # initialize dynamics, mainly sampling 
    using TensorOperations

    function sampleGC(Ns,Nr,ρ)
        sample = rand(Ns,2,Nr)
        L = [[1, ρ] [0, sqrt(1-ρ^2)]]

        @tensoropt sample[a,b,c] = L[b,e] * sample[a,e,c]

        G = sample[:,1,:]
        C = sample[:,2,:]

        C = C .* (0.01 .+ 0.99*rand(Nr))'

        return G, C
    end

    function sampleSt(Ns,Nr, low = 0.01, upp = 1.0)
        # uniform distribution
        Ss = low .+ (upp-low)*rand(Ns)
        Rs = low .+ (upp-low)*rand(Nr)
        return Ss, Rs
    end

    function getδ(G,Rs)
        return G * Rs
    end

    function getgK(C,Ss,Rs,low=0.1,upp=1.0)
        # sample g and calculate K 
        Nr = size(C)[2]
        g = low .+ (upp - low)*rand(Nr)
        K = (C' * Ss)./g + Rs
        return g, K
    end

    function getlkappa(C,Ss,Rs,low=0.1,upp=1.0)
        # sample l and calculate κ :(Rs * (Ss @ C))/l + Rs
        Nr = size(C)[2]
        l = low .+ (upp - low)*rand(Nr)
        κ = Rs.*(C' * Ss)./l + Rs
        return l, κ
    end

    export sampleGC, sampleSt, getδ, getgK, getlkappa
end;



module EGC

    using Flux
    using Functors
    using LinearAlgebra
    #using LaTeXStrings
    using Statistics

    export trainI, trainInew

    # Functions to calculate E(G,C)
    # Flux and Functor needed


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

    function Hs(G)
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
        
        return minDis/2
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

    function Lossfun1(NG,CD,GGdiss)
        # normaized to the simplex
        CD = sum(CD,dims=2).^(-1) .* CD

        # G-C distances
        GCdiss = norm.(eachrow(NG-CD))

        # G-G distances given by H
        # by definition
        Es = GCdiss ./ GGdiss
        # + std(Es)
        return mean(Es) + 3*std(Es)
    end

    function meanE(NG,CD,GGdiss)
        # normaized to the simplex
        CD = sum(CD,dims=2).^(-1) .* CD

        # G-C distances
        GCdiss = norm.(eachrow(NG-CD))

        # G-G distances given by H
        # by definition
        Es = GCdiss ./ GGdiss
        return mean(Es)
    end

    # begin to define one layer

    struct Rescale
        Diagonal::Matrix{Float64}
    end

    # init
    function Rescale(Nr::Integer)
        Rescale(ones(1,Nr))
    end

    @functor Rescale

    # forward
    function (R::Rescale)(C)
        return (C .* R.Diagonal)
    end

    # end of definition

    function trainI(G,C,lr=0.1,num_epoch=1000)
        HG = H(G) 
        NorG = NormalG(G)
        layer = Rescale(size(G)[2])

        opt_state = Flux.setup(Adam(lr), layer)

        for epoch = 1:num_epoch
            grads = gradient(layer) do m
                CD = m(C)
                Lossfun(NorG,CD,HG)
            end
            Flux.update!(opt_state, layer, grads[1])
        end
        return Lossfun(NorG,layer(C),HG)
    end

    function trainInew(G,C,lr=0.1,num_epoch=1500)
        GGdiss = Hs(G) 
        NorG = NormalG(G)
        layer = Rescale(size(G)[2])

        opt_state = Flux.setup(Adam(lr), layer)

        for epoch = 1:num_epoch
            grads = gradient(layer) do m
                CD = m(C)
                Lossfun1(NorG,CD,GGdiss)
            end
            Flux.update!(opt_state, layer, grads[1])
        end
        return meanE(NorG,layer(C),GGdiss)
    end

end;