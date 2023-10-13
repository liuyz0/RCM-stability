# calcuate new E(G,C)
##
using MAT
include("./scr/Functions.jl")
using .EGC
##
#loaded = matread("./data/GCRS.mat")
loaded = matread("./data/GCRSv1.mat")
G_span = loaded["G_span"]
C_span = loaded["C_span"]

(num_ρ,num_samp,Ns,Nr) = size(G_span)

egc = zeros(num_ρ, num_samp)

for i = 1:num_ρ
    Threads.@threads for j = 1:num_samp
        egc[i,j] = trainInew(G_span[i,j,:,:],C_span[i,j,:,:])
    end
end

##
myfilename = "./data/EGCresultv1v1.mat"
file = matopen(myfilename, "w")
write(file, "egc", egc)
close(file)