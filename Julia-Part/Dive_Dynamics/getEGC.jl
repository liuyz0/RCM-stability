# calcuate E(G,C)
##
using MAT
include("./scr/Functions.jl")
using .EGC
##
loaded = matread("./data/GCRS.mat")
G_span = loaded["G_span"]
C_span = loaded["C_span"]

(num_ρ,num_samp,Ns,Nr) = size(G_span)

egc = zeros(num_ρ, num_samp)

for i = 1:num_ρ
    for j = 1:num_samp
        egc[i,j] = trainI(G_span[i,j,:,:],C_span[i,j,:,:])
    end
end

##
myfilename = "./data/EGCresult.mat"
file = matopen(myfilename, "w")
write(file, "egc", egc)
close(file)