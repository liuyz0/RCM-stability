# Test during constructions

using LinearAlgebra
using MAT

vars = matread("./data/mindim_task0.mat")
##
G = vars["G"]
C = vars["G"]
num_att = vars["num_att"]