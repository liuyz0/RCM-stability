# to calculate the new Encroachment which has mean and std

# no need to sample, read file directly.

import torch
import math
from newEGCcalculator import trainI
from scipy.io import loadmat, savemat # type: ignore
import time

loaded = loadmat("./data/allsamples.mat")

C_span = torch.tensor(loaded["allC"])
G_span = torch.tensor(loaded["allG"])

rho_span = loaded["rho_span"][0]
num_rho = len(rho_span)

Ns_span = loaded["Ns_span"][0]
Nr = int(loaded["Nr"])
num_try = int(loaded["num_try"])

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# initialization of outputs
# Outputs[0]: EGC_mean; Outputs[1]: EGC_std
Outputs = torch.zeros(2,num_try,len(Ns_span),num_rho)

tik = time.time()
for i in range(len(Ns_span)):
    Ns = Ns_span[i]
    for j in range(num_rho):
        G = G_span[i,j,0:Ns,:]
        C = C_span[i,j,0:Ns,:]

        Emean, Estd = trainI(G,C)
        Outputs[0,:,i,j] = Emean*torch.ones(num_try)
        Outputs[1,:,i,j] = Estd*torch.ones(num_try)

    print('Progress %d%%, (%s)' % ((i+1)/len(Ns_span) * 100, timeSince(tik)))

tensordic = {"EGC_m_std": Outputs.numpy()}
savemat("./data/newEGCnstd.mat", tensordic)