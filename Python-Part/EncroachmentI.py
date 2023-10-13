# to calculate the I for all sampled cases

# no need to sample, read file directly.

import torch
import math
from newEGCcalculator import NormalG, Heteros, Inew
from scipy.io import loadmat # type: ignore
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
# Outputs[0]: IGC_mean; Outputs[1]: IGC_std
Outputs = torch.zeros(2,num_try,len(Ns_span),num_rho)

tik = time.time()
for i in range(len(Ns_span)):
    Ns = Ns_span[i]
    for j in range(num_rho):
        G = G_span[i,j,0:Ns,:]
        C = C_span[i,j,0:Ns,:]

        NG = NormalG(G)
        GGdiss = Heteros(G)

        Imean, Istd = Inew(NG,C,GGdiss)
        Outputs[0,:,i,j] = Imean*torch.ones(num_try)
        Outputs[1,:,i,j] = Istd*torch.ones(num_try)

    print('Progress %d%%, (%s)' % ((i+1)/len(Ns_span) * 100, timeSince(tik)))

tenfile = {"I_m_std": Outputs}
torch.save(tenfile,"./data/newInstd.pt")