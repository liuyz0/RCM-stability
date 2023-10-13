# This code evaluate the encroachment, E(G,C), defined, using the real 
# Jacobian, J, as the evidence.

# packages
import torch
import torch.nn as nn
from torch import optim
import math
import time
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.io import loadmat
import numpy as np

#torch.set_default_tensor_type(torch.DoubleTensor)

# functions

## this one is used to get CD
class rescale(nn.Module):
    def __init__(self,Nr):
        super(rescale, self).__init__()
        self.D = nn.Parameter(torch.ones(Nr)) # not sure if devided by sqrt(Nr)

    def forward(self, C):
        output = C @ torch.diag(self.D)
        return output

## the following three used for calculating loss function
def NormalG(G):
    G = torch.diag((torch.sum(G, 1))**(-1)) @ G
    return G

def H(G,Ns):
    # normaized to the simplex
    G = torch.diag((torch.sum(G, 1))**(-1)) @ G

    # G-G distances
    distances = torch.zeros(Ns,Ns)
    for i in range(Ns-1):
        for j in range(i+1, Ns):
            distances[i,j] = torch.linalg.norm(G[i] - G[j])
    distances = distances + distances.transpose(0,1) + 2*torch.diag(torch.ones(Ns))
    minDis = torch.min(distances, 1).values
    H = torch.mean(minDis)
    return H


def Loss(NG,CD,H):
    # normaized to the simplex
    CD = torch.diag((torch.sum(CD, 1))**(-1)) @ CD

    # G-C distances
    I = torch.mean(torch.linalg.norm(NG - CD, dim=1))

    # G-G distances given by H
    # by definition
    E = I/H
    return E

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# load all parameters need
loaded = loadmat("./data/allsamples.mat")
Nr = int(loaded["Nr"])
Ns_span = loaded["Ns_span"][0]
num_try = int(loaded["num_try"])

rho_span = torch.tensor(loaded["rho_span"][0])

# initialization of outputs
# Outputs[0]: FU_J; Outputs[1]: FU_GC; Outputs[2]: E;
Outputs = torch.zeros(3,num_try,len(Ns_span),len(rho_span))

tik = time.time()
for i in range(len(Ns_span)):
    Ns = Ns_span[i]
    for j in range(len(rho_span)):
        
        G = torch.tensor(loaded["allG"][i,j,:Ns,:])
        C = torch.tensor(loaded["allC"][i,j,:Ns,:])

        HG = H(G,Ns) 
        NorG = NormalG(G)
        lr = 0.1
        epochs = 1000

        layer = rescale(Nr) # get new model!
        opt = optim.SGD(layer.parameters(), lr=lr, momentum=0.9)

        E = 0 # initailize E(G,C)
        # training process
        for epoch in range(epochs):
            CD = layer(C)
            loss = Loss(NorG,CD,HG)

            loss.backward()
            opt.step()
            opt.zero_grad()

            if epoch >= epochs - 100:
                E = (E*(epoch - epochs + 100) + loss.item())/(epoch - epochs + 101)
        
        Outputs[2,:,i,j] = E*torch.ones(num_try)

        for k in range(num_try):
            Sstar = torch.tensor(loaded["allSstar"][i,j,k,:Ns])
            Rstar = torch.tensor(loaded["allRstar"][i,j,k,:])

            Jstar = torch.zeros(Ns+Nr,Ns+Nr)
            Jstar[0:Ns,Ns:Ns+Nr] = torch.diag(Sstar) @ G
            Jstar[Ns:Ns+Nr,0:Ns] = - torch.diag(Rstar) @ C.transpose(0,1)
            Jstar[Ns:Ns+Nr,Ns:Ns+Nr] = - torch.diag(C.transpose(0,1) @ Sstar)

            E_J = torch.linalg.eigvals(Jstar).real
            Outputs[0,k,i,j] = len(E_J[E_J >= 1.0e-3])/Ns # Fraction of Unstable modes of the real Jacobian
            E_GC = torch.linalg.eigvals(- G @ C.transpose(0,1)).real
            Outputs[1,k,i,j] = len(E_GC[E_GC >= 1.0e-3])/Ns
    
    print('Progress %d%%, (%s)' % ((i+1)/len(Ns_span) * 100, timeSince(tik)))

datapath = Path('./data/').expanduser()
tensorfile = {'Outputs': Outputs, 'Nr': Nr, 'Ns_span': Ns_span, 'rho_span': rho_span}
torch.save(tensorfile, datapath/'torch_test')