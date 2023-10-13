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
import numpy as np
from scipy.io import savemat

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

def H(G):
    # normaized to the simplex
    Ns = G.size(dim=0)
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


# input parameters
Nr = 32
Ns_span = range(4,48 + 1)
num_try = 10 # given G,C, how many Sstar and Rstar

num_rho = 144 # must be even
rho_span = torch.zeros(num_rho)
rho_span[:int(num_rho/2)] = torch.linspace(0, .85, steps=int(num_rho/2))
rho_span[int(num_rho/2):] = torch.linspace(.87, 1, steps=int(num_rho/2))

# initialization of outputs
# Outputs[0]: FU_J; Outputs[1]: FU_GC; Outputs[2]: E;
Outputs = torch.zeros(3,num_try,len(Ns_span),len(rho_span))

Csample = torch.zeros(len(Ns_span),len(rho_span),max(Ns_span),Nr)
Gsample = torch.zeros(len(Ns_span),len(rho_span),max(Ns_span),Nr)
Rsample = torch.zeros(len(Ns_span),len(rho_span),num_try,Nr)
Ssample = torch.zeros(len(Ns_span),len(rho_span),num_try,max(Ns_span))

tik = time.time()
for i in range(len(Ns_span)):
    Ns = Ns_span[i]
    for j in range(len(rho_span)):
        rho = rho_span[j]
        # sampling 
        sample = torch.rand(Ns,2,Nr)
        L = torch.tensor([[1, 0],
                        [rho, math.sqrt(1-rho**2)]]) # Cholesky decomposition

        sample = torch.matmul(L,sample)

        G = sample[0:Ns,0]
        Gsample[i,j,0:Ns,:] = G

        C = sample[0:Ns,1] # C has not been pushed away

        C = C @ torch.diag(0.01+ 0.99*torch.rand(Nr))
        Csample[i,j,0:Ns,:] = C

        HG = H(G)
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
            Sstar = 0.01 + 0.99*torch.rand(Ns)
            Rstar = 0.01 + 0.99*torch.rand(Nr)
            Ssample[i,j,k,:Ns] = Sstar
            Rsample[i,j,k,:] = Rstar

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

tensordic = {"allC":Csample.numpy(), "allG": Gsample.numpy(),"allSstar": Ssample.numpy(), 
             "allRstar": Rsample.numpy(), "rho_span": rho_span.numpy(), 
             "Ns_span": Ns_span, "Nr": Nr, "num_try": num_try}
savemat("./data/allsamples.mat", tensordic)
# test plot, will be commented
#plt.figure()
#plt.scatter(Outputs[0],Outputs[1])
#plt.xlabel('Fraction of unstable modes for $J$')
#plt.ylabel('Fraction of unstable modes for $GC^T$')
#plt.show()