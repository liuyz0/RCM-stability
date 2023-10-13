# provides functions to calulate new EGC

import torch
import torch.nn as nn
from torch import optim
import math

def Heteros(G:torch.Tensor
            ) -> torch.Tensor:
    """input G, output heterogeneity, whcih is a Ns-dim vector"""
    Ns = G.shape[0]
    # normaized to the simplex
    G = torch.diag((torch.sum(G, 1))**(-1)) @ G

    # G-G distances
    distances = torch.zeros(Ns,Ns)
    for i in range(Ns-1):
        for j in range(i+1, Ns):
            distances[i,j] = torch.linalg.norm(G[i] - G[j])
    distances = distances + distances.transpose(0,1) + 2*torch.diag(torch.ones(Ns))
    minDis = torch.min(distances, 1).values
    
    return minDis/2

def NormalG(G:torch.Tensor
            ):
    """normaized to the simplex"""
    G = torch.diag((torch.sum(G, 1))**(-1)) @ G
    return G

## this one is used to get CD
class virtualFP(nn.Module):
    def __init__(self,Nr):
        super(virtualFP, self).__init__()
        self.D = nn.Parameter(torch.ones(Nr)) # not sure if devided by sqrt(Nr)

    def forward(self, C):
        output = C @ torch.diag(self.D)
        return output
    


def Inew(NG:torch.Tensor, # growth rates on simplex
         CD:torch.Tensor, # consumption rates rescaled
         GGdiss:torch.Tensor # Ns-dim vector G-G distances
        ):
    # normaized to the simplex
    CD = torch.diag((torch.sum(CD, 1))**(-1)) @ CD

    GCdiss = torch.linalg.norm(NG - CD, dim=1)

    Is = GCdiss/GGdiss

    return Is.mean(), Is.std()

def trainI(G:torch.Tensor,
           C:torch.Tensor,
           lr = 0.1,
           epochs:int = 1000
           ) -> tuple[float,float]:
    """input G and C, try to get E(G,C) and the std"""
    Nr = G.shape[1]

    model = virtualFP(Nr)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    NG = NormalG(G)
    GGdiss = Heteros(G)

    model.train()

    losses = torch.zeros(epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()

        CD = model(C)
        loss, _ = Inew(NG,CD,GGdiss)

        loss.backward()
        optimizer.step()

        losses[epoch] = loss.item()

    if ((losses[-50:]).mean() - losses[-1]).abs() >= 0.01:
        print("Warning convergence!")

    model.eval()
    with torch.no_grad():
        CD = model(C)
        Emean, Estd = Inew(NG,CD,GGdiss)

    return Emean.item(), Estd.item()