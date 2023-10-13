# functions needed

import torch
import numpy as np
import math

def SampleSimplex(Ns:int, # number of species
                  Nr:int, # number of resources
                  rho:float, # correlation in smapling G and C
                  comp:float # how concentrate it is on simplex
                  ):
    sample = torch.rand(Ns,2,Nr-1).sort().values

    sample1 = torch.zeros(Ns,2,Nr)
    sample1[:,:,1:Nr] = sample

    sample2 = torch.ones(Ns,2,Nr)
    sample2[:,:,0:Nr-1] = sample

    props = sample2 - sample1
    #distocen = props - torch.ones(Ns,2,Nr)/Nr

    #props = comp*distocen + torch.ones(Ns,2,Nr)/Nr

    L = torch.tensor([[1, 0],
                 [rho, math.sqrt(1-rho**2)]]) # Cholesky decomposition
    
    L = (torch.sum(L,dim=1)**(-1)).unsqueeze(1) * L # to simplex

    props = torch.matmul(L,props)
    distocen = props - torch.ones(Ns,2,Nr)/Nr
    G = comp*distocen[:,0] + torch.ones(Ns,Nr)/Nr
    C = comp*distocen[:,1] + torch.ones(Ns,Nr)/Nr

    #G = props[:,0]
    #C = props[:,1]

    Ss = 0.01 + 0.99*torch.rand(Ns)

    gamma = Ss @ C

    return G, C, Ss, gamma

def Jacobian(G:torch.Tensor, # growth rates
             C:torch.Tensor, # consumption rates
             Ss:torch.Tensor, # species abundances S^*
             criterion = 1e-6 # larger than which we think is unstable
             ):
    Ns, Nr = G.shape[:2]
    Jstar = torch.zeros(Ns+Nr,Ns+Nr)
    Jstar[0:Ns,Ns:Ns+Nr] = Ss.unsqueeze(1) * G
    Jstar[Ns:Ns+Nr,0:Ns] = - C.transpose(0,1)
    Jstar[Ns:Ns+Nr,Ns:Ns+Nr] = - torch.diag(C.transpose(0,1) @ Ss)

    E_J = torch.linalg.eigvals(Jstar).real
    #critical = torch.max(criterion*torch.abs(E_J))
    umodel = len(E_J[E_J >= criterion])/Ns

    return umodel


def IntMatrix(G:torch.Tensor, # growth rates
             C:torch.Tensor, # consumption rates
             Ss:torch.Tensor, # species abundances S^*
             ):
    # interaction matrix:
    IM = (G * ((Ss @ C)**(-1)).unsqueeze(0)) @ C.transpose(0,1)

    # normalize
    IM = ((IM.diag())**(-0.5)).unsqueeze(1) * IM * ((IM.diag())**(-0.5)).unsqueeze(0)

    Mask = torch.ones(IM.size()).triu(diagonal=1)==1
    UTri = IM[Mask]
    LTri = IM.transpose(0,1)[Mask]
    
    ndiag = torch.cat((UTri,LTri))
    mean = ndiag.mean()
    sigma = ndiag.std()

    corr = torch.corrcoef(torch.cat((UTri.unsqueeze(0),LTri.unsqueeze(0))))[0,1]
    return mean, sigma, corr

def NormalG(G):
    G = torch.diag((torch.sum(G, 1))**(-1)) @ G
    return G

def Heteros(G:torch.Tensor
            ) -> torch.Tensor:
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
    #minDiss = distances.sort().values
    #minDis = (minDiss[:,0] + minDiss[:,1])/2
    
    return minDis/2 # a Ns tensor

def Inew(G:torch.Tensor, # growth rates
         C:torch.Tensor, # consumption rates
        ):
    # normaized to the simplex
    C = torch.diag((torch.sum(C, 1))**(-1)) @ C
    G = torch.diag((torch.sum(G, 1))**(-1)) @ G

    GGdiss = Heteros(G)
    #GGdis = GGdiss.mean()

    GCdiss = torch.linalg.norm(G - C, dim=1)

    Is = GCdiss/GGdiss
    #Is = GCdiss/GGdis

    return Is.mean(), Is.std()