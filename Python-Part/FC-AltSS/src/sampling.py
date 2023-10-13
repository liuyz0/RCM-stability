# sample G, C, R, S needed
import torch
import math

def GCRS(Ns:int,
         Nr:int,
         rho:float,
         upp:float = 1.0,
         low:float = 0.1) -> tuple:
    """
    given number of species and resources,
    get G, C, R, and S
    """
    sample = torch.rand(Ns,2,Nr)
    L = torch.tensor([[1, 0],
                    [rho, math.sqrt(1-rho**2)]]) # Cholesky decomposition

    sample = torch.matmul(L,sample)

    G = sample[0:Ns,0]
    C = sample[0:Ns,1] # C has not been pushed away

    C = C @ torch.diag(0.1+ 0.9*torch.rand(Nr))

    Sstar = low + (upp - low)*torch.rand(Ns)
    Rstar = low + (upp - low)*torch.rand(Nr)

    return G, C, Rstar, Sstar
