# sample all the parameters G, C, R^*, S^* needed.

import torch
import math
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat

#torch.set_default_tensor_type(torch.float64)

Nr = 32
Ns_span = range(4,48 + 1)
num_try = 10

num_rho = 100 # must be even
rho_span = torch.zeros(num_rho)
rho_span[:int(num_rho/2)] = torch.linspace(0, .85, steps=int(num_rho/2))
rho_span[int(num_rho/2):] = torch.linspace(.87, 1, steps=int(num_rho/2))

Csample = torch.zeros(len(Ns_span),len(rho_span),max(Ns_span),Nr)
Gsample = torch.zeros(len(Ns_span),len(rho_span),max(Ns_span),Nr)
Rsample = torch.zeros(len(Ns_span),len(rho_span),num_try,Nr)
Ssample = torch.zeros(len(Ns_span),len(rho_span),num_try,max(Ns_span))

for i in range(len(Ns_span)):
    Ns = Ns_span[i]
    for j in range(len(rho_span)):
        rho = rho_span[j]

        # sampling 
        sample = torch.rand(Ns,2,Nr)
        L = torch.tensor([[1, 0],
                        [rho, math.sqrt(1-rho**2)]]) # Cholesky decomposition

        sample = torch.matmul(L,sample)

        Gsample[i,j,0:Ns,:] = sample[0:Ns,0]
        C = sample[0:Ns,1] # C has not been pushed away

        Csample[i,j,0:Ns,:] = C @ torch.diag(0.01+ 0.99*torch.rand(Nr))
        for k in range(num_try):
            Ssample[i,j,k,:Ns] = 0.01 + 0.99*torch.rand(Ns)
            Rsample[i,j,k,:] = 0.01 + 0.99*torch.rand(Nr)

tensordic = {"allC":Csample.numpy(), "allG": Gsample.numpy(),"allSstar": Ssample.numpy(), 
             "allRstar": Rsample.numpy(), "rho_span": rho_span.numpy(), 
             "Ns_span": Ns_span, "Nr": Nr, "num_try": num_try}
savemat("./data/allsamples.mat", tensordic)