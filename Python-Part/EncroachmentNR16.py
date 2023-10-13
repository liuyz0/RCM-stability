# to calculate the new Encroachment which has mean and std

# no need to sample, read file directly.

import torch
import math
from newEGCcalculator import trainI
from scipy.io import loadmat, savemat
import time
from pathlib import Path

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# input parameters
Nr = 64
Ns_span = range(8,96 + 1)

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
    for j in range(num_rho):
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

        Emean, Estd = trainI(G,C)
        Outputs[2,:,i,j] = Emean*torch.ones(num_try)
        
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
            Outputs[0,k,i,j] = len(E_J[E_J >= 1.0e-6])/Ns # Fraction of Unstable modes of the real Jacobian
            E_GC = torch.linalg.eigvals(- G @ C.transpose(0,1)).real
            Outputs[1,k,i,j] = len(E_GC[E_GC >= 1.0e-6])/Ns

    print('Progress %d%%, (%s)' % ((i+1)/len(Ns_span) * 100, timeSince(tik)))

datapath = Path('./data/').expanduser()
tensorfile = {'Outputs': Outputs, 'Nr': Nr, 'Ns_span': Ns_span, 'rho_span': rho_span}
torch.save(tensorfile, datapath/'torch_testNR64.pt')

tensordic = {"allC":Csample.numpy(), "allG": Gsample.numpy(),"allSstar": Ssample.numpy(), 
             "allRstar": Rsample.numpy(), "rho_span": rho_span.numpy(), 
             "Ns_span": Ns_span, "Nr": Nr, "num_try": num_try}
savemat("./data/allsamplesNR64.mat", tensordic)