## use scipy to solve!

import torch
from scipy.optimize import fsolve
import numpy as np

def fixedpoint(x,*para):
    (G_sub, C_sub, delta_sub, gamma) = para # which should be .numpy!!!
    Ns_sub = G_sub.shape[0]
    '''
    x[:Ns_sub]: species, the left are resources.  
    '''
    y = np.zeros(x.shape)
    y[:Ns_sub] = x[Ns_sub:] @ G_sub.T - delta_sub
    y[Ns_sub:] = x[Ns_sub:] * (x[:Ns_sub] @ C_sub) - gamma
    return y

def fixedpointLin(x,*para):
    (G_sub, C_sub, delta_sub, l, kappa) = para # which should be .numpy!!!
    Ns_sub = G_sub.shape[0]
    '''
    x[:Ns_sub]: species, the left are resources.  
    '''
    y = np.zeros(x.shape)
    y[:Ns_sub] = x[Ns_sub:] @ G_sub.T - delta_sub
    y[Ns_sub:] = x[Ns_sub:] * (x[:Ns_sub] @ C_sub) - l * (kappa - x[Ns_sub:])
    return y

def fixedpointLog(x,*para):
    (G_sub, C_sub, delta_sub, g, K) = para # which should be .numpy!!!
    Ns_sub = G_sub.shape[0]
    '''
    x[:Ns_sub]: species, the left are resources.  
    '''
    y = np.zeros(x.shape)
    y[:Ns_sub] = x[Ns_sub:] @ G_sub.T - delta_sub
    y[Ns_sub:] = ((x[:Ns_sub] @ C_sub) - g * (K - x[Ns_sub:])) * x[Ns_sub:]
    return y



def FindAltSS(G,C,Sstar,Rstar,num_ic = 10):
    """
    Given the number of subcommunity species, find all possible Alt SS, and output their diversity
    num_ic number of initial conditions in each case
    """

    Ns = G.size(dim=0)
    Nr = G.size(dim=1)
    delta = Rstar @ G.transpose(0,1)
    gamma = Rstar * (Sstar @ C)

    # initialize the output
    AltSSdiv = torch.tensor([])
    solutions = []
    sur_spe_name = []

    # determine whether Ns is odd
    even = (Ns % 2) == 0
    # if even we go from 0 to Ns/2 - 1, and deal with Ns/2 independently
    # else, we go from 0 to int(Ns/2)

    # Ns_sub what is the diversity of sub-community

    Ns_sub_span = range(int(Ns/2)+1)

    for Ns_sub in Ns_sub_span:
        sub_cmbins = torch.combinations(torch.tensor(range(Ns)), r=Ns_sub)
        num_cmbins = sub_cmbins.size(dim=0)

        if num_cmbins != 0:
            cmbrange = range(num_cmbins)
        else:
            cmbrange = [0]

        for cmbins in cmbrange:
            "go through all possible combinations"
            if num_cmbins == 0:
                cmbins = []
            self_spe_name = sub_cmbins[cmbins]
            uniques, counts = torch.cat((self_spe_name, torch.tensor(range(Ns)))).unique(return_counts=True)
            other_spe_name = uniques[counts == 1]

            if Ns_sub == 0:
                other_span = [1]
            else:
                if (Ns_sub == int(Ns/2)) & even:
                    other_span = [0]
                else:
                    other_span = [0,1]

            for other in other_span:
            # two combinations
                if other == 1:
                    spe_name = other_spe_name
                    complement = self_spe_name
                else:
                    spe_name = self_spe_name
                    complement = other_spe_name

                G_sub = G[spe_name,:]
                C_sub = C[spe_name,:]

                delta_sub = delta[spe_name]

                para = (G_sub.numpy(), C_sub.numpy(), delta_sub.numpy(), gamma.numpy())
                sols = torch.tensor([])
                for ic in range(num_ic):
                    init = np.random.rand(len(spe_name)+Nr) * 5.0
                    npsol = fsolve(fixedpoint, init, args=para)
                    steady = np.allclose(fixedpoint(npsol,*para), np.zeros(len(spe_name)+Nr))
                    if steady:
                        sol = torch.tensor(npsol).type('torch.FloatTensor')
                        if torch.all(sol > 0): # feasible
                            invade = (sol[len(spe_name):] @ G.transpose(0,1) - delta)[complement]
                            if torch.all(invade < 0.0): # uninvadable
                                Jstar = torch.zeros(len(spe_name)+Nr,len(spe_name)+Nr)
                                Jstar[:len(spe_name),len(spe_name):] = torch.diag(sol[:len(spe_name)]) @ G_sub
                                Jstar[len(spe_name):,:len(spe_name)] = - torch.diag(sol[len(spe_name):]) @ C_sub.transpose(0,1)
                                Jstar[len(spe_name):,len(spe_name):] = - torch.diag(C_sub.transpose(0,1) @ sol[:len(spe_name)])

                                Eig_J = torch.linalg.eigvals(Jstar).real
                                if len(Eig_J[Eig_J >= 1.0e-6]) == 0:
                                    # stable!
                                    sols = torch.cat((sols, sol.unsqueeze(0)))
                
                # PCA for sols
                if sols.size() != torch.Size([0]):
                    prod = sols @ sols.transpose(0,1)
                    prod = torch.diag((torch.diag(prod)**(-1/2))) @ prod @ torch.diag((torch.diag(prod)**(-1/2)))
                    Eig_PCA = torch.linalg.eigvals(prod).real
                    num_AltSS = len(Eig_PCA[Eig_PCA >= 1.0e-4 * torch.max(Eig_PCA)])
                    AltSSdiv = torch.cat((AltSSdiv,len(spe_name)*torch.ones(num_AltSS)))
                    solutions.append(sols)
                    sur_spe_name.append(spe_name)
    
    return AltSSdiv, solutions, sur_spe_name



def FindAltSSLin(G,C,delta,l,kappa,num_ic = 10):
    """
    Given the number of subcommunity species, find all possible Alt SS, and output their diversity
    num_ic number of initial conditions in each case
    """

    Ns = G.size(dim=0)
    Nr = G.size(dim=1)
    #delta = Rstar @ G.transpose(0,1)
    #gamma = Rstar * (Sstar @ C)

    # initialize the output
    AltSSdiv = torch.tensor([])

    # determine whether Ns is odd
    even = (Ns % 2) == 0
    # if even we go from 0 to Ns/2 - 1, and deal with Ns/2 independently
    # else, we go from 0 to int(Ns/2)

    # Ns_sub what is the diversity of sub-community

    Ns_sub_span = range(int(Ns/2)+1)

    for Ns_sub in Ns_sub_span:
        sub_cmbins = torch.combinations(torch.tensor(range(Ns)), r=Ns_sub)
        num_cmbins = sub_cmbins.size(dim=0)

        if num_cmbins != 0:
            cmbrange = range(num_cmbins)
        else:
            cmbrange = [0]

        for cmbins in cmbrange:
            "go through all possible combinations"
            if num_cmbins == 0:
                cmbins = []
            self_spe_name = sub_cmbins[cmbins]
            uniques, counts = torch.cat((self_spe_name, torch.tensor(range(Ns)))).unique(return_counts=True)
            other_spe_name = uniques[counts == 1]

            if Ns_sub == 0:
                other_span = [1]
            else:
                if (Ns_sub == int(Ns/2)) & even:
                    other_span = [0]
                else:
                    other_span = [0,1]

            for other in other_span:
            # two combinations
                if other == 1:
                    spe_name = other_spe_name
                    complement = self_spe_name
                else:
                    spe_name = self_spe_name
                    complement = other_spe_name

                G_sub = G[spe_name,:]
                C_sub = C[spe_name,:]

                delta_sub = delta[spe_name]
                # G_sub, C_sub, delta_sub, l, kappa
                para = (G_sub.numpy(), C_sub.numpy(), delta_sub.numpy(), l.numpy(), kappa.numpy())
                sols = torch.tensor([])
                for ic in range(num_ic):
                    init = np.random.rand(len(spe_name)+Nr)
                    sol = torch.tensor(fsolve(fixedpointLin, init, args=para)).type('torch.FloatTensor')
                    if torch.all(sol > 1.0e-5): # feasible
                        invade = (sol[len(spe_name):] @ G.transpose(0,1) - delta)[complement]
                        if torch.all(invade < 0.0): # uninvadable
                            Jstar = torch.zeros(len(spe_name)+Nr,len(spe_name)+Nr)
                            Jstar[:len(spe_name),len(spe_name):] = torch.diag(sol[:len(spe_name)]) @ G_sub
                            Jstar[len(spe_name):,:len(spe_name)] = - torch.diag(sol[len(spe_name):]) @ C_sub.transpose(0,1)
                            Jstar[len(spe_name):,len(spe_name):] = - torch.diag(C_sub.transpose(0,1) @ sol[:len(spe_name)] + l)

                            Eig_J = torch.linalg.eigvals(Jstar).real
                            if len(Eig_J[Eig_J >= 1.0e-3]) == 0:
                                # stable!
                                sols = torch.cat((sols, sol.unsqueeze(0)))
                
                # PCA for sols
                if sols.size() != torch.Size([0]):
                    prod = sols @ sols.transpose(0,1)
                    prod = torch.diag((torch.diag(prod)**(-1/2))) @ prod @ torch.diag((torch.diag(prod)**(-1/2)))
                    Eig_PCA = torch.linalg.eigvals(prod).real
                    num_AltSS = len(Eig_PCA[Eig_PCA >= 1.0e-3])
                    AltSSdiv = torch.cat((AltSSdiv,len(spe_name)*torch.ones(num_AltSS)))
    
    return AltSSdiv


def FindAltSSLog(G,C,delta,g,K,num_ic = 10):
    """
    Given the number of subcommunity species, find all possible Alt SS, and output their diversity
    num_ic number of initial conditions in each case
    """

    Ns = G.size(dim=0)
    Nr = G.size(dim=1)
    #delta = Rstar @ G.transpose(0,1)
    #gamma = Rstar * (Sstar @ C)

    # initialize the output
    AltSSdiv = torch.tensor([])

    # determine whether Ns is odd
    even = (Ns % 2) == 0
    # if even we go from 0 to Ns/2 - 1, and deal with Ns/2 independently
    # else, we go from 0 to int(Ns/2)

    # Ns_sub what is the diversity of sub-community

    Ns_sub_span = range(int(Ns/2)+1)

    for Ns_sub in Ns_sub_span:
        sub_cmbins = torch.combinations(torch.tensor(range(Ns)), r=Ns_sub)
        num_cmbins = sub_cmbins.size(dim=0)

        if num_cmbins != 0:
            cmbrange = range(num_cmbins)
        else:
            cmbrange = [0]

        for cmbins in cmbrange:
            "go through all possible combinations"
            if num_cmbins == 0:
                cmbins = []
            self_spe_name = sub_cmbins[cmbins]
            uniques, counts = torch.cat((self_spe_name, torch.tensor(range(Ns)))).unique(return_counts=True)
            other_spe_name = uniques[counts == 1]

            if Ns_sub == 0:
                other_span = [1]
            else:
                if Ns_sub == int(Ns/2) & even:
                    other_span = [0]
                else:
                    other_span = [0,1]

            for other in other_span:
            # two combinations
                if other == 1:
                    spe_name = other_spe_name
                    complement = self_spe_name
                else:
                    spe_name = self_spe_name
                    complement = other_spe_name

                G_sub = G[spe_name,:]
                C_sub = C[spe_name,:]

                delta_sub = delta[spe_name]
                # (G_sub, C_sub, delta_sub, g, K)
                para = (G_sub.numpy(), C_sub.numpy(), delta_sub.numpy(), g.numpy(), K.numpy())
                sols = torch.tensor([])
                for ic in range(num_ic):
                    init = np.random.rand(len(spe_name)+Nr) # *100 try?
                    sol = torch.tensor(fsolve(fixedpointLog, init, args=para)).type('torch.FloatTensor')
                    if torch.all(sol[len(spe_name):] > -1.0e-5) & torch.all(sol[:len(spe_name)] > 1.0e-5): # feasible
                        # dieresource = (sol[len(spe_name):] <= 1.0e-5).nonzero().squeeze(1) 
                        invaderes = (- sol[:len(spe_name)] @ C_sub + g * (K - sol[len(spe_name):]) )[sol[len(spe_name):] <= 1.0e-5]
                        invadespe = (sol[len(spe_name):] @ G.transpose(0,1) - delta)[complement]

                        if torch.all(invadespe < 0.0) & torch.all(invaderes < 0.0): # uninvadable
                            Jstar = torch.zeros(len(spe_name)+Nr,len(spe_name)+Nr)
                            Jstar[:len(spe_name),len(spe_name):] = torch.diag(sol[:len(spe_name)]) @ G_sub
                            Jstar[len(spe_name):,:len(spe_name)] = - torch.diag(sol[len(spe_name):]) @ C_sub.transpose(0,1)
                            Jstar[len(spe_name):,len(spe_name):] = - torch.diag(C_sub.transpose(0,1) @ sol[:len(spe_name)]) + torch.diag(g * (K - 2*sol[len(spe_name):]))

                            Eig_J = torch.linalg.eigvals(Jstar).real
                            if len(Eig_J[Eig_J >= 1.0e-3]) == 0:
                                # stable!
                                sols = torch.cat((sols, sol.unsqueeze(0)))
                
                # PCA for sols
                if sols.size() != torch.Size([0]):
                    prod = sols @ sols.transpose(0,1)
                    prod = torch.diag((torch.diag(prod)**(-1/2))) @ prod @ torch.diag((torch.diag(prod)**(-1/2)))
                    Eig_PCA = torch.linalg.eigvals(prod).real
                    num_AltSS = len(Eig_PCA[Eig_PCA >= 1.0e-3])
                    AltSSdiv = torch.cat((AltSSdiv,len(spe_name)*torch.ones(num_AltSS)))
    
    return AltSSdiv