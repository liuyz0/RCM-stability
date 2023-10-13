import torch
import torch.nn as nn
import torch.optim as optim
import math

class SolveAltSS(nn.Module):
    def __init__(self,Ns_sub,Nr,Sstar,Rstar):
        super(SolveAltSS, self).__init__()
        self.Ns = Ns_sub
        self.Nr = Nr
        self.Star = nn.Parameter(torch.cat((Sstar,Rstar)))

    def forward(self, G, C, delta, gamma):
        dudt = torch.zeros_like(self.Star)
        dudt[:self.Ns] = self.Star[:self.Ns]*(self.Star[self.Ns:] @ G.transpose(0,1) - delta)
        dudt[self.Ns:] = gamma - self.Star[self.Ns:] * (self.Star[:self.Ns] @ C)

        return dudt, self.Star
    
def Lossfun(dudt, Star):
    loss = torch.norm(dudt) + 100*torch.sum(nn.functional.relu(-(Star - 1.0e-5)))
    return loss

def FindAltSS(Ns_sub,G,C,Sstar,Rstar):
    """
    Given the number of subcommunity species, find all possible Alt SS, and output their diversity
    Ns_sub in [1,2,...,6]
    Nr = 12 = Ns
    """
    Ns = G.size(dim=0)
    Nr = G.size(dim=1)
    delta = Rstar @ G.transpose(0,1)
    gamma = Rstar * (Sstar @ C)
    
    sub_cmbins = torch.combinations(torch.tensor(range(Ns)), r=Ns_sub) # keep r lower than 5
    num_cmbins = sub_cmbins.size(dim=0)

    # initialize the output
    AltSSdiv = torch.tensor([])
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

        if Ns_sub == 6:
            other_span = [0]
        else:
            if Ns_sub == 0:
                other_span = [1]
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
            S_sub = Sstar[spe_name]

            delta_sub = delta[spe_name]

            optisolver = SolveAltSS(len(spe_name),Nr,S_sub,Rstar)
            optimizer = optim.Adam(optisolver.parameters())

            lossv = 1
            losses = []
            epoch = 1
            Star = torch.zeros(len(spe_name)+Nr)

            while lossv > 0.005:

                optimizer.zero_grad()

                dudt, Star = optisolver(G_sub, C_sub, delta_sub, gamma)
                loss = Lossfun(dudt,Star)

                loss.backward()

                optimizer.step()

                lossv = loss.item()
                Star = Star.detach()
                losses.append(lossv)
                if epoch >= 50000:
                    break

                epoch = epoch + 1

            if lossv <= 0.005:
                # feasible
                invade = (Star[len(spe_name):] @ G.transpose(0,1) - delta)[complement]
                if len(invade[invade>0]) < 1:
                    # uninvadable
                    Jstar = torch.zeros(len(spe_name)+Nr,len(spe_name)+Nr)
                    Jstar[:len(spe_name),len(spe_name):] = torch.diag(Star[:len(spe_name)]) @ G_sub
                    Jstar[len(spe_name):,:len(spe_name)] = - torch.diag(Star[len(spe_name):]) @ C_sub.transpose(0,1)
                    Jstar[len(spe_name):,len(spe_name):] = - torch.diag(C_sub.transpose(0,1) @ Star[:len(spe_name)])

                    Eig_J = torch.linalg.eigvals(Jstar).real
                    if len(Eig_J[Eig_J >= 1.0e-3]) == 0:
                        # stable!
                        AltSSdiv = torch.cat((AltSSdiv,torch.tensor([len(spe_name)])))

    return AltSSdiv