# Brute-force looking for alternative stable states.

# packages and funcs
import torch
from pathlib import Path
from FindAltSSfuncsv1 import FindAltSS
import sys # needed on supercloud for passing task_id

# import data
datapath = Path('./data/').expanduser()
loaded = torch.load(datapath/'AltSSsamplesv1.pt')
C_span = loaded['C_span']
G_span = loaded['G_span']
R_span = loaded['R_span']
S_span = loaded['S_span']

num_test = C_span.shape[1]

# Grab the argument that is passed in
# This is the index into fnames for this process
#task_id = int(sys.argv[1])
#num_tasks = int(sys.argv[2])
task_id = 0

AltSSdiv = torch.tensor([])

i = task_id # which correlation should we use 
for j in range(num_test):  # which test community

    C = C_span[i,j]
    G = G_span[i,j]
    Rstar = R_span[i,j]
    Sstar = S_span[i,j]

    AltSSdiv = torch.cat((AltSSdiv,torch.tensor([0.5]))) # 0.5 is the flag to identify new test community
    newAlt = FindAltSS(G,C,Sstar,Rstar)
    AltSSdiv = torch.cat((AltSSdiv, newAlt))

filename = 'v1AltSSdiv'+str(task_id)+'.pt'
torch.save({'AltSSdiv': AltSSdiv}, datapath/filename)