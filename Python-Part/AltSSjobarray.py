# Brutal force looking for alternative stable states.

# packages and funcs
import torch
from pathlib import Path
from FindAltSSfuncs import FindAltSS
#import sys # needed on supercloud for passing task_id

# import data
datapath = Path('./data/').expanduser()
loaded = torch.load(datapath/'AltSSsamples.pt')
C_span = loaded['C_span']
G_span = loaded['G_span']
R_span = loaded['R_span']
S_span = loaded['S_span']


# Grab the argument that is passed in
# This is the index into fnames for this process
#task_id = parse(Int,ARGS[1])
#num_tasks = parse(Int,ARGS[2]) # wrong this is for julia! 
task_id = 0

i = task_id // 4 # which correlation should we use 
j = task_id % 4 # which Ns_sub should we use

C = C_span[i]
G = G_span[i]
Rstar = R_span[i]
Sstar = S_span[i]

if j == 0:
    Ns_sub_range = [0,1,2,3]
else:
    if j == 1:
        Ns_sub_range = [4]
    else:
        if j == 2:
            Ns_sub_range = [5]
        else:
            Ns_sub_range = [6]

AltSSdiv = torch.tensor([])
for Ns_sub in Ns_sub_range:
    newAlt = FindAltSS(Ns_sub,G,C,Sstar,Rstar)
    AltSSdiv = torch.cat((AltSSdiv,newAlt))

filename = 'AltSSdiv'+str(task_id)+'.pt'
torch.save({'AltSSdiv': AltSSdiv}, datapath/filename)