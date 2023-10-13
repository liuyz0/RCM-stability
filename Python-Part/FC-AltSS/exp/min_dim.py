# try to find the minimum dimesion of fully coexisting alternative stable states
import sys
sys.path.append('../src')
#sys.path.append('../data')
from pathlib import Path
import torch
from solving import FindAltSS
from sampling import GCRS
import time

# Grab the argument that is passed in
# This is the index into fnames for this process
#task_id = int(sys.argv[1])
#num_tasks = int(sys.argv[2])
task_id = 7
num_tasks = 48

num_rho = num_tasks # must be even
rho_span = torch.zeros(num_rho)
rho_span[:int(num_rho/2)] = torch.linspace(0, .85, steps=int(num_rho/2))
rho_span[int(num_rho/2):] = torch.linspace(.86, 1, steps=int(num_rho/2))

rho = rho_span[num_tasks - 1 - task_id]

num_N = 9
num_samp = 10 # same para sample 10 diff communities
# iterations over Ns, Nr
find = False

for j in range(num_N):
    Ns = j + 3
    Nr = Ns
    if find:
        break
    
    start_time = time.time()
    for k in range(num_samp):
        G, C, Rs, Ss = GCRS(Ns, Nr, rho)
        newSS = FindAltSS(G,C,Ss,Rs)
        if len(newSS) > 1:
            # alternative
            if Ns in newSS:
                # coexisting
                find = True
                community = {"G": G,
                             "C": C,
                             "Rs": Rs,
                             "Ss": Ss,
                             "div": newSS}
                datapath = Path('../data/')
                filename = 'min_dim'+str(task_id)+'.pt'
                torch.save(community, datapath/filename)
                print("Success!")

        if find:
            break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int((elapsed_time - (elapsed_mins * 60))*100)/100
    print(f'Complete Ns = {Ns} | Time: {elapsed_mins}m {elapsed_secs}s')

if not find:
    print('No community satisfies the condition')
