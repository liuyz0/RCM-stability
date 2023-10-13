#!/bin/bash

# run with: LLsub ./Submit_LLsub.sh [1,48,1]

# Initialize Modules
source /etc/profile

# Load Julia Module
module load julia/1.6.1

echo "My SLURM_ARRAY_TASK_ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

julia Simulation.jl $LLSUB_RANK $LLSUB_SIZE