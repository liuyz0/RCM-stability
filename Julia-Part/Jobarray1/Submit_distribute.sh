#!/bin/bash

#SBATCH -o Distributed_Simulation.log-%j
#SBATCH -n 30 # Number of tasks
#SBATCH -c 8 # CPUs each task

# Initialize the module command first source
source /etc/profile

# Load cuda and Julia Module
module load julia/1.8.5

# Call your script as you would from the command line
julia Simulation_distrv1.jl