#!/bin/bash

#SBATCH -o simulation_threads.txt
#SBATCH --nodes=1 # Number of node
#SBATCH --ntasks=1 # Number of tasks
#SBATCH -c 48 # 48 CPUs

# Initialize the module command first source
source /etc/profile

# Load cuda and Julia Module
module load julia/1.8.5

# Call your script as you would from the command line
julia --threads 48 SimulationThread.jl