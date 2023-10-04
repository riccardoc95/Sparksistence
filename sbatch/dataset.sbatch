#!/bin/sh
#SBATCH --nodes=1
#SBATCH --output="slurm-%j.out"

# Note defaults of MAGPIE_STARTUP_TIME & MAGPIE_SHUTDOWN_TIME, this
# timelimit should be a fair amount larger than them combined.
#SBATCH --time=10:00:00

# Job name.  This will be used in naming directories for the job.
#SBATCH --job-name=astro_test

# Partition to launch job in
#SBATCH --partition=dss

## SLURM Values
# Generally speaking, don't touch the following, misc other configuration

#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G


srun python ../dataset.py -dd 100 -id 10000