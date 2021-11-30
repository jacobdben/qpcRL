#!/bin/bash


#SBATCH --job-name=NN_data_generation
#SBATCH --partition=cmt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --threads-per-core=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=100Gb
#SBATCH --output=../slurm_output/job-%j.out
#SBATCH --error=../slurm_output/job-%j.err

python fourier_modes_run.py
