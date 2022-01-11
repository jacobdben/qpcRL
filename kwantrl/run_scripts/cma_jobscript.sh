#!/bin/bash


#SBATCH --job-name=disorder_test
#SBATCH --partition=cmt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --threads-per-core=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=100Gb

python run.py
