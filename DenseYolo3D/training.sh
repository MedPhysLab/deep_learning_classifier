#!/bin/bash
#SBATCH --job-name=test_denseyolo3d
#SBATCH --output=python_job_output.out
#SBATCH --error=python_job_error.err
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

conda activate duke_3d

python main.py
