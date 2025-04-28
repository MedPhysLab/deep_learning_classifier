#!/bin/bash
#SBATCH --job-name=salva_immagini
#SBATCH --output=python_job_output.out
#SBATCH --error=python_job_error.err
#SBATCH --time=05:00:00
#SBATCH --mem=64GB
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

conda activate duke_3d

python elabora_immagini.py
