#!/bin/bash
#SBATCH --job-name=smnist
#SBATCH --error=smnist%j_%a.err         
#SBATCH --output=smnist%j_%a.out     
#SBATCH --time=23:59:59                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint='GPU_SKU:A100_SXM4&GPU_MEM:80GB'
#SBATCH --mail-type=ALL

# Environment setup
module load cuda/12.4

# Activate conda environment
source /scratch/users/xavier18/miniconda3/bin/activate deep_ssm_jax

python example.py