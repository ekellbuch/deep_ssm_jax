#!/bin/bash
#SBATCH --job-name=jax
#SBATCH --error=jax%j_%a.err         
#SBATCH --output=jax%j_%a.out     
#SBATCH --time=23:59:59                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint='GPU_SKU:A100_SXM4&GPU_MEM:80GB'
#SBATCH --mail-type=ALL

/scratch/users/xavier18/venvs/deep_ssm_jax/bin/python3 example.py $@