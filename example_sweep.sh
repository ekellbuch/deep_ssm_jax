#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --error=log/sweep%j_%a.err         
#SBATCH --output=log/sweep%j_%a.out     
#SBATCH --time=47:59:59                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint='GPU_SKU:A100_SXM4&GPU_MEM:80GB'
#SBATCH --mail-type=ALL

# Activate conda environment
source /scratch/users/xavier18/venvs/deep_ssm_jax/bin/activate

# Get Sweep ID from the command line argument
SWEEP_ID=$1

# Check if a Sweep ID was provided
if [ -z "$SWEEP_ID" ]; then
  echo "Error: No Sweep ID provided. Usage: sbatch wandb_agent.sbatch <SWEEP_ID>"
  exit 1
fi

# Construct the full Sweep Identifier
FULL_SWEEP_ID="xavier_gonzalez/nonlinear_ssm/$SWEEP_ID"

# Run the W&B agent
wandb agent $FULL_SWEEP_ID