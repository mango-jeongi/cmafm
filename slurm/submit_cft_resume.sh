#!/bin/bash
#SBATCH --job-name=cft_resume
#SBATCH --output=logs/cft_resume_%A_%a.out
#SBATCH --error=logs/cft_resume_%A_%a.err
#SBATCH --array=1-10               # Resumes all 10 seeds
#SBATCH --partition=gpu            # Set to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00            # Plenty of time to finish the remaining ~20 epochs
#SBATCH --get-user-env             # Load your login environment

# No module load needed as venv is self-contained
source .venv/bin/activate

# Disable W&B logging for cluster stability
export WANDB_MODE=disabled

mkdir -p logs

cd cft_engine

# Explicitly point to the checkpoint. YOLOv5 automatically pulls all args/hyperparameters from the .pt file
python train.py \
    --resume ../runs/cft_m3fd_flir/seed_$SLURM_ARRAY_TASK_ID/cft_baseline_run/weights/last.pt
