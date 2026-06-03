#!/bin/bash
#SBATCH --job-name=rgbt_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --array=1-10               # Run 10 seeds
#SBATCH --partition=gpu            # Set to your cluster's GPU partition (see .env SLURM_PARTITION)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # 1 GPU (DataParallel with bs=8 causes 6x slowdown)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00            # Single GPU finishes 30 epochs in ~3h
#SBATCH --get-user-env             # Load your login environment

# No module load needed as venv is self-contained
source .venv/bin/activate

# Disable W&B logging for cluster stability (no internet on compute nodes)
export WANDB_MODE=disabled

# Create log directory if it doesn't exist
mkdir -p logs

# YAML config paths are handled by setup.sh patching
# Enter the engine directory
cd cft_engine

# --- CONFIGURATION SELECTION ---
# SCENARIO A: M3FD + FLIR Train, M3FD ONLY Val (The 85.9% Benchmark)
DATA_CONFIG="../cft_engine/data/M3FD_FLIR.yaml"

# SCENARIO B: M3FD ONLY Train, M3FD ONLY Val (Safety Fallback)
# DATA_CONFIG="../cft_engine/data/M3FD_ONLY.yaml"
# -------------------------------

python train.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --weights ../weights/CMAFM_Pretrained.pt \
    --data $DATA_CONFIG \
    --cfg ../cft_engine/models/yolov5l_cmafm_M3FD.yaml \
    --hyp ../data/hyp.scratch.yaml \
    --epochs 30 \
    --batch-size 8 \
    --workers 8 \
    --project ../runs/seed_$SLURM_ARRAY_TASK_ID \
    --name cmafm_hpc_run
