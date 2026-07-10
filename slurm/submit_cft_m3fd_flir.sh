#!/bin/bash
#SBATCH --job-name=cft_m3fd
#SBATCH --output=logs/cft_m3fd_%A_%a.out
#SBATCH --error=logs/cft_m3fd_%A_%a.err
#SBATCH --array=1-10               # Run 10 seeds
#SBATCH --partition=gpu            # Set to your cluster's GPU partition (see .env SLURM_PARTITION)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00            # 6h wall-time for 100 epochs on M3FD+FLIR (7485 images)
#SBATCH --get-user-env             # Load your login environment

# No module load needed as venv is self-contained
source .venv/bin/activate

# Disable W&B logging for cluster stability (no internet on compute nodes)
export WANDB_MODE=disabled

# Create log directory if it doesn't exist
mkdir -p logs

# Enter the engine directory
cd cft_engine

# CFT baseline: train on M3FD+FLIR unified, validate on M3FD only
# Uses the original GPT (transformer) fusion blocks, not CMAFM
python train.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --weights ../weights/yolov5l.pt \
    --data ../cft_engine/data/M3FD_FLIR_val_m3fd.yaml \
    --cfg ../cft_engine/models/transformer/yolov5l_cft_M3FD_FLIR.yaml \
    --hyp ../data/hyp.scratch.yaml \
    --epochs 100 \
    --batch-size 8 \
    --workers 8 \
    --project ../runs/cft_m3fd_flir/seed_$SLURM_ARRAY_TASK_ID \
    --name cft_baseline_run
