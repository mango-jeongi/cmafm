#!/bin/bash
#SBATCH --job-name=cmafm_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --partition=genai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5

source .venv/bin/activate
export WANDB_MODE=disabled
mkdir -p logs

# We train the 6 variants here. 
# "late_fusion" requires no training and will be evaluated separately later!
VARIANTS=("rgb_only" "thermal_only" "early_fusion" "dual_no_attn" "simplified_cmafm" "full")
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "Starting Ablation Variant: $VARIANT"

cd src/fusion
python run_ablation.py --variants $VARIANT --epochs 30 --batch-size 8
