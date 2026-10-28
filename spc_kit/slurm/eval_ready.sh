#!/bin/bash
#SBATCH --job-name=spc_eval_ready
#SBATCH --output=logs/spc_eval_ready_%j.out
#SBATCH --error=logs/spc_eval_ready_%j.err
#SBATCH --partition=genai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
VENV=${SPC_VENV:-$PWD/.venv}
UV=${UV_BIN:-$HOME/.local/bin/uv}
export UV_PROJECT_ENVIRONMENT="$VENV"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export WANDB_MODE=disabled
CONFIGS=rgb_only,thermal_only,dual_add,ctrl_concat_1x1,cmafm_channel_only,cmafm_spatial_only,cmafm_no_alpha
"$UV" run --no-project --python "$VENV/bin/python" spc_kit/tools/eval_runs.py \
    --engine cft_engine --runs runs/spc --subsets data/spc_subsets --out results/eval.csv \
    --configs "$CONFIGS" --expected-runs 35 --require-complete
