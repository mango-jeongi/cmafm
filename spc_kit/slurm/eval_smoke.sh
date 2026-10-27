#!/bin/bash
#SBATCH --job-name=spc_smoke_eval
#SBATCH --output=logs/spc_smoke_eval_%j.out
#SBATCH --error=logs/spc_smoke_eval_%j.err
#SBATCH --partition=genai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
VENV=${SPC_VENV:-$PWD/.venv}
UV=${UV_BIN:-$HOME/.local/bin/uv}
export UV_PROJECT_ENVIRONMENT="$VENV"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export WANDB_MODE=disabled
uvpy() { "$UV" run --no-project --python "$VENV/bin/python" "$@"; }
mkdir -p results logs
uvpy spc_kit/tools/eval_runs.py --engine cft_engine \
  --runs "runs/spc_smoke/job_${SPC_SMOKE_JOB:?smoke training array ID required}" \
  --subsets data/spc_subsets --only m3fd_all --expected-runs 1 \
  --out results/spc_smoke_eval.csv
