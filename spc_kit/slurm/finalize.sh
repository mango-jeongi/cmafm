#!/bin/bash
#SBATCH --job-name=spc_finalize
#SBATCH --output=logs/spc_finalize_%j.out
#SBATCH --error=logs/spc_finalize_%j.err
#SBATCH --partition=primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
VENV=${SPC_VENV:-$PWD/.venv}
UV=${UV_BIN:-$HOME/.local/bin/uv}
export UV_PROJECT_ENVIRONMENT="$VENV"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export WANDB_MODE=disabled
"$UV" run --no-project --python "$VENV/bin/python" spc_kit/tools/finalize_checkpoints.py "$@"
