#!/bin/bash
#SBATCH --job-name=spc_prepare
#SBATCH --output=logs/spc_prepare_%j.out
#SBATCH --error=logs/spc_prepare_%j.err
#SBATCH --partition=genai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
VENV=${SPC_VENV:-$PWD/.venv}
UV=${UV_BIN:-$HOME/.local/bin/uv}
export UV_PROJECT_ENVIRONMENT="$VENV"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export WANDB_MODE=disabled
uvpy() { "$UV" run --no-project --python "$VENV/bin/python" "$@"; }
mkdir -p results logs
bash scripts/setup.sh
uvpy spc_kit/tools/init_weights.py --engine cft_engine --src weights/yolov5l_v5.0.pt --out weights/spc_init
uvpy spc_kit/tools/make_splits.py --merged "${SPC_MERGED:-$HOME/data/merged_txt}" --out data/spc_split
uvpy spc_kit/tools/make_subsets.py --split data/spc_split --out data/spc_subsets
uvpy spc_kit/tools/verify_preflight.py
