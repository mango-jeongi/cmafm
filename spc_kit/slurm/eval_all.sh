#!/bin/bash
# After all training jobs finish: FLOPs+latency, evaluation on every subset, alpha statistics, tables.
#SBATCH --job-name=spc_eval
#SBATCH --output=logs/spc_eval_%j.out
#SBATCH --partition=genai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --get-user-env
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
VENV=${SPC_VENV:-$PWD/.venv}
UV=${UV_BIN:-$HOME/.local/bin/uv}
export UV_PROJECT_ENVIRONMENT="$VENV"
export WANDB_MODE=disabled
uvpy() { "$UV" run --no-project --python "$VENV/bin/python" "$@"; }
mkdir -p results logs
if [[ ! -s results/flops_640.json ]]; then
  uvpy spc_kit/tools/profile_models.py --engine cft_engine --out results/flops_640.json --latency
fi
uvpy spc_kit/tools/eval_runs.py --engine cft_engine --runs runs/spc --subsets data/spc_subsets --out results/eval.csv --expected-runs 60
for s in 1 2 3 4 5; do
  W=$(ls runs/spc/cmafm_full/seed_$s/train/weights/best.pt 2>/dev/null || true)
  [ -n "$W" ] && uvpy spc_kit/tools/alpha_stats.py --engine cft_engine --weights "$W" \
      --subsets data/spc_subsets --out results/alpha_seed$s.json
done
uvpy spc_kit/tools/aggregate.py --eval results/eval.csv --flops results/flops_640.json --ref cmafm_full --out results/tables.md
