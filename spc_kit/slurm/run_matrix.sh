#!/bin/bash
# One SLURM array task = one (config, seed) run under the single SPC protocol.
#   sbatch --array=0-29 spc_kit/slurm/run_matrix.sh spc_kit/slurm/matrix_main.txt 5
#   sbatch --array=0-29 spc_kit/slurm/run_matrix.sh spc_kit/slurm/matrix_ablation.txt 5
# Task index i -> config = line (i / SEEDS) of the matrix file, seed = (i % SEEDS) + 1.
#SBATCH --job-name=spc_matrix
#SBATCH --output=logs/spc_%A_%a.out
#SBATCH --error=logs/spc_%A_%a.err
#SBATCH --partition=genai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --get-user-env
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
MATRIX=${1:?matrix file}
SEEDS=${2:-5}
EPOCHS=${EPOCHS:-100}
[[ "$SEEDS" =~ ^[1-9][0-9]*$ && "$EPOCHS" =~ ^[1-9][0-9]*$ ]]
mapfile -t CFGS < <(sed 's/\r$//' "$MATRIX" | grep -vE '^\s*(#|$)')
IDX=${SLURM_ARRAY_TASK_ID:-0}
[[ "$IDX" =~ ^[0-9]+$ && "$IDX" -lt $((${#CFGS[@]} * SEEDS)) ]]
CFG=${CFGS[$((IDX / SEEDS))]}
SEED=$(( IDX % SEEDS + 1 ))
ROOT=$(pwd)
VENV=${SPC_VENV:-$ROOT/.venv}
UV=${UV_BIN:-$HOME/.local/bin/uv}
export UV_PROJECT_ENVIRONMENT="$VENV"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
RUNS=${SPC_RUNS:-$ROOT/runs/spc}
if [[ "$EPOCHS" != 100 && -z "${SPC_RUNS:-}" ]]; then
    RUNS="$ROOT/runs/spc_smoke/job_${SLURM_ARRAY_JOB_ID:-manual}"
fi
[[ "$RUNS" == /* ]] || RUNS="$ROOT/$RUNS"
OUT="$RUNS/$CFG/seed_$SEED/train"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 1; }
[[ -x "$VENV/bin/python" && -x "$UV" ]]
mkdir -p "$OUT"
git -C "$ROOT" rev-parse HEAD > "$OUT/source_commit.txt"
export WANDB_MODE=disabled
mkdir -p logs
cd cft_engine
echo "config=$CFG seed=$SEED epochs=$EPOCHS host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
"$UV" run --no-project --python "$VENV/bin/python" train.py \
    --seed "$SEED" \
    --weights "$ROOT/weights/spc_init/$CFG.pt" \
    --cfg "models/spc/$CFG.yaml" \
    --data "$ROOT/data/spc_split/spc_train.yaml" \
    --hyp "$ROOT/data/hyp.scratch.yaml" \
    --img-size 640 640 \
    --device 0 \
    --epochs "$EPOCHS" \
    --batch-size 8 \
    --workers 8 \
    --project "$RUNS/$CFG/seed_$SEED" \
    --name train --exist-ok
