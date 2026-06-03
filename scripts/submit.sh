#!/bin/bash
# scripts/submit.sh
# A wrapper script to apply global .env configurations to SLURM submissions

# Automatically export all variables loaded
set -a
[ -f .env ] && . .env
set +a

# Fallback to 'gpu' if SLURM_PARTITION is not defined in .env
PART=${SLURM_PARTITION:-gpu}

echo "Submitting job to partition: $PART"
sbatch --partition=$PART "$@"
