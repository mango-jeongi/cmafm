#!/bin/bash
#SBATCH --job-name=paper_tables
#SBATCH --output=logs/paper_tables.out
#SBATCH --error=logs/paper_tables.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Activate venv
source .venv/bin/activate

# Ensure setup.sh was run first to configure paths
# Step 1: Split Day/Night (Fast, OpenCV based)
python tools/split_day_night.py

# Step 2: Generate all tables (Fast with GPU)
python tools/generate_paper_tables.py
