#!/bin/bash
#SBATCH --gres=gpu:1
source .venv/bin/activate

echo "Running inference on Full CMAFM and plotting curves..."
python src/fusion/plot_ablation_curves.py
