import os
import re
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
RUNS_DIR = Path("runs")
NUM_SEEDS = 10
# ---------------------

def extract_metrics(results_file):
    """
    Parses results.txt to get the best metrics.
    YOLOv5 format: epoch, train_loss..., precision, recall, mAP.5, mAP.5:.95, val_loss...
    """
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    # Get last line or the line with the best mAP.5
    best_map50 = -1.0
    best_metrics = None
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10: continue
        
        # YOLOv5 results.txt layout (usually):
        # 0: epoch, 1-3: train_losses, 4: precision, 5: recall, 6: mAP.5, 7: mAP.5:.95
        try:
            m50 = float(parts[6])
            if m50 > best_map50:
                best_map50 = m50
                best_metrics = {
                    "P": float(parts[4]),
                    "R": float(parts[5]),
                    "mAP50": float(parts[6]),
                    "mAP50-95": float(parts[7])
                }
        except (ValueError, IndexError):
            continue
            
    return best_metrics

def main():
    print("--- BMVC 2026: 10-Seed Metric Aggregator ---")
    
    seeds_data = []
    for i in range(1, NUM_SEEDS + 1):
        # Path logic: adjust this if your folders are named differently
        res_path = RUNS_DIR / f"seed_{i}" / "cmafm_hpc_run" / "results.txt"
        
        metrics = extract_metrics(res_path)
        if metrics:
            seeds_data.append(metrics)
            print(f"Seed {i:2d}: mAP@.5={metrics['mAP50']:.4f}, mAP@.5:.95={metrics['mAP50-95']:.4f}, R={metrics['R']:.4f}")
        else:
            print(f"Seed {i:2d}: [MISSING] {res_path}")

    if not seeds_data:
        print("No results found.")
        return

    # Calculate statistics
    for key in ["P", "R", "mAP50", "mAP50-95"]:
        vals = [d[key] for d in seeds_data]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"\n{key:8s}: {mean:.4f} ± {std:.4f}")
        
    print("\n--- TABLE 3-B READY DATA ---")
    m50_mean = np.mean([d["mAP50"] for d in seeds_data]) * 100
    m50_std = np.std([d["mAP50"] for d in seeds_data]) * 100
    print(f"CMAFM-YOLO (Ours): {m50_mean:.2f}% ± {m50_std:.2f}%")

if __name__ == "__main__":
    main()
