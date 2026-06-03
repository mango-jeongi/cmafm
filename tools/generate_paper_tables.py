import sys, os
import yaml
from pathlib import Path

# Add root and CFT_repo to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "CFT_repo"))

import torch
import test
from models.experimental import attempt_load
from utils.torch_utils import select_device

# --- CONFIGURATION ---
# Use the best seed result from the cluster
BEST_WEIGHTS = ROOT / "runs" / "seed_6" / "cmafm_hpc_run" / "weights" / "best.pt"

# Path logic for Hamming/SSD
if Path(r"../data").exists():
    BASE = Path(r"../data")
elif Path("../data").exists():
    BASE = Path("../data").resolve()
else:
    BASE = Path.cwd() / "data"

MP = BASE / "merged_txt"
M3FD_CLASSES = ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']

# ---------------------

def generate_sub_configs():
    """Generates the .yaml files for each test scenario."""
    configs = {
        "M3FD_ALL":    {"tr": "train_rgb_unified.txt", "val": "val_rgb_unified.txt"},
        "M3FD_DAY":    {"tr": "train_rgb_unified.txt", "val": "val_rgb_m3fd_day.txt"},
        "M3FD_NIGHT":  {"tr": "train_rgb_unified.txt", "val": "val_rgb_m3fd_night.txt"},
        "FLIR_ONLY":   {"tr": "train_rgb_unified.txt", "val": "val_rgb_flir_only.txt"},
        "FLIR_DAY":    {"tr": "train_rgb_unified.txt", "val": "val_rgb_flir_day.txt"},
        "FLIR_NIGHT":  {"tr": "train_rgb_unified.txt", "val": "val_rgb_flir_night.txt"},
    }
    
    for name, files in configs.items():
        yaml_content = {
            "train_rgb": str(MP / files["tr"]).replace("\\", "/"),
            "val_rgb":   str(MP / files["val"]).replace("\\", "/"),
            "train_ir":  str(MP / files["tr"].replace("rgb", "ir")).replace("\\", "/"),
            "val_ir":    str(MP / files["val"].replace("rgb", "ir")).replace("\\", "/"),
            "nc": 6,
            "names": M3FD_CLASSES
        }
        with open(ROOT / "src" / "engine" / "CFT_repo" / f"{name}.yaml", 'w') as f:
            yaml.dump(yaml_content, f)
    return configs.keys()

def run_scenario(name, device_str):
    print(f"\nEvaluating Scenario: {name}")
    data_path = ROOT / "src" / "engine" / "CFT_repo" / f"{name}.yaml"
    
    # Create a dummy opt object required by the author's test.py engine
    class DummyOpt:
        def __init__(self):
            self.device = device_str
            self.project = 'runs/paper_eval'
            self.name = name
            self.exist_ok = True
            self.task = 'val'
            self.single_cls = False
            self.verbose = False
            self.save_txt = False
            self.save_hybrid = False
            self.save_conf = False
            self.save_json = False
    
    opt = DummyOpt()
    
    # Run official test.py engine
    # Setting model=None so test.py handles its own dataloader creation
    results, maps, times = test.test(
        str(data_path),
        weights=str(BEST_WEIGHTS),
        batch_size=32,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.6,
        model=None,
        single_cls=False,
        dataloader=None,
        save_dir=Path("runs/paper_eval"),
        plots=False,
        is_coco=False,
        opt=opt
    )
    
    # Aggregate metrics
    metrics = {
        "mAP@.5": results[2],
        "mAP@.5:.95": results[3],
        "Recall": results[1],
        "Precision": results[0],
        "F1": 2 * (results[0] * results[1]) / (results[0] + results[1] + 1e-16),
        "MissRate": 1 - results[1],
        "ClassAPs": maps # This is a list of AP@.5 for each class
    }
    return metrics

def main():
    if not BEST_WEIGHTS.exists():
        print(f"Error: Weights not found at {BEST_WEIGHTS}")
        return

    device_str = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    
    scenarios = generate_sub_configs()
    all_results = {}
    
    for s in scenarios:
        all_results[s] = run_scenario(s, device_str)

    # --- PRINT TABLE 6: Night/Day Performance ---
    print("\n" + "="*30)
    print("TABLE 6: Night/Day Performance (M3FD)")
    print("="*30)
    print("| Condition | mAP@0.5 | Recall | Miss Rate |")
    print("| :--- | :--- | :--- | :--- |")
    for s in ["M3FD_ALL", "M3FD_NIGHT", "M3FD_DAY"]:
        r = all_results[s]
        print(f"| {s.split('_')[1]} | {r['mAP@.5']:.3f} | {r['Recall']:.3f} | {r['MissRate']:.3f} |")

    # --- PRINT TABLE 7: Class-wise AP@0.5 Comparison ---
    print("\n" + "="*30)
    print("TABLE 7: Class-wise AP@0.5 (Night vs Day)")
    print("="*30)
    print("| Class | Night | Day | Night - Day |")
    print("| :--- | :--- | :--- | :--- |")
    
    night_maps = all_results["M3FD_NIGHT"]["ClassAPs"]
    day_maps   = all_results["M3FD_DAY"]["ClassAPs"]
    
    for i, cls in enumerate(M3FD_CLASSES):
        diff = night_maps[i] - day_maps[i]
        print(f"| {cls:<10} | {night_maps[i]:.3f} | {day_maps[i]:.3f} | {diff:+.3f} |")

    # --- PRINT GENERALIZATION: FLIR Results ---
    print("\n" + "="*30)
    print("GENERALIZATION: FLIR Aligned (Day vs Night)")
    print("="*30)
    print("| Condition | mAP@0.5 | Recall | Miss Rate |")
    print("| :--- | :--- | :--- | :--- |")
    for s in ["FLIR_ONLY", "FLIR_NIGHT", "FLIR_DAY"]:
        r = all_results[s]
        cond = s.split('_')[1] if "_" in s else "Total"
        if s == "FLIR_ONLY": cond = "Total"
        print(f"| {cond:<10} | {r['mAP@.5']:.3f} | {r['Recall']:.3f} | {r['MissRate']:.3f} |")

if __name__ == "__main__":
    main()
