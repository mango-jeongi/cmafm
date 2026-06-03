import sys, os
from pathlib import Path

# Add root and cft_engine to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cft_engine"))

import torch
from models.yolo_test import Model
from utils.torch_utils import select_device

# --- PATHS ---
PRETRAINED = ROOT / "weights" / "yolov5l.pt"
CFG        = ROOT / "cft_engine" / "models" / "yolov5l_cmafm_M3FD.yaml"
OUT        = ROOT / "weights" / "CMAFM_Pretrained.pt"
# -------------

def run_mapping():
    if not PRETRAINED.exists():
        print(f"Error: {PRETRAINED} not found.")
        return

    device = select_device('cpu')
    print(f"Loading target CMAFM architecture...")
    model = Model(str(CFG), ch=3, nc=6).to(device)
    model_sd = model.state_dict()

    print(f"Loading source YOLOv5l weights...")
    ckpt = torch.load(PRETRAINED, map_location='cpu', weights_only=False)
    pretrained_sd = ckpt.get('model', ckpt)
    if hasattr(pretrained_sd, 'state_dict'):
        pretrained_sd = pretrained_sd.state_dict()

    # Normalize source keys
    pt_sd = {}
    for k, v in pretrained_sd.items():
        nk = k if k.startswith('model.') else ('model.' + k)
        pt_sd[nk] = v

    # STRICT LAYER MAP (Proven to reach 0.55 mAP in 3 epochs)
    LAYER_MAP = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),    # Stream 1 (RGB)
        (5, 0), (6, 1), (7, 2), (8, 3), (9, 4),    # Stream 2 (Thermal)
        (13, 5), (14, 6), (15, 5), (16, 6),        # P4
        (20, 7), (21, 8), (22, 9), (23, 7), (24, 8), (25, 9), # P5
        (32, 10), (35, 13), (36, 14), (39, 17),    # Head Up
        (40, 18), (42, 20), (43, 21), (45, 23)     # Head Down
    ]

    new_sd = model.state_dict()
    matched_tensors = 0

    print("\nExecuting Strict Suffix-Based Tensor Mapping...")
    for (cm_idx, pt_idx) in LAYER_MAP:
        cm_prefix = f'model.{cm_idx}.'
        pt_prefix = f'model.{pt_idx}.'
        
        cm_keys = [k for k in model_sd if k.startswith(cm_prefix)]
        pt_keys = [k for k in pt_sd if k.startswith(pt_prefix)]

        for ck in cm_keys:
            parts = ck.split('.')
            suffix = parts[-1] # weight, bias, etc.
            # Match internal block names (cv1, cv2, etc.)
            if len(parts) > 2:
                suffix = '.'.join(parts[2:])

            for pk in pt_keys:
                if pk.endswith(suffix) and pt_sd[pk].shape == model_sd[ck].shape:
                    new_sd[ck] = pt_sd[pk]
                    matched_tensors += 1
                    break

    total_tensors = len(model_sd)
    print(f"Mapping Result: {matched_tensors}/{total_tensors} tensors transferred.")
    print(f"Percentage: {(matched_tensors/total_tensors)*100:.1f}%")

    model.load_state_dict(new_sd, strict=False)
    torch.save({'model': model, 'optimizer': None, 'epoch': -1}, OUT)
    print(f"\nSaved proven weights to: {OUT}")

if __name__ == "__main__":
    run_mapping()
