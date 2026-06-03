import sys, os
from pathlib import Path

# Add root and CFT_repo to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cft_engine"))

import torch
from models.yolo import Model
from utils.torch_utils import select_device, model_info

# --- PATHS ---
CFG = ROOT / "src" / "engine" / "CFT_repo" / "yolov5l_cmafm_M3FD.yaml"
# -------------

def get_efficiency():
    device = select_device('cpu')
    print(f"\n[INFO] Loading model for profiling: {CFG}")
    
    # Create model
    model = Model(str(CFG), ch=3, nc=6).to(device)
    
    print("[INFO] Running THOP profiler...")
    
    # Manual GFLOPs calculation (matching YOLOv5 logic)
    from thop import profile
    from copy import deepcopy
    
    img_size = 640
    stride = 32
    # Dual-stream input: we simulate with the primary input channel count
    # THOP will count all operations in the graph (both streams + fusion)
    img = torch.zeros((1, 3, stride, stride), device=device)
    
    # Profile the model
    flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1e9 * 2
    gflops = flops * (img_size / stride) * (img_size / stride)
    
    n_p = sum(x.numel() for x in model.parameters())
    
    print("-" * 50)
    print(f"FINAL EFFICIENCY REPORT:")
    print(f"  Architecture: CMAFM-YOLO (Dual-Stream)")
    print(f"  Total Params: {n_p / 1e6:.2f}M")
    print(f"  Total GFLOPs: {gflops:.1f} (at {img_size}x{img_size})")
    print("-" * 50)
    print("\n[SUCCESS] Efficiency metrics extracted.")

if __name__ == "__main__":
    get_efficiency()
