import torch
from pathlib import Path
import os
import sys
import requests

def download_weights(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url}...")
        r = requests.get(url, allow_redirects=True)
        open(dest, 'wb').write(r.content)

def bootstrap():
    print("--- CMAFM Weight Bootstrapper (v9) ---")
    
    # 1. Add engine to path
    engine_dir = Path('cft_engine').absolute()
    if engine_dir.exists() and str(engine_dir) not in sys.path:
        sys.path.append(str(engine_dir))
    
    # 2. Dynamic SPPF Injection
    try:
        import torch.nn as nn
        import models.common
        if not hasattr(models.common, 'SPPF'):
            class SPPF(nn.Module):
                def __init__(self, c1, c2, k=5): super().__init__()
            models.common.SPPF = SPPF
    except: pass
    
    # 3. Get Source Weights (v5.0)
    src_pt = 'yolov5l.pt'
    download_weights('https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt', src_pt)
    
    # 4. Load & Graft
    checkpoint = torch.load(src_pt, map_location='cpu', weights_only=False)
    src_state = checkpoint['model'].state_dict()
    
    mapping = {}
    for s_idx in range(0, 5): mapping[s_idx] = [s_idx, s_idx + 5]
    for s_idx in range(5, 7): mapping[s_idx] = [s_idx + 8, s_idx + 10]
    for s_idx in range(7, 10): mapping[s_idx] = [s_idx + 13, s_idx + 16]
    for s_idx in range(10, 24): mapping[s_idx] = [s_idx + 22]

    new_state = {}
    for key, value in src_state.items():
        parts = key.split('.')
        if parts[0] == 'model' and parts[1].isdigit():
            src_idx = int(parts[1])
            if src_idx in mapping:
                for target_idx in mapping[src_idx]:
                    new_parts = parts.copy(); new_parts[1] = str(target_idx)
                    new_state['.'.join(new_parts)] = value.clone()
        else:
            new_state[key] = value.clone()

    # 5. SAVE RAW STATE DICT (Matches our train.py patch)
    out_pt = 'weights/CMAFM_Pretrained.pt'
    Path('weights').mkdir(parents=True, exist_ok=True)
    torch.save(new_state, out_pt)
    print(f"--- SUCCESS: Pretrained weights saved to {out_pt} ---")

if __name__ == "__main__":
    bootstrap()
