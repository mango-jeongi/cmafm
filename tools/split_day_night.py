import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv

# Load .env from repository root
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

DATA_DIR = os.environ.get("CMAFM_DATA_DIR")
if DATA_DIR:
    BASE = Path(DATA_DIR).resolve()
elif Path(r"../data").exists():
    BASE = Path(r"../data")
elif Path("../data").exists():
    BASE = Path("../data").resolve()
else:
    BASE = Path.cwd() / "data"

BRIGHTNESS_THRESHOLD = 60
# ---------------------

def split_day_night():
    # Define targets to split with sensor-specific thresholds
    # M3FD: 60 (standard)
    # FLIR: 100 (adaptive - captures the darkest tail since min is 74)
    targets = [
        {"name": "M3FD", "list": BASE / "merged_txt" / "val_rgb_unified.txt", "filter": "M3FD", "threshold": 60},
        {"name": "FLIR", "list": BASE / "merged_txt" / "val_rgb_flir_only.txt", "filter": "FLIR", "threshold": 100}
    ]

    for target in targets:
        if not target["list"].exists():
            print(f"Skipping {target['name']} (list not found at {target['list']})")
            continue

        print(f"\nScanning {target['name']} for Day/Night split (Threshold: {target['threshold']})...")
        with open(target["list"], 'r') as f:
            img_paths = [line.strip() for line in f if line.strip()]
        
        day_imgs = []
        night_imgs = []
        
        for img_path in tqdm(img_paths):
            if target["filter"] not in img_path:
                continue
                
            img = cv2.imread(img_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < target["threshold"]:
                night_imgs.append(img_path)
            else:
                day_imgs.append(img_path)
                
        print(f"  {target['name']} Results: Day={len(day_imgs)}, Night={len(night_imgs)}")
        
        # Save results
        out_dir = BASE / "merged_txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = "val_rgb_" + target['name'].lower()
        (out_dir / f"{prefix}_day.txt").write_text("\n".join(day_imgs))
        (out_dir / f"{prefix}_night.txt").write_text("\n".join(night_imgs))
        
        ir_prefix = "val_ir_" + target['name'].lower()
        (out_dir / f"{ir_prefix}_day.txt").write_text("\n".join([p.replace("/images/", "/images_ir/").replace(".jpg", ".jpg").replace(".png", ".png") for p in day_imgs]))
        (out_dir / f"{ir_prefix}_night.txt").write_text("\n".join([p.replace("/images/", "/images_ir/").replace(".jpg", ".jpg").replace(".png", ".png") for p in night_imgs]))

if __name__ == "__main__":
    split_day_night()
