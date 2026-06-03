import json
import random
import matplotlib.pyplot as plt
import PIL.Image
from pathlib import Path

import os
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

EXTRACTED_ROOT = BASE / "FLIR_ADAS"

def verify_flir(num_samples=5):
    if EXTRACTED_ROOT.exists():
        print(f"Checking extracted data: {EXTRACTED_ROOT}")
        json_path = next(EXTRACTED_ROOT.glob("**/coco_annotations/train.json"), None)
        if json_path:
            coco = json.load(open(json_path))
            img_info = {img["id"]: img for img in coco["images"]}
            ids = random.sample(list(img_info.keys()), min(num_samples, len(img_info)))
            
            for i, img_id in enumerate(ids):
                info = img_info[img_id]
                fname = info["file_name"]
                rgb_path = next(EXTRACTED_ROOT.glob(f"**/visible/train/{fname}"), None)
                ir_path  = next(EXTRACTED_ROOT.glob(f"**/thermal/train/{fname}"), None)
                
                if not rgb_path or not ir_path: 
                    print(f"  [Skip] Missing images for ID {img_id}")
                    continue
                
                img_rgb = PIL.Image.open(rgb_path)
                img_ir  = PIL.Image.open(ir_path)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"Sample {i+1} - RGB (ID: {img_id})")
                axes[0].axis('off')
                
                axes[1].imshow(img_ir, cmap='gray')
                axes[1].set_title(f"Sample {i+1} - Thermal (ID: {img_id})")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.show()
            return

    print(f"Extracted data not found at {EXTRACTED_ROOT}. Please run scripts/prepare_all.py first.")

if __name__ == "__main__":
    verify_flir(5)
