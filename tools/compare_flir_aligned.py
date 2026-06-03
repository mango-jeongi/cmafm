import json
import random
import matplotlib.pyplot as plt
import PIL.Image
from pathlib import Path

# --- PATHS ---
# This matches the extracted structure on Hamming/SSD after running prepare_all.py
if Path(r"../data").exists():
    BASE = Path(r"../data")
elif Path("../data").exists():
    BASE = Path("../data").resolve()
else:
    BASE = Path.cwd() / "data"

# Point to the source images in the extracted FLIR_ADAS directory
RGB_ROOT = BASE / "FLIR_ADAS" / "visible" / "train"
IR_ROOT  = BASE / "FLIR_ADAS" / "thermal" / "train"
JSON_PATH = BASE / "FLIR_ADAS" / "coco_annotations" / "train.json"

def compare_flir_samples(num_samples=3):
    if not JSON_PATH.exists():
        print(f"Error: Aligned JSON not found at {JSON_PATH}")
        print("Please ensure you have run 'python scripts/prepare_all.py' to extract the 7z.")
        return

    print(f"Loading Aligned FLIR metadata...")
    with open(JSON_PATH) as f:
        coco = json.load(f)
    
    # Filter for images that exist in both folders
    img_info = {img["id"]: img for img in coco["images"]}
    valid_ids = []
    for img_id, info in img_info.items():
        if (RGB_ROOT / info["file_name"]).exists() and (IR_ROOT / info["file_name"]).exists():
            valid_ids.append(img_id)
    
    if not valid_ids:
        print("Error: No image pairs found in visible/train and thermal/train folders.")
        return

    selected_ids = random.sample(valid_ids, min(num_samples, len(valid_ids)))
    
    print(f"Displaying {len(selected_ids)} Aligned Research Pairs:")
    for i, img_id in enumerate(selected_ids):
        info = img_info[img_id]
        fname = info["file_name"]
        
        img_rgb = PIL.Image.open(RGB_ROOT / fname)
        img_ir  = PIL.Image.open(IR_ROOT / fname)
        
        # Create a side-by-side comparison with a coordinate grid to check alignment
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Sample {i+1} - Aligned RGB (ID: {img_id})")
        axes[0].axis('on') # Keep axis on to check coordinate parity
        
        axes[1].imshow(img_ir, cmap='gray')
        axes[1].set_title(f"Sample {i+1} - Aligned Thermal (ID: {img_id})")
        axes[1].axis('on')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    compare_flir_samples(3)
