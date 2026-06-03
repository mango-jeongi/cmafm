import json
import random
import matplotlib.pyplot as plt
import PIL.Image
from pathlib import Path
from collections import defaultdict

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

# Aligned Research Subset IDs
CAR_CAT_ID = 0 

def view_flir_cars(num_samples=3):
    if not JSON_PATH.exists():
        print(f"Error: Aligned JSON not found at {JSON_PATH}")
        return

    print(f"Loading Aligned FLIR metadata and searching for cars...")
    with open(JSON_PATH) as f:
        coco = json.load(f)
    
    # 1. Map Image IDs to their categories
    img_to_cats = defaultdict(set)
    for ann in coco["annotations"]:
        img_to_cats[ann["image_id"]].add(ann["category_id"])
    
    # 2. Filter images that contain a car AND exist on disk
    img_info = {img["id"]: img for img in coco["images"]}
    car_image_ids = []
    for img_id, cats in img_to_cats.items():
        if CAR_CAT_ID in cats:
            info = img_info[img_id]
            if (RGB_ROOT / info["file_name"]).exists() and (IR_ROOT / info["file_name"]).exists():
                car_image_ids.append(img_id)
    
    if not car_image_ids:
        print("Error: No images containing cars found.")
        return

    print(f"Found {len(car_image_ids)} images containing cars.")
    selected_ids = random.sample(car_image_ids, min(num_samples, len(car_image_ids)))
    
    for i, img_id in enumerate(selected_ids):
        info = img_info[img_id]
        fname = info["file_name"]
        
        img_rgb = PIL.Image.open(RGB_ROOT / fname)
        img_ir  = PIL.Image.open(IR_ROOT / fname)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Car Sample {i+1} - RGB (ID: {img_id})")
        axes[0].axis('on')
        
        axes[1].imshow(img_ir, cmap='gray')
        axes[1].set_title(f"Car Sample {i+1} - Thermal (ID: {img_id})")
        axes[1].axis('on')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    view_flir_cars(3)
