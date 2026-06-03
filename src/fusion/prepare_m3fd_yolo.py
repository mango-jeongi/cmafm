import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# --- CONFIGURATION ---
# UPDATE THIS PATH to where your raw M3FD data is on HPC-Cluster
RAW_DATA_ROOT = Path("src/fusion/data/M3FD")
OUTPUT_ROOT = Path("src/fusion/data/M3FD_YOLO")

classes = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
split_ratio = 0.8  # 80% train, 20% val

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def process_file(xml_path, img_type, split):
    # Determine target paths
    target_img_dir = OUTPUT_ROOT / "images" / split
    target_lab_dir = OUTPUT_ROOT / "labels" / split
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lab_dir, exist_ok=True)

    # Convert XML to TXT
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        with open(target_lab_dir / (xml_path.stem + ".txt"), "w") as f:
            for obj in root.findall('object'):
                cls = obj.find('name').text
                if cls not in classes: continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")
        
        # Copy Image (Using Vis as primary for now, or you can catenate)
        src_img = RAW_DATA_ROOT / "Vis" / (xml_path.stem + ".png")
        if src_img.exists():
            shutil.copy(src_img, target_img_dir / (xml_path.stem + ".png"))
            return True
    except Exception as e:
        print(f"Error: {e}")
    return False

def main():
    ann_dir = RAW_DATA_ROOT / "Annotation"
    all_xmls = list(ann_dir.glob("*.xml"))
    random.shuffle(all_xmls)
    
    split_idx = int(len(all_xmls) * split_ratio)
    train_xmls = all_xmls[:split_idx]
    val_xmls = all_xmls[split_idx:]

    print(f"Processing {len(train_xmls)} train and {len(val_xmls)} val files...")
    
    for x in train_xmls: process_file(x, "Vis", "train")
    for x in val_xmls: process_file(x, "Vis", "val")
    
    print(f"Done! Data ready at {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
