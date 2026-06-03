"""
LLVIP dataset -> YOLO format conversion + M3FD integration script

Assumes LLVIP folder structure:
  LLVIP/
      visible/train/   (RGB)
      visible/test/
      infrared/train/  (IR)
      infrared/test/
      Annotations/     (VOC XML, mixed train+test)

Usage:
  python prepare_llvip.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random

# -- Path Configurations ----------------------------------------------------
BASE         = Path(".")
LLVIP_ROOT   = BASE / "LLVIP"
OUT_ROOT     = BASE / "src/fusion/data/LLVIP_yolo"
M3FD_TXT_DIR = BASE / "src/fusion/data/M3FD_yolo"

# LLVIP only contains pedestrians -> maps to People (0) in M3FD
LLVIP_CLASS_ID = 0  # M3FD names: ['People','Car','Bus','Motorcycle','Lamp','Truck']

SPLITS = ["train", "test"]  # LLVIP split names (using test as validation)


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int):
    """VOC XML -> YOLO txt line list"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in ("person", "people"):
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        lines.append(f"{LLVIP_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def process_split(split: str):
    rgb_img_dir = LLVIP_ROOT / "visible"   / split
    ir_img_dir  = LLVIP_ROOT / "infrared"  / split
    ann_dir     = LLVIP_ROOT / "Annotations"

    out_split = "val" if split == "test" else "train"

    rgb_out_img = OUT_ROOT / out_split / "rgb"
    ir_out_img  = OUT_ROOT / out_split / "ir"
    lbl_out     = OUT_ROOT / out_split / "labels"

    for d in [rgb_out_img, ir_out_img, lbl_out]:
        d.mkdir(parents=True, exist_ok=True)

    img_files = sorted(rgb_img_dir.glob("*.jpg")) + sorted(rgb_img_dir.glob("*.png"))
    print(f"[{split}] Processing {len(img_files)} images...")

    rgb_txt_lines = []
    ir_txt_lines  = []
    skipped = 0

    for rgb_path in img_files:
        stem = rgb_path.stem
        ir_path  = ir_img_dir  / rgb_path.name
        xml_path = ann_dir / f"{stem}.xml"

        if not ir_path.exists() or not xml_path.exists():
            skipped += 1
            continue

        # Get image dimensions (fallback to defaults if PIL not available)
        try:
            from PIL import Image
            with Image.open(rgb_path) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 1280, 1024  # Standard LLVIP resolution

        yolo_lines = voc_to_yolo(xml_path, img_w, img_h)
        if not yolo_lines:
            skipped += 1
            continue

        # Copy images
        shutil.copy2(rgb_path, rgb_out_img / rgb_path.name)
        shutil.copy2(ir_path,  ir_out_img  / ir_path.name)

        # Save label
        lbl_path = lbl_out / f"{stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines))

        rgb_txt_lines.append(str(rgb_out_img / rgb_path.name))
        ir_txt_lines.append(str(ir_out_img   / rgb_path.name))

    print(f"  -> Complete: {len(rgb_txt_lines)} files, Skipped: {skipped}")
    return out_split, rgb_txt_lines, ir_txt_lines


def merge_with_m3fd(split: str, rgb_lines: list, ir_lines: list):
    """Add LLVIP paths to M3FD text files to create a merged training set"""
    suffix = "train" if split == "train" else "val"

    for modal, new_lines in [("rgb", rgb_lines), ("ir", ir_lines)]:
        m3fd_txt = M3FD_TXT_DIR / f"{suffix}_{modal}.txt"
        merged_txt = M3FD_TXT_DIR / f"{suffix}_{modal}_merged.txt"

        if m3fd_txt.exists():
            existing = m3fd_txt.read_text().strip().splitlines()
        else:
            existing = []
            print(f"  [Warning] {m3fd_txt} not found - using LLVIP only")

        merged = existing + new_lines
        random.shuffle(merged)
        merged_txt.write_text("\n".join(merged))
        print(f"  {merged_txt.name}: M3FD {len(existing)} + LLVIP {len(new_lines)} = {len(merged)}")


def write_yaml():
    yaml_path = BASE / "src/engine/CFT_repo/M3FD_LLVIP.yaml"
    m3fd_txt  = M3FD_TXT_DIR.as_posix()
    content = f"""# M3FD + LLVIP Unified Dataset
# M3FD 6 classes + LLVIP pedestrian augmentation

train_rgb: {m3fd_txt}/train_rgb_merged.txt
val_rgb:   {m3fd_txt}/val_rgb_merged.txt
train_ir:  {m3fd_txt}/train_ir_merged.txt
val_ir:    {m3fd_txt}/val_ir_merged.txt

nc: 6
names: ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(content)
    print(f"\nSaved YAML: {yaml_path}")


if __name__ == "__main__":
    if not LLVIP_ROOT.exists():
        print(f"[Error] LLVIP directory not found: {LLVIP_ROOT}")
        print("Please download and extract LLVIP to the specified directory first.")
        exit(1)

    all_rgb = {"train": [], "val": []}
    all_ir  = {"train": [], "val": []}

    for split in SPLITS:
        out_split, rgb_lines, ir_lines = process_split(split)
        all_rgb[out_split] += rgb_lines
        all_ir[out_split]  += ir_lines

    print("\n── Merging with M3FD ──")
    for split in ["train", "val"]:
        merge_with_m3fd(split, all_rgb[split], all_ir[split])

    write_yaml()
    print("\nDone! Ready to train with M3FD_LLVIP.yaml.")
