"""
Final Fix: Robust ID-based alignment for FLIR ADAS images with mismatched filenames.
Scenario A: Train on M3FD + FLIR (supplementary), Val on M3FD ONLY (Parity Benchmark)
Scenario B: Train on M3FD ONLY, Val on M3FD ONLY (Safety Fallback)
"""

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Optional .7z support
try:
    import py7zr
    HAS_7Z = True
except ImportError:
    HAS_7Z = False

random.seed(42)

# Load .env from repository root (parent of scripts/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
load_dotenv(_REPO_ROOT / ".env")

# --- PATHS ---
# 1. Search for source .zip/.7z files — CMAFM_DATA_DIR from .env takes priority
_ENV_DATA_DIR = os.environ.get("CMAFM_DATA_DIR")
DATA_SEARCH_PATHS = (
    [Path(_ENV_DATA_DIR)] if _ENV_DATA_DIR else []
) + [
    Path("../data"),   # Parallel folder fallback
    Path("data"),      # Local fallback
]

def get_zip_path(name):
    print(f"  Searching for: {name}")
    for p in DATA_SEARCH_PATHS:
        zp = p / name
        # Try both exact and lowercase match for robustness on Linux
        if zp.exists(): 
            print(f"    [FOUND] {zp}")
            return zp
        
        # Check for case-insensitive match on Linux
        if p.exists() and p.is_dir():
            for existing_file in p.iterdir():
                if existing_file.name.lower() == name.lower():
                    print(f"    [FOUND (Case-Insensitive)] {existing_file}")
                    return existing_file
                    
        print(f"    [Not found at] {zp}")
    return None

# 2. Define extraction/output base — prefer CMAFM_DATA_DIR from .env
if _ENV_DATA_DIR:
    BASE = Path(_ENV_DATA_DIR).resolve()
elif Path("../data").exists():
    # Recommended for HPC clusters to keep data in parallel folder
    BASE = Path("../data").resolve()
else:
    BASE = Path.cwd() / "data"

print(f"Working Directory (BASE): {BASE}")
M3FD_ROOT    = BASE / "M3FD"
FLIR_ROOT    = BASE / "FLIR_ADAS"
OUT_M3FD     = BASE / "M3FD_YOLO"
OUT_FLIR     = BASE / "FLIR_YOLO"

M3FD_CLASSES = ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']
M3FD_CLS_MAP = {c: i for i, c in enumerate(M3FD_CLASSES)}

# Aligned Research Subset Mapping:
# Aligned ID 0 (Car) -> M3FD 1 (Car)
# Aligned ID 1 (Person) -> M3FD 0 (People)
FLIR_CAT_MAP = {0: 1, 1: 0} 

VAL_RATIO = 0.2

def extract_datasets():
    """Unzip datasets if root folders don't exist."""
    import zipfile
    
    # 1. M3FD
    if not M3FD_ROOT.exists():
        zip_path = get_zip_path("M3FD_Detection.zip")
        if zip_path:
            print(f"Extracting M3FD from {zip_path} to {M3FD_ROOT}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(M3FD_ROOT)
                
    # 2. FLIR (Prioritize Aligned 7z)
    if not FLIR_ROOT.exists():
        flir_7z = get_zip_path("flir_align.7z")
        flir_zip = get_zip_path("FLIR_ADAS_v2.zip")
        
        if flir_7z:
            if HAS_7Z:
                print(f"Extracting Aligned FLIR from {flir_7z} to {FLIR_ROOT}...")
                with py7zr.SevenZipFile(flir_7z, mode='r') as z:
                    z.extractall(FLIR_ROOT)
            else:
                print(f"\n[CRITICAL ERROR] Found {flir_7z}, but 'py7zr' is not installed!")
                print("Please run: uv pip install py7zr")
                print("Skipping FLIR extraction.\n")
        elif flir_zip:
            print(f"Extracting Raw FLIR from {flir_zip} to {FLIR_ROOT}...")
            with zipfile.ZipFile(flir_zip, 'r') as z:
                z.extractall(FLIR_ROOT)

def convert_m3fd():
    print("── [1/3] M3FD VOC -> YOLO Conversion ──")
    ann_dir = next(M3FD_ROOT.glob("**/Annotation"), None)
    vis_dir = next(M3FD_ROOT.glob("**/Vis"), None)
    ir_dir  = next(M3FD_ROOT.glob("**/Ir"), None)

    if not ann_dir:
        print("Error: M3FD directory not found.")
        return {}

    all_stems = sorted([p.stem for p in ann_dir.glob("*.xml")])
    random.shuffle(all_stems)
    n_val = int(len(all_stems) * VAL_RATIO)
    val_stems  = set(all_stems[:n_val])
    train_stems = set(all_stems[n_val:])

    txt_files = {"train_rgb": [], "train_ir": [], "val_rgb": [], "val_ir": []}

    for split, stems in [("train", train_stems), ("val", val_stems)]:
        rgb_out = OUT_M3FD / split / "images"
        ir_out  = OUT_M3FD / split / "images_ir"
        lbl_out = OUT_M3FD / split / "labels"
        for d in [rgb_out, ir_out, lbl_out]: d.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            xml_path = ann_dir / f"{stem}.xml"
            vis_path = vis_dir  / f"{stem}.png"
            ir_path  = ir_dir   / f"{stem}.png"
            if not vis_path.exists() or not ir_path.exists(): continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            img_w, img_h = int(size.find("width").text), int(size.find("height").text)

            lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                if name not in M3FD_CLS_MAP: continue
                cls = M3FD_CLS_MAP[name]
                bb = obj.find("bndbox")
                xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
                xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)
                lines.append(f"{cls} {((xmin+xmax)/2)/img_w:.6f} {((ymin+ymax)/2)/img_h:.6f} {(xmax-xmin)/img_w:.6f} {(ymax-ymin)/img_h:.6f}")

            if not lines: continue
            shutil.copy2(vis_path, rgb_out / f"{stem}.png")
            shutil.copy2(ir_path,  ir_out  / f"{stem}.png")
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            txt_files[f"{split}_rgb"].append(str(rgb_out / f"{stem}.png").replace("\\", "/"))
            txt_files[f"{split}_ir"].append(str(ir_out   / f"{stem}.png").replace("\\", "/"))

    for key, lines in txt_files.items():
        (OUT_M3FD / f"{key}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  {key}.txt: {len(lines)} files")
    return txt_files

def convert_flir():
    print("\n── [2/3] FLIR ADAS -> YOLO Conversion (ID-based Alignment) ──")
    if not FLIR_ROOT.exists():
        print("Skipping FLIR (Root not found).")
        return {}

    # Find JSONs anywhere in the tree (handles both Aligned and Raw structure)
    train_json = next(FLIR_ROOT.glob("**/coco_annotations/train.json"), None) or \
                 next(FLIR_ROOT.glob("**/images_rgb_train/coco.json"), None)
    
    val_json   = next(FLIR_ROOT.glob("**/coco_annotations/test.json"), None) or \
                 next(FLIR_ROOT.glob("**/images_rgb_val/coco.json"), None)

    if not train_json:
        print("Error: FLIR JSON files not found.")
        return {}

    txt_files = {"train_rgb": [], "train_ir": [], "val_rgb": [], "val_ir": []}

    for json_path, split in [(train_json, "train"), (val_json, "val")]:
        if not json_path: continue
        
        coco = json.load(open(json_path))
        img_info = {img["id"]: img for img in coco["images"]}
        ann_map = defaultdict(list)
        for ann in coco["annotations"]: ann_map[ann["image_id"]].append(ann)

        rgb_out = OUT_FLIR / split / "images"
        ir_out  = OUT_FLIR / split / "images_ir"
        lbl_out = OUT_FLIR / split / "labels"
        for d in [rgb_out, ir_out, lbl_out]: d.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_id, info in img_info.items():
            fname = info["file_name"]
            # Search for image file in the same parent as JSON or adjacent sibling folders
            rgb_path = next(FLIR_ROOT.glob(f"**/visible/*/{fname}"), None) or \
                       next(FLIR_ROOT.glob(f"**/images_rgb_*/{fname}"), None)
            ir_path  = next(FLIR_ROOT.glob(f"**/thermal/*/{fname}"), None) or \
                       next(FLIR_ROOT.glob(f"**/images_thermal_*/{fname}"), None)

            if not rgb_path or not ir_path: continue

            lines = []
            for ann in ann_map[img_id]:
                cat = ann["category_id"]
                if cat not in FLIR_CAT_MAP: continue
                cls = FLIR_CAT_MAP[cat]
                x, y, w, h = ann["bbox"]
                img_w, img_h = info["width"], info["height"]
                lines.append(f"{cls} {min(1.0, max(0.0,(x+w/2)/img_w)):.6f} {min(1.0, max(0.0,(y+h/2)/img_h)):.6f} {min(1.0, max(0.0,w/img_w)):.6f} {min(1.0, max(0.0,h/img_h)):.6f}")

            if not lines: continue
            # Handle img_id as int for formatting
            try:
                img_id_int = int(img_id)
                stem = f"flir_{split}_{img_id_int:05d}"
            except ValueError:
                stem = f"flir_{split}_{img_id}"
            
            shutil.copy2(rgb_path, rgb_out / f"{stem}.jpg")
            shutil.copy2(ir_path,  ir_out  / f"{stem}.jpg")
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            txt_files[f"{split}_rgb"].append(str(rgb_out / f"{stem}.jpg").replace("\\", "/"))
            txt_files[f"{split}_ir"].append(str(ir_out   / f"{stem}.jpg").replace("\\", "/"))
            count += 1
        print(f"  [{split}] Complete: {count} files")
    return txt_files

def generate_configs(m3fd_txt, flir_txt):
    print("\n── [3/3] Generating Unified txt and yaml configurations ──")
    merged_dir = BASE / "merged_txt"
    merged_dir.mkdir(parents=True, exist_ok=True)
    mp = merged_dir.absolute().as_posix()

    # 1. SCENARIO A: M3FD + FLIR Train, M3FD ONLY Val (The 85.9% Benchmark)
    for key in ["train_rgb", "train_ir", "val_rgb", "val_ir"]:
        if "train" in key:
            merged = m3fd_txt.get(key, []) + flir_txt.get(key, [])
        else:
            merged = m3fd_txt.get(key, []) # Benchmark ONLY on Aligned M3FD
        random.shuffle(merged)
        (merged_dir / f"{key}_unified.txt").write_text("\n".join(merged), encoding="utf-8")

    # Ensure engine CFT config directory exists
    cfg_dir = _REPO_ROOT / "cft_engine" / "data"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    (cfg_dir / "M3FD_FLIR.yaml").write_text(f"train_rgb: {mp}/train_rgb_unified.txt\nval_rgb: {mp}/val_rgb_unified.txt\ntrain_ir: {mp}/train_ir_unified.txt\nval_ir: {mp}/val_ir_unified.txt\nnc: 6\nnames: {M3FD_CLASSES}\n", encoding="utf-8")

    # 2. SCENARIO B: M3FD + FLIR Train, FLIR ONLY Val (Generalization Check, 84.8% target)
    for key in ["val_rgb", "val_ir"]:
        lines = flir_txt.get(key, [])
        (merged_dir / f"{key}_flir_only.txt").write_text("\n".join(lines), encoding="utf-8")
    
    (cfg_dir / "FLIR_VAL.yaml").write_text(f"train_rgb: {mp}/train_rgb_unified.txt\nval_rgb: {mp}/val_rgb_flir_only.txt\ntrain_ir: {mp}/train_ir_unified.txt\nval_ir: {mp}/val_ir_flir_only.txt\nnc: 6\nnames: {M3FD_CLASSES}\n", encoding="utf-8")

    # 3. SCENARIO C: M3FD + FLIR Train, MIXED Val (Comprehensive evaluation)
    for key in ["val_rgb", "val_ir"]:
        merged = m3fd_txt.get(key, []) + flir_txt.get(key, [])
        random.shuffle(merged)
        (merged_dir / f"{key}_mixed.txt").write_text("\n".join(merged), encoding="utf-8")

    (cfg_dir / "MIXED_VAL.yaml").write_text(f"train_rgb: {mp}/train_rgb_unified.txt\nval_rgb: {mp}/val_rgb_mixed.txt\ntrain_ir: {mp}/train_ir_unified.txt\nval_ir: {mp}/val_ir_mixed.txt\nnc: 6\nnames: {M3FD_CLASSES}\n", encoding="utf-8")

    # 4. M3FD ONLY (Safety Fallback)
    for key in ["train_rgb", "train_ir", "val_rgb", "val_ir"]:
        lines = m3fd_txt.get(key, [])
        (merged_dir / f"{key}_m3fd_only.txt").write_text("\n".join(lines), encoding="utf-8")
    
    (cfg_dir / "M3FD_ONLY.yaml").write_text(f"train_rgb: {mp}/train_rgb_m3fd_only.txt\nval_rgb: {mp}/val_rgb_m3fd_only.txt\ntrain_ir: {mp}/train_ir_m3fd_only.txt\nval_ir: {mp}/val_ir_m3fd_only.txt\nnc: 6\nnames: {M3FD_CLASSES}\n", encoding="utf-8")
    
    print(f"Configs generated in src/engine/CFT_repo/")

if __name__ == "__main__":
    extract_datasets()
    m_txt = convert_m3fd()
    f_txt = convert_flir()
    generate_configs(m_txt, f_txt)
