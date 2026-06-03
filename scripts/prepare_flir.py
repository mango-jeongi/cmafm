import json
import shutil
from pathlib import Path
from collections import defaultdict

# -- Path Configurations ----------------------------------------------------
BASE         = Path(".")
FLIR_ROOT    = BASE / "FLIR_ADAS"
OUT_ROOT     = BASE / "src/fusion/data/FLIR_yolo"
M3FD_TXT_DIR = BASE / "src/fusion/data/M3FD_yolo"

# FLIR category_id -> M3FD class_id
# FLIR: car=0, person=1, bicycle=2, dog=3
CATEGORY_MAP = {
    0: 1,   # car    -> Car(1)
    1: 0,   # person -> People(0)
    # 2: bicycle -> excluded
    # 3: dog     -> excluded
}

SPLITS = {
    "train": "train",
    "test":  "val",
}


def coco_to_yolo(ann_list, img_w, img_h):
    """COCO bbox [x,y,w,h] -> YOLO [cls cx cy w h] line list"""
    lines = []
    for ann in ann_list:
        cat_id = ann["category_id"]
        if cat_id not in CATEGORY_MAP:
            continue
        cls = CATEGORY_MAP[cat_id]
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        # Range clipping
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        if nw > 0 and nh > 0:
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def process_split(flir_split: str, out_split: str):
    ann_path = FLIR_ROOT / "coco_annotations" / f"{flir_split}.json"
    rgb_img_dir = FLIR_ROOT / "visible"  / flir_split
    ir_img_dir  = FLIR_ROOT / "thermal"  / flir_split

    with open(ann_path) as f:
        coco = json.load(f)

    # image_id -> image info
    img_info = {img["id"]: img for img in coco["images"]}

    # image_id -> annotations
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        ann_map[ann["image_id"]].append(ann)

    out_rgb = OUT_ROOT / out_split / "rgb"
    out_ir  = OUT_ROOT / out_split / "ir"
    out_lbl = OUT_ROOT / out_split / "labels"
    for d in [out_rgb, out_ir, out_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    rgb_txt_lines = []
    ir_txt_lines  = []
    skipped = 0

    for img_id, info in img_info.items():
        fname  = info["file_name"]
        img_w  = info["width"]
        img_h  = info["height"]
        stem   = Path(fname).stem

        rgb_path = rgb_img_dir / fname
        ir_path  = ir_img_dir  / fname

        if not rgb_path.exists() or not ir_path.exists():
            skipped += 1
            continue

        yolo_lines = coco_to_yolo(ann_map[img_id], img_w, img_h)
        if not yolo_lines:
            skipped += 1
            continue

        shutil.copy2(rgb_path, out_rgb / fname)
        shutil.copy2(ir_path,  out_ir  / fname)

        lbl_path = out_lbl / f"{stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines))

        rgb_txt_lines.append(str(out_rgb / fname))
        ir_txt_lines.append(str(out_ir   / fname))

    print(f"[{flir_split}] Complete: {len(rgb_txt_lines)} files, Skipped: {skipped}")
    return rgb_txt_lines, ir_txt_lines


def merge_with_m3fd(out_split, rgb_lines, ir_lines):
    for modal, new_lines in [("rgb", rgb_lines), ("ir", ir_lines)]:
        m3fd_txt   = M3FD_TXT_DIR / f"{out_split}_{modal}.txt"
        merged_txt = M3FD_TXT_DIR / f"{out_split}_{modal}_merged.txt"

        existing = m3fd_txt.read_text().strip().splitlines() if m3fd_txt.exists() else []
        merged   = existing + new_lines
        merged_txt.write_text("\n".join(merged))
        print(f"  {merged_txt.name}: M3FD {len(existing)} + FLIR {len(new_lines)} = {len(merged)}")


def write_yaml():
    yaml_path = BASE / "src/engine/CFT_repo/M3FD_FLIR.yaml"
    m3fd_txt  = M3FD_TXT_DIR.as_posix()
    content = f"""# M3FD + FLIR ADAS Aligned Unified Dataset
# M3FD 6 classes + FLIR People/Car augmentation

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
    if not FLIR_ROOT.exists():
        print(f"[Error] FLIR_ADAS directory not found: {FLIR_ROOT}")
        exit(1)

    print("── Converting FLIR ADAS to YOLO format ──")
    all_rgb = {"train": [], "val": []}
    all_ir  = {"train": [], "val": []}

    for flir_split, out_split in SPLITS.items():
        rgb, ir = process_split(flir_split, out_split)
        all_rgb[out_split] += rgb
        all_ir[out_split]  += ir

    print("\n── Merging with M3FD ──")
    for split in ["train", "val"]:
        merge_with_m3fd(split, all_rgb[split], all_ir[split])

    write_yaml()
    print("\nDone! Ready to train with M3FD_FLIR.yaml.")
