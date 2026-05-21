"""
M3FD 데이터셋을 CFT(YOLOv5) 형식으로 변환하는 스크립트.
- VOC XML → YOLO txt 변환
- 8:2 train/val 분할 (seed=42, CMAFM과 동일)
- RGB(Vis) / Thermal(Ir) 각각 txt 파일 생성
"""

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────────────
M3FD_ROOT = Path(r"C:\CFT_M3FD\data\M3FD_yolo\src_m3fd")
OUT_ROOT   = Path(r"C:\CFT_M3FD\data\M3FD_yolo")

VIS_DIR = M3FD_ROOT / "Vis"
IR_DIR  = M3FD_ROOT / "Ir"
ANN_DIR = M3FD_ROOT / "Annotation"

# 이미지 복사 없이 영문 경로 txt만 생성 (junction 경유)
USE_JUNCTION = True  # True: src_m3fd junction 경로로 txt 작성, 이미지 복사 생략

# ── 클래스 (CMAFM과 동일 순서) ──────────────────────────────────────────────
CLASSES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
CLS2ID  = {c: i for i, c in enumerate(CLASSES)}

VAL_RATIO = 0.2
SEED      = 42


def xml_to_yolo(xml_path: Path, img_w: int, img_h: int):
    """VOC XML → YOLO [(cls_id, cx, cy, w, h), ...] (정규화)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLS2ID:
            continue
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        labels.append((CLS2ID[name], cx, cy, w, h))
    return labels


def main():
    # 전체 이미지 ID 수집
    all_ids = sorted([p.stem for p in ANN_DIR.glob("*.xml")])
    print(f"전체 이미지 수: {len(all_ids)}")

    # 8:2 분할 (seed=42)
    random.seed(SEED)
    ids = all_ids.copy()
    random.shuffle(ids)
    n_val = int(len(ids) * VAL_RATIO)
    val_ids   = ids[:n_val]
    train_ids = ids[n_val:]
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    # img2label_paths는 경로의 /images/ → /labels/ 로 치환함
    # 구조: OUT_ROOT/{split}/rgb/images/{id}.png  →  OUT_ROOT/{split}/rgb/labels/{id}.txt
    for split in ("train", "val"):
        for mod in ("rgb", "ir"):
            (OUT_ROOT / split / mod / "images").mkdir(parents=True, exist_ok=True)
            (OUT_ROOT / split / mod / "labels").mkdir(parents=True, exist_ok=True)

    def process(img_ids, split):
        rgb_list, ir_list = [], []
        for img_id in img_ids:
            vis_path = VIS_DIR / f"{img_id}.png"
            ir_path  = IR_DIR  / f"{img_id}.png"
            xml_path = ANN_DIR / f"{img_id}.xml"
            if not (vis_path.exists() and ir_path.exists() and xml_path.exists()):
                continue

            img_w, img_h = 1024, 768
            labels = xml_to_yolo(xml_path, img_w, img_h)

            # images 폴더 내 심볼릭 링크(junction 불가) → 하드링크 시도, 실패 시 복사
            rgb_img = OUT_ROOT / split / "rgb" / "images" / f"{img_id}.png"
            ir_img  = OUT_ROOT / split / "ir"  / "images" / f"{img_id}.png"
            if not rgb_img.exists():
                try:
                    os.link(vis_path, rgb_img)
                except Exception:
                    import shutil; shutil.copy2(vis_path, rgb_img)
            if not ir_img.exists():
                try:
                    os.link(ir_path, ir_img)
                except Exception:
                    import shutil; shutil.copy2(ir_path, ir_img)

            # YOLO 라벨 파일 저장
            lbl_content = "\n".join(
                f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for c, cx, cy, w, h in labels
            )
            (OUT_ROOT / split / "rgb" / "labels" / f"{img_id}.txt").write_text(lbl_content)
            (OUT_ROOT / split / "ir"  / "labels" / f"{img_id}.txt").write_text(lbl_content)

            rgb_list.append(str(rgb_img))
            ir_list.append(str(ir_img))

        # txt 리스트 파일 저장
        (OUT_ROOT / f"{split}_rgb.txt").write_text("\n".join(rgb_list))
        (OUT_ROOT / f"{split}_ir.txt").write_text("\n".join(ir_list))
        print(f"[{split}] RGB: {len(rgb_list)}장  IR: {len(ir_list)}장 처리 완료")

    process(train_ids, "train")
    process(val_ids,   "val")
    print("\n완료! YOLO 형식 데이터셋 저장 위치:", OUT_ROOT)


if __name__ == "__main__":
    main()
