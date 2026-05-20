"""
LLVIP 데이터셋 → YOLO 포맷 변환 + M3FD와 통합 스크립트

LLVIP 폴더 구조 가정:
  d:/★RGB-LWIR(멘토ver-최종)/LLVIP/
      visible/train/   (RGB)
      visible/test/
      infrared/train/  (IR)
      infrared/test/
      Annotations/     (VOC XML, train+test 혼합)

실행:
  python prepare_llvip.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random

# ── 경로 설정 ──────────────────────────────────────────────────────────────
LLVIP_ROOT   = Path("d:/★RGB-LWIR(멘토ver-최종)/LLVIP")
OUT_ROOT     = Path("d:/★RGB-LWIR(멘토ver-최종)/RGBThermal/data/LLVIP_yolo")
M3FD_TXT_DIR = Path("d:/★RGB-LWIR(멘토ver-최종)/RGBThermal/data/M3FD_yolo")

# LLVIP는 보행자만 → M3FD 클래스 인덱스에서 People=0
LLVIP_CLASS_ID = 0  # M3FD names: ['People','Car','Bus','Motorcycle','Lamp','Truck']

SPLITS = ["train", "test"]  # LLVIP split 이름 (test → val로 사용)


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int):
    """VOC XML → YOLO txt 라인 리스트 반환"""
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
    print(f"[{split}] {len(img_files)}개 이미지 처리 중...")

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

        # 이미지 크기 읽기 (PIL 없으면 고정값 사용)
        try:
            from PIL import Image
            with Image.open(rgb_path) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 1280, 1024  # LLVIP 기본 해상도

        yolo_lines = voc_to_yolo(xml_path, img_w, img_h)
        if not yolo_lines:
            skipped += 1
            continue

        # 이미지 복사
        shutil.copy2(rgb_path, rgb_out_img / rgb_path.name)
        shutil.copy2(ir_path,  ir_out_img  / ir_path.name)

        # 라벨 저장
        lbl_path = lbl_out / f"{stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines))

        rgb_txt_lines.append(str(rgb_out_img / rgb_path.name))
        ir_txt_lines.append(str(ir_out_img   / ir_path.name))

    print(f"  → 완료: {len(rgb_txt_lines)}개, 건너뜀: {skipped}개")
    return out_split, rgb_txt_lines, ir_txt_lines


def merge_with_m3fd(split: str, rgb_lines: list, ir_lines: list):
    """M3FD txt 파일에 LLVIP 경로 추가해서 merged txt 생성"""
    suffix = "train" if split == "train" else "val"

    for modal, new_lines in [("rgb", rgb_lines), ("ir", ir_lines)]:
        m3fd_txt = M3FD_TXT_DIR / f"{suffix}_{modal}.txt"
        merged_txt = M3FD_TXT_DIR / f"{suffix}_{modal}_merged.txt"

        if m3fd_txt.exists():
            existing = m3fd_txt.read_text().strip().splitlines()
        else:
            existing = []
            print(f"  [경고] {m3fd_txt} 없음 — LLVIP만으로 구성")

        merged = existing + new_lines
        random.shuffle(merged)
        merged_txt.write_text("\n".join(merged))
        print(f"  {merged_txt.name}: M3FD {len(existing)} + LLVIP {len(new_lines)} = {len(merged)}개")


def write_yaml():
    yaml_path = Path("d:/★RGB-LWIR(멘토ver-최종)/CFT_repo/data/multispectral/M3FD_LLVIP.yaml")
    m3fd_txt  = M3FD_TXT_DIR.as_posix()
    content = f"""# M3FD + LLVIP 통합 데이터셋
# M3FD 6클래스 + LLVIP 보행자 보강

train_rgb: {m3fd_txt}/train_rgb_merged.txt
val_rgb:   {m3fd_txt}/val_rgb_merged.txt
train_ir:  {m3fd_txt}/train_ir_merged.txt
val_ir:    {m3fd_txt}/val_ir_merged.txt

nc: 6
names: ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']
"""
    yaml_path.write_text(content)
    print(f"\nyaml 저장: {yaml_path}")


if __name__ == "__main__":
    if not LLVIP_ROOT.exists():
        print(f"[오류] LLVIP 폴더 없음: {LLVIP_ROOT}")
        print("먼저 LLVIP 다운로드 후 위 경로에 압축 해제하세요.")
        exit(1)

    all_rgb = {"train": [], "val": []}
    all_ir  = {"train": [], "val": []}

    for split in SPLITS:
        out_split, rgb_lines, ir_lines = process_split(split)
        all_rgb[out_split] += rgb_lines
        all_ir[out_split]  += ir_lines

    print("\n── M3FD와 통합 중 ──")
    for split in ["train", "val"]:
        merge_with_m3fd(split, all_rgb[split], all_ir[split])

    write_yaml()
    print("\n완료! M3FD_LLVIP.yaml 으로 학습 가능합니다.")
