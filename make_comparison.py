"""
CMAFM-YOLO vs CMAFM(Faster R-CNN) 주야간 비교 이미지 생성 스크립트
- YOLO: best.pt로 추론
- Faster R-CNN: 기존 paper_figures 사용
- 출력: paper_figures/yolo_night_samples.png, yolo_day_samples.png
"""
import sys, os
sys.path.insert(0, "d:/★RGB-LWIR(멘토ver-최종)/CFT_repo")
os.chdir("d:/★RGB-LWIR(멘토ver-최종)/CFT_repo")

import cv2
import torch
import numpy as np
from pathlib import Path

# ── 설정 ────────────────────────────────────────────────────────────────────
WEIGHTS   = "runs/train/cmafm_m3fd_flir/weights/best.pt"
RGB_DIR   = Path("d:/★RGB-LWIR(멘토ver-최종)/RGBThermal/data/M3FD_yolo/val/rgb")
IR_DIR    = Path("d:/★RGB-LWIR(멘토ver-최종)/RGBThermal/data/M3FD_yolo/val/ir")
OUT_DIR   = Path("d:/★RGB-LWIR(멘토ver-최종)/RGBThermal/runs/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES   = ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']
COLORS    = [(0,255,0),(0,128,255),(255,0,0),(0,0,255),(255,255,0),(255,0,255)]
IMG_SIZE  = 640
CONF_THR  = 0.35
IOU_THR   = 0.45

# ── 모델 로드 ────────────────────────────────────────────────────────────────
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

device = select_device('0')
model  = attempt_load(WEIGHTS, map_location=device)
model.half()
model.eval()
stride = int(model.stride.max())
print(f"모델 로드 완료: {WEIGHTS}")

# ── 대표 샘플 선별 ────────────────────────────────────────────────────────────
def get_samples(n=3, night=True):
    files = sorted(RGB_DIR.iterdir())
    samples = []
    for f in files:
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        b = img.mean()
        if (night and b < 55) or (not night and 80 <= b <= 140):
            ir_path = IR_DIR / f.name
            if ir_path.exists():
                samples.append(f.name)
        if len(samples) >= n:
            break
    return samples

# ── 전처리 ────────────────────────────────────────────────────────────────────
def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    r = new_shape / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    img = cv2.resize(img, (new_w, new_h))
    dw = (new_shape - new_w) // 2
    dh = (new_shape - new_h) // 2
    img = cv2.copyMakeBorder(img, dh, new_shape-new_h-dh, dw, new_shape-new_w-dw,
                              cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, r, dw, dh

def preprocess(path):
    img0 = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    img, r, dw, dh = letterbox(img0, IMG_SIZE)
    img_t = torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0)
    img_t = img_t.half().to(device) / 255.0
    return img0, img_t, r, dw, dh

# ── 추론 ────────────────────────────────────────────────────────────────────
def infer(fname):
    rgb_path = RGB_DIR / fname
    ir_path  = IR_DIR  / fname
    img0_rgb, t_rgb, r, dw, dh = preprocess(rgb_path)
    _, t_ir, _, _, _            = preprocess(ir_path)

    with torch.no_grad():
        pred = model(t_rgb, t_ir)[0]
    pred = non_max_suppression(pred, CONF_THR, IOU_THR)[0]

    result = img0_rgb.copy()
    h0, w0 = img0_rgb.shape[:2]
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords((IMG_SIZE, IMG_SIZE), pred[:, :4],
                                    (h0, w0), ratio_pad=((r,r),(dh,dw))).round()
        for *xyxy, conf, cls in pred:
            x1,y1,x2,y2 = map(int, xyxy)
            c = int(cls)
            col = COLORS[c % len(COLORS)]
            cv2.rectangle(result, (x1,y1), (x2,y2), col, 2)
            label = f"{CLASSES[c]} {conf:.2f}"
            cv2.putText(result, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return img0_rgb, result

# ── 비교 패널 생성 ────────────────────────────────────────────────────────────
def make_panel(samples, label, out_path):
    panels = []
    for fname in samples:
        rgb_img, yolo_det = infer(fname)
        ir_img = cv2.imdecode(
            np.fromfile(str(IR_DIR/fname), dtype=np.uint8), cv2.IMREAD_COLOR)

        H = 300
        def resize_h(img):
            h, w = img.shape[:2]
            nw = int(w * H / h)
            return cv2.resize(img, (nw, H))

        rgb_r   = resize_h(rgb_img)
        ir_r    = resize_h(ir_img)
        yolo_r  = resize_h(yolo_det)

        # 헤더 텍스트 추가
        def add_title(img, text):
            out = img.copy()
            cv2.putText(out, text, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1,
                        cv2.LINE_AA)
            return out

        rgb_r  = add_title(rgb_r,  "(a) RGB Input")
        ir_r   = add_title(ir_r,   "(b) Thermal Input")
        yolo_r = add_title(yolo_r, "(c) CMAFM-YOLO")

        row = np.hstack([rgb_r, ir_r, yolo_r])
        panels.append(row)

    # 행 폭 통일
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        dw = max_w - p.shape[1]
        p = cv2.copyMakeBorder(p, 0, 0, 0, dw, cv2.BORDER_CONSTANT, value=(30,30,30))
        padded.append(p)

    # 구분선
    sep = np.full((4, max_w, 3), 80, dtype=np.uint8)
    final = padded[0]
    for p in padded[1:]:
        final = np.vstack([final, sep, p])

    # 제목 바
    title_bar = np.full((40, max_w, 3), 30, dtype=np.uint8)
    cv2.putText(title_bar, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220,220,220), 2)
    final = np.vstack([title_bar, final])

    cv2.imencode('.png', final)[1].tofile(str(out_path))
    print(f"저장: {out_path}")

# ── 실행 ────────────────────────────────────────────────────────────────────
print("야간 샘플 선별...")
night_samples = get_samples(3, night=True)
print(f"  → {night_samples}")

print("주간 샘플 선별...")
day_samples   = get_samples(3, night=False)
print(f"  → {day_samples}")

make_panel(night_samples, "Night Scene — RGB / Thermal / CMAFM-YOLO Detection",
           OUT_DIR / "yolo_night_comparison.png")
make_panel(day_samples,   "Day Scene   — RGB / Thermal / CMAFM-YOLO Detection",
           OUT_DIR / "yolo_day_comparison.png")

print("\n완료!")
