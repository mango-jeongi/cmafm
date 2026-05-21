"""
YOLOv5l pretrained 가중치를 CMAFM-YOLO 모델에 구조적 매핑으로 부분 로드.

CMAFM 구조 → YOLOv5l 매핑:
  Stream1: model.0~4   ← yolo model.0~4  (P1~P3)
  Stream2: model.5~9   ← yolo model.0~4  (동일 가중치 복사)
  P4 stream1: model.13~14 ← yolo model.5~6
  P4 stream2: model.15~16 ← yolo model.5~6
  P5 stream1: model.20~22 ← yolo model.7~9 (SPP→SPP 호환)
  P5 stream2: model.23~25 ← yolo model.7~9
  Head: model.32~46   ← yolo model.10~23 (일부 채널 다를 수 있음)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.yolo import Model
from utils.torch_utils import select_device

PRETRAINED = 'C:/Users/CAU/.cache/torch/hub/checkpoints/yolov5l.pt'
CFG        = 'models/transformer/yolov5l_cmafm_M3FD.yaml'
OUT        = 'weights/cmafm_pretrained_init.pt'

os.makedirs('weights', exist_ok=True)
device = select_device('cpu')

print("Loading CMAFM model structure...")
model = Model(CFG, ch=3, nc=6).to(device)
model_sd = model.state_dict()

print("Loading YOLOv5l pretrained weights...")
ckpt = torch.load(PRETRAINED, map_location='cpu', weights_only=False)
pretrained_sd = ckpt.get('model', ckpt)
if hasattr(pretrained_sd, 'state_dict'):
    pretrained_sd = pretrained_sd.state_dict()

# pretrained 키를 'model.X.*' 형식으로 정규화
pt_sd = {}
for k, v in pretrained_sd.items():
    nk = k if k.startswith('model.') else ('model.' + k)
    pt_sd[nk] = v

# CMAFM → YOLOv5l 레이어 매핑
# (cmafm_layer_idx, pretrained_layer_idx)
LAYER_MAP = [
    # Stream 1 (P1~P3)
    (0,  0), (1,  1), (2,  2), (3,  3), (4,  4),
    # Stream 2 (P1~P3) — 동일 pretrained 가중치 복사
    (5,  0), (6,  1), (7,  2), (8,  3), (9,  4),
    # P4 stream 1
    (13, 5), (14, 6),
    # P4 stream 2
    (15, 5), (16, 6),
    # P5 stream 1 (SPP 호환)
    (20, 7), (21, 8), (22, 9),
    # P5 stream 2
    (23, 7), (24, 8), (25, 9),
    # Head (채널이 달라 shape mismatch 발생 시 skip)
    (32, 10), (35, 13), (36, 14), (39, 17),
    (40, 18), (42, 20), (43, 21), (45, 23),
]

new_sd = dict(model_sd)  # 기본: 랜덤 초기화 유지
matched, shape_miss, total_tried = 0, 0, 0

for (cm_idx, pt_idx) in LAYER_MAP:
    # 해당 레이어의 모든 서브키 수집
    cm_prefix = f'model.{cm_idx}.'
    pt_prefix = f'model.{pt_idx}.'

    cm_keys = [k for k in model_sd if k.startswith(cm_prefix)]
    for ck in cm_keys:
        suffix = ck[len(cm_prefix):]
        pk = pt_prefix + suffix
        total_tried += 1
        if pk not in pt_sd:
            continue
        if pt_sd[pk].shape != model_sd[ck].shape:
            shape_miss += 1
            continue
        new_sd[ck] = pt_sd[pk]
        matched += 1

print(f"\nStructural mapping load summary:")
print(f"  Tried   : {total_tried}")
print(f"  Matched : {matched} ({matched/total_tried*100:.1f}%)")
print(f"  Shape mismatch (random init): {shape_miss}")
print(f"  CMAFM/Add2/Detect (random init): {len(model_sd) - total_tried}")

model.load_state_dict(new_sd)

ckpt_save = {'model': model, 'optimizer': None, 'epoch': -1}
torch.save(ckpt_save, OUT)
print(f"\nSaved: {OUT}")
print("Use: --weights weights/cmafm_pretrained_init.pt")
