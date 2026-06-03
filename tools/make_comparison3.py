"""
Day/Night scenes visualization:
  Row A: RGB | Thermal | CMAFM-YOLO detection
Saves 3 samples for night and 3 samples for day into separate PNG files.
"""
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
CFT_DIR  = os.path.join(REPO_ROOT, "cft_engine")
sys.path.insert(0, CFT_DIR)
os.chdir(CFT_DIR)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

BASE     = Path(REPO_ROOT)
WEIGHTS  = BASE / "weights" / "best.pt"
RGB_DIR  = BASE / "src/fusion/data/M3FD_yolo" / "val" / "rgb"
IR_DIR   = BASE / "src/fusion/data/M3FD_yolo" / "val" / "ir"
OUT_DIR  = BASE / "src/fusion/runs/paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['People','Car','Bus','Motorcycle','Lamp','Truck']
COLORS  = [(0,255,0),(0,128,255),(255,0,0),(0,0,255),(255,255,0),(255,0,255)]
IMG_H   = 300

device = select_device('0')
model  = attempt_load(str(WEIGHTS), map_location=device)
model.half().eval()
print("Model loaded")

def read(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def lb(img, s=640):
    h,w = img.shape[:2]; r=s/max(h,w)
    nw,nh = int(w*r),int(h*r)
    img = cv2.resize(img,(nw,nh))
    dw,dh = (s-nw)//2,(s-nh)//2
    img = cv2.copyMakeBorder(img,dh,s-nh-dh,dw,s-nw-dw,cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,dw,dh

def rh(img, H=IMG_H):
    h,w=img.shape[:2]; return cv2.resize(img,(int(w*H/h),H))

def title(img, text, col=(230,230,230)):
    o=img.copy()
    cv2.rectangle(o,(0,0),(img.shape[1],28),(20,20,20),-1)
    cv2.putText(o,text,(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,1,cv2.LINE_AA)
    return o

def boxes_to_orig(boxes, r, dw, dh, h0, w0):
    """Inverse transform letterbox coordinates back to original image space"""
    boxes = boxes.clone()
    boxes[:, 0] = (boxes[:, 0] - dw) / r   # x1
    boxes[:, 1] = (boxes[:, 1] - dh) / r   # y1
    boxes[:, 2] = (boxes[:, 2] - dw) / r   # x2
    boxes[:, 3] = (boxes[:, 3] - dh) / r   # y2
    boxes[:, 0].clamp_(0, w0)
    boxes[:, 1].clamp_(0, h0)
    boxes[:, 2].clamp_(0, w0)
    boxes[:, 3].clamp_(0, h0)
    return boxes

def detect(fname):
    rgb0=read(RGB_DIR/fname); ir0=read(IR_DIR/fname)
    h0,w0=rgb0.shape[:2]
    lr,r,dw,dh=lb(rgb0); li,*_=lb(ir0)
    tr=torch.from_numpy(lr[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).half().to(device)/255.
    ti=torch.from_numpy(li[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).half().to(device)/255.
    with torch.no_grad(): pred=model(tr,ti)[0]
    dets=non_max_suppression(pred,0.35,0.45)[0]
    out=rgb0.copy()
    if dets is not None and len(dets):
        dets[:,:4]=boxes_to_orig(dets[:,:4], r, dw, dh, h0, w0).round()
        for *xyxy,conf,cls in dets:
            x1,y1,x2,y2=map(int,xyxy); c=int(cls); col=COLORS[c%len(COLORS)]
            cv2.rectangle(out,(x1,y1),(x2,y2),col,2)
            cv2.putText(out,f"{CLASSES[c]} {conf:.2f}",(x1,max(y1-4,12)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1)
    return rgb0, ir0, out

def get_samples(n=3, night=True):
    res=[]
    for f in sorted(RGB_DIR.iterdir()):
        if not (IR_DIR/f.name).exists(): continue
        img=read(f)
        if img is None: continue
        b=img.mean()
        if (night and b<55) or (not night and 75<=b<=150):
            res.append(f.name)
        if len(res)>=n: break
    return res

def make(samples, headline, out_path):
    rows=[]
    for fname in samples:
        rgb0,ir0,det=detect(fname)
        a=title(rh(rgb0), "(a) RGB Input")
        # Convert IR to colourmap
        ir_gray=cv2.cvtColor(ir0,cv2.COLOR_BGR2GRAY)
        ir_color=cv2.applyColorMap(ir_gray,cv2.COLORMAP_INFERNO)
        b_color=title(rh(ir_color),"(b) Thermal (Inferno)")
        c_=title(rh(det),  "(c) CMAFM-YOLO Detection",(80,255,80))
        row=np.hstack([a, b_color, c_])
        rows.append(row)

    max_w=max(r.shape[1] for r in rows)
    rows=[cv2.copyMakeBorder(r,0,0,0,max_w-r.shape[1],cv2.BORDER_CONSTANT,value=(15,15,15))
          for r in rows]
    sep=np.full((4,max_w,3),50,dtype=np.uint8)
    final=rows[0]
    for r in rows[1:]: final=np.vstack([final,sep,r])

    hdr=np.full((46,max_w,3),15,dtype=np.uint8)
    cv2.putText(hdr,headline,(10,31),cv2.FONT_HERSHEY_SIMPLEX,0.75,(210,210,210),2,cv2.LINE_AA)
    final=np.vstack([hdr,final])
    cv2.imencode('.png',final)[1].tofile(str(out_path))
    print(f"Saved: {out_path.name}")

night_s=get_samples(3,night=True)
day_s  =get_samples(3,night=False)
print(f"Night: {night_s}")
print(f"Day:   {day_s}")

make(night_s,"Night Scene  |  RGB / Thermal / CMAFM-YOLO Detection",
     OUT_DIR/"fig_cmafm_yolo_night.png")
make(day_s,  "Day Scene    |  RGB / Thermal / CMAFM-YOLO Detection",
     OUT_DIR/"fig_cmafm_yolo_day.png")
print("Done!")
