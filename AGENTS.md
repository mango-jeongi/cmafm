# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

CMAFM is a **Python monorepo** for RGB + LWIR (thermal) multispectral object detection. The primary interactive surface is the **Streamlit dashboard** in `RGBThermal/dashboard.py`. A separate YOLOv5/CFT stack lives under `CFT_repo/` (training/eval only; not required for the dashboard).

### Services

| Service | Port | Required? | Start command |
|---------|------|-----------|---------------|
| Streamlit dashboard | 8501 | Yes (UI E2E) | See below |
| CFT YOLO training (`CFT_repo/`) | — | Optional | `cd CFT_repo && python train.py ...` |

There is no database, Docker Compose stack, or npm frontend.

### Python environment

- Use the repo-root venv: `/workspace/venv` (create with `python3 -m venv venv` after `python3.12-venv` is installed on Debian/Ubuntu).
- **CPU VMs (no NVIDIA GPU):** install CPU PyTorch before other deps. The committed `requirements.txt` pins CUDA wheels (`torch==2.6.0+cu124`), which will fail on CPU-only hosts.
- **Recommended CPU install** (matches dashboard + `inference.py` checkpoint loading):

```bash
cd /workspace
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install streamlit==1.51.0 opencv-python numpy pandas Pillow albumentations plotly pycocotools matplotlib tqdm gdown
```

- **GPU hosts:** follow `RGBThermal/MANUAL.md` or `install.bat` (CUDA 12.4 index) instead of the CPU index URL.

### Checkpoints and data (not in git)

- `*.pth` / `RGBThermal/runs/` are **gitignored**. The dashboard expects `RGBThermal/runs/best.pth` by default.
- M3FD images (`RGBThermal/data/M3FD/Vis`, `Ir`) are also gitignored. The **sample tab** needs them; **image upload tab** only needs a local checkpoint + user-provided RGB/thermal pair.
- M3FD download: `RGBThermal/MANUAL.md` (gdown Google Drive folder).
- **Bundled smoke-test images:** `RGBThermal/demo_assets/demo_rgb.jpg` and `demo_thermal.jpg` (upload tab; still need a local `runs/best.pth`).
- For a quick smoke test without the full dataset, build and save a checkpoint once (loads ImageNet backbones, ~765MB):

```bash
cd /workspace/RGBThermal
../venv/bin/python -c "import torch; from pathlib import Path; from config import Config; from model import build_model; cfg=Config(); m=build_model(cfg.model,num_classes=cfg.data.num_classes); Path('runs').mkdir(exist_ok=True); torch.save({'model':m.state_dict(),'config':cfg,'epoch':0,'best_map':0.0},'runs/best.pth')"
```

### Run the dashboard (development)

```bash
cd /workspace/RGBThermal
export KMP_DUPLICATE_LIB_OK=TRUE
/workspace/venv/bin/streamlit run dashboard.py --server.port 8501 --server.headless true --server.maxUploadSize 500
```

Health check: `curl -sf http://127.0.0.1:8501/_stcore/health`

Windows shortcut: `run_dashboard.bat` (uses `venv` if present).

### Lint / tests

- **No repo-wide linter or pytest suite** is configured.
- Syntax smoke check: `venv/bin/python -m py_compile RGBThermal/*.py`
- Ad-hoc scripts: `RGBThermal/test_encoding.py` (needs M3FD paths), `RGBThermal/inference.py` (needs checkpoint).
- **PyTorch ≥2.6 note:** `torch.load` defaults to `weights_only=True`. `dashboard.py` passes `weights_only=False`; `inference.py` does not — use PyTorch 2.5.x on CPU or pass `weights_only=False` when loading trusted local checkpoints.

### CFT_repo (optional)

```bash
venv/bin/pip install -r CFT_repo/requirements.txt
cd CFT_repo && python train.py --cfg models/transformer/yolov5l_cmafm_M3FD.yaml ...
```

Requires prepared multispectral YOLO data under `CFT_repo/data/` (gitignored) and pretrained `.pt` weights.
