# CMAFM: Cross-Modal Attention Fusion Module for RGB-Thermal Object Detection

> **Anonymous WACV 2027 Applications Track Supplementary Code** — Submission ID 3068  
> *"CMAFM: Cross-Modal Attention Fusion Module for Efficient Real-Time RGB-Thermal Object Detection on Edge Systems"*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://python.org)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](Dockerfile)

---

## Overview

CMAFM is a lightweight, plug-and-play **Cross-Modal Attention Fusion Module** that merges RGB and thermal feature maps within a dual-stream YOLOv5-based multispectral detector. It combines:

- **Global Channel Cross-Attention** — O(C²) via Global Average Pooling, enabling modality-aware feature recalibration.
- **Local Spatial Cross-Gating** — O(CHW) via Depthwise Convolution, preserving fine spatial correspondence between modalities.

This eliminates the quadratic spatial complexity O(H²W²C) of transformer-based fusion while achieving state-of-the-art detection accuracy on M3FD and FLIR ADAS v2.

### Key Results

| Dataset & Split | mAP@0.5 | mAP@0.5:0.95 | Latency | Hardware |
|:---|:---:|:---:|:---:|:---|
| M3FD Validation (10-seed avg.) | **85.75% ± 0.28%** | **56.71% ± 0.26%** | 17.2 ms | RTX 4070 Laptop (PyTorch) |
| M3FD Validation (TensorRT FP16) | **85.60%** | **56.62%** | ~48–50 ms | Jetson Orin Nano 8 GB |
| FLIR Aligned Test (TensorRT FP16) | **89.67%** | **53.97%** | ~48–50 ms | Jetson Orin Nano 8 GB |

---

## Repository Structure

```
CMAFM/
├── README.md                    ← This file
├── REPRODUCIBILITY.md           ← ML Reproducibility Checklist answers
├── UPDATED_TENSORRT_METRICS.md  ← Edge deployment benchmark report
├── LICENSE
├── Dockerfile                   ← Reproducibility container (CPU + GPU)
├── docker-compose.yml           ← One-command container launch
├── run_demo.py                  ← Architecture verification (no weights needed)
├── requirements.txt
├── .env.example                 ← Environment variable template
│
├── data/                        ← Dataset configuration & bundled sample pairs
│   ├── samples/                 ← 3 registered RGB+Thermal pairs for dashboard demo
│   ├── M3FD_FLIR.yaml           ← Training config (M3FD + FLIR supplementary)
│   ├── m3fd_rgbt.yaml           ← Training config (M3FD only)
│   ├── yolov5l_cmafm_M3FD.yaml  ← CMAFM-YOLO model architecture config
│   └── hyp.scratch.yaml         ← Hyperparameter config
│
├── src/
│   ├── fusion/                  ← Core CMAFM implementation
│   │   ├── model.py             ← Faster R-CNN CMAFM (PyTorch native)
│   │   ├── model_yolo.py        ← CMAFM module adapter for YOLOv5
│   │   ├── ablation_models.py   ← Ablation variants (channel-only, spatial-only)
│   │   ├── config.py            ← Training & evaluation configuration dataclasses
│   │   ├── dataset.py           ← Multispectral dataset loader
│   │   ├── train.py             ← Faster R-CNN training loop
│   │   ├── train_yolo.py        ← CMAFM-YOLO training entry point
│   │   ├── evaluate.py          ← Evaluation & mAP calculation
│   │   ├── inference.py         ← Single-image inference pipeline
│   │   ├── dashboard.py         ← Interactive Streamlit demo
│   │   ├── run_ablation.py      ← Ablation study runner
│   │   ├── visualize_pipeline.py← Architecture visualization
│   │   ├── analyze_night.py     ← Day/night performance breakdown
│   │   ├── analyze_modality.py  ← RGB vs. Thermal modality analysis
│   │   ├── generate_report_plots.py ← Paper figure generation
│   │   ├── plot_ablation_curves.py  ← Ablation training curves
│   │   ├── prepare_m3fd_yolo.py ← M3FD → YOLO format conversion
│   │   ├── convert_to_yolo.py   ← General annotation converter
│   │   └── download_data.py     ← Dataset download helper
│   └── engine/
│       ├── cft_engine_patches/  ← Drop-in patches for the base CFT engine
│       │   └── train.py         ← Patched trainer (DDP fix, CMAFM integration)
│       └── engine_fixes/
│           ├── cmafm.py         ← CMAFM module (injected into cft_engine/models/)
│           ├── patch_parser.py  ← Automated AST-based engine patch injector
│           └── bootstrap_weights.py ← YOLOv5l → dual-stream weight mapper
│
├── scripts/
│   ├── setup.sh                 ← Linux/HPC environment setup & engine clone
│   ├── setup.ps1                ← Windows environment setup (uv-based)
│   ├── submit.sh                ← SLURM job launcher wrapper
│   ├── prepare_all.py           ← Unified M3FD + FLIR dataset preparation
│   ├── prepare_flir.py          ← FLIR ADAS v2 download & format conversion
│   ├── prepare_flir_validation.py ← FLIR official test split preparation
│   ├── prepare_m3fd_validation.py ← M3FD validation split preparation
│   ├── prepare_llvip.py         ← LLVIP dataset preparation (optional)
│   └── build_supplementary.sh  ← Package repo as anonymized supplementary zip
│
├── slurm/                       ← HPC 10-seed job array scripts
│   ├── submit_cmafm_yolo_m3fd.sh← Primary CMAFM-YOLO training (10 seeds)
│   ├── submit_yolov5_eval.sh    ← Full 10-seed evaluation sweep
│   ├── submit_cft_m3fd_flir.sh  ← CFT baseline comparison
│   ├── submit_cft_resume.sh     ← CFT baseline resumption
│   ├── submit_channel_ablation.sh ← Channel-only ablation
│   ├── submit_spatial_ablation.sh ← Spatial-only ablation
│   ├── run_ablation.sh          ← Local ablation runner
│   └── run_paper_tables.sh      ← Paper table generation
│
├── tools/                       ← Dataset verification & metric tools
│   ├── load_pretrained_partial.py ← Map YOLOv5l weights to dual-stream
│   ├── verify_flir.py           ← FLIR alignment visual check
│   ├── verify_m3fd.py           ← M3FD dataset integrity check
│   ├── aggregate_final_metrics.py ← Aggregate 10-seed HPC results
│   ├── generate_paper_tables.py ← Generate LaTeX/markdown tables
│   ├── get_efficiency.py        ← GFLOPs & FPS profiling
│   ├── split_day_night.py       ← Day/night dataset split
│   ├── compare_flir_aligned.py  ← FLIR alignment verification
│   ├── make_comparison3.py      ← Side-by-side qualitative figures
│   ├── make_video_demo.py       ← FLIR driving sequence video demo
│   └── view_flir_cars.py        ← FLIR car class visualization
│
└── jetson_deploy/               ← Jetson Orin Nano edge deployment
    ├── README.md                ← Jetson-specific instructions
    ├── artifacts/               ← Saved TensorRT evaluation results
    ├── compat/                  ← Checkpoint compatibility shim
    ├── scripts/                 ← Engine build & benchmark scripts
    ├── src/                     ← TensorRT inference & evaluation modules
    ├── tests/                   ← CPU preprocessing & dashboard UI tests
    └── vendor/                  ← Vendor dependencies placeholder
```

---

## Quickstart: Architecture Verification (No Weights Needed)

Verify the CMAFM module dimensions and forward pass are correct:

```bash
# Linux / Mac
source .venv/bin/activate
python run_demo.py

# Windows
.venv\Scripts\Activate.ps1
python run_demo.py
```

Expected output: all 3 feature pyramid scales (C3 80×80, C4 40×40, C5 20×20) pass with `PASSED` integrity checks.

---

## Environment Setup

### Option A: Docker (Recommended for Reviewers)

The Docker image includes all dependencies and launches the dashboard automatically.

```bash
# Build and launch the interactive dashboard
docker compose up --build
```

Then open **http://localhost:8501** in your browser.

> **Note:** GPU support requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The container falls back gracefully to CPU for image demo mode.

### Option B: Local Python Environment (Linux / HPC)

```bash
# 1. Create isolated virtual environment (Python 3.10)
uv venv .venv --python 3.10
source .venv/bin/activate

# 2. Install dependencies with CUDA 12.4 support
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match

# 3. Clone base CFT engine and apply CMAFM patches
bash scripts/setup.sh
```

### Option B: Local Python Environment (Windows)

```powershell
# Requires 'uv' package manager: https://astral.sh/uv
.\scripts\setup.ps1
```

---

## Dataset Preparation

CMAFM uses two datasets. Download them to a `../data/` directory (sibling of the repo root):

| Dataset | Source | Format | Notes |
|:---|:---|:---|:---|
| **M3FD** | [DroneVehicle / M3FD_Detection.zip](https://github.com/dlsrbgg33/M3FD) | VOC XML | 4,200 registered RGB+Thermal pairs, 6 classes |
| **FLIR ADAS v2** | [HuggingFace: jsonhash/FLIR_aligned](https://huggingface.co/datasets/jsonhash/FLIR_aligned) (`flir_align.7z`) | COCO JSON | Aligned subset for spatial correspondence |

```bash
# After downloading, run the unified preparation script
# This converts both datasets to YOLO format and creates train/val splits
python scripts/prepare_all.py

# Verify dataset alignment
python tools/verify_flir.py
python tools/verify_m3fd.py
```

---

## Training

### Minimal 1-Epoch Smoke Test

Verify the full training pipeline is functional with a single epoch:

```bash
# Activate environment
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

# 1. Map pretrained YOLOv5l weights to dual-stream architecture
python tools/load_pretrained_partial.py

# 2. Run 1 epoch on M3FD to verify the pipeline end-to-end
python -m cft_engine.train \
    --data data/M3FD_FLIR.yaml \
    --cfg data/yolov5l_cmafm_M3FD.yaml \
    --weights weights/cmafm_yolo_init.pt \
    --batch-size 8 \
    --epochs 1 \
    --img 640 \
    --project runs/train \
    --name smoke_test \
    --exist-ok
```

> **Expected:** Training loss decreases, no NaN/Inf values. Checkpoint saved to `runs/train/smoke_test/weights/last.pt`.

### Full 30-Epoch Reproduction (Single GPU)

```bash
# Train CMAFM-YOLO on M3FD + FLIR supplementary set, validate on M3FD
python -m cft_engine.train \
    --data data/M3FD_FLIR.yaml \
    --cfg data/yolov5l_cmafm_M3FD.yaml \
    --weights weights/cmafm_yolo_init.pt \
    --batch-size 8 \
    --epochs 30 \
    --img 640 \
    --hyp data/hyp.scratch.yaml \
    --project runs/train \
    --name cmafm_yolo_repro \
    --device 0
```

> **Expected range:** mAP@0.5 ≈ 84–86% after 30 epochs (our 10-seed mean: 85.75% ± 0.28%).

### 10-Seed HPC Reproduction (SLURM)

```bash
# Configure your cluster partition in .env:
# SLURM_PARTITION=gpu

# Launch 10-seed job array (each seed = 1 GPU, ~3h wall time per seed)
bash scripts/submit.sh slurm/submit_cmafm_yolo_m3fd.sh

# After completion, aggregate metrics across all seeds
python tools/aggregate_final_metrics.py
```

---

## Evaluation (Reproducing Paper Results)

### Reproduce mAP@0.5 on M3FD Validation Set

```bash
# Evaluate a trained checkpoint
python src/fusion/evaluate.py \
    --weights weights/best.pt \
    --data data/M3FD_FLIR.yaml \
    --img-size 640 \
    --batch-size 8 \
    --conf-thres 0.001 \
    --iou-thres 0.6
```

**Expected results (from our best seed):**

| Class | mAP@0.5 | mAP@0.5:0.95 |
|:---|:---:|:---:|
| People | 90.4% | 62.5% |
| Car | 93.8% | 66.1% |
| Bus | 88.1% | 60.9% |
| Motorcycle | 78.2% | 47.3% |
| Lamp | 89.1% | 58.9% |
| Truck | 82.1% | 47.3% |
| **Mean (mAP)** | **86.27%** | **57.16%** |

### Run Ablation Studies

```bash
# Full ablation (channel-only vs. spatial-only vs. full CMAFM)
python src/fusion/run_ablation.py --device 0

# Or via SLURM job array
bash slurm/run_ablation.sh
```

### Generate Paper Tables and Figures

```bash
# Quantitative tables (LaTeX format)
python tools/generate_paper_tables.py

# GFLOPs / latency profiling
python tools/get_efficiency.py

# Day vs. night breakdown
python src/fusion/analyze_night.py

# Ablation learning curves
python src/fusion/plot_ablation_curves.py
```

---

## Interactive Dashboard

The Streamlit dashboard demonstrates live dual-stream detection on the 3 bundled sample pairs without requiring model weights (demo mode), or full GPU inference if weights are provided.

```bash
# Standard launch
streamlit run src/fusion/dashboard.py

# Or with Docker (no local setup required)
docker compose up --build
# → Open http://localhost:8501
```

Dashboard features:
- **Tab 1 — Image Detection**: Upload custom or select bundled RGB+Thermal pairs, visualize three-way detection (Fusion vs. RGB-only vs. Thermal-only).
- **Tab 2 — Video Tracking**: Process the bundled FLIR ADAS driving sequences at real-time speed.
- **Tab 3 — Architecture & Benchmarks**: Layer-wise profiling, complexity comparison table, and detector-agnostic generalization results.
- **Tab 4 — Model Evaluation**: Ground-truth TensorRT FP16 accuracy metrics on M3FD and FLIR test sets (from `jetson_deploy/artifacts/`).

---

## Containerization

The [`Dockerfile`](Dockerfile) provides a fully self-contained reproducibility environment:

```bash
# Build image
docker build -t cmafm:wacv2027 .

# Run architecture verification (no GPU required)
docker run --rm cmafm:wacv2027 python run_demo.py

# Launch dashboard (GPU)
docker run --rm --gpus all -p 8501:8501 cmafm:wacv2027

# Launch dashboard (CPU only)
docker run --rm -p 8501:8501 cmafm:wacv2027
```

> **Healthcheck**: The container exposes `/healthz` via Streamlit's built-in health endpoint at `http://localhost:8501/healthz`.

---

## Jetson Orin Nano Edge Deployment

See [`jetson_deploy/README.md`](jetson_deploy/README.md) for complete instructions on:
- Exporting to ONNX and building the TensorRT FP16 engine
- Running the Jetson-native dashboard
- Benchmarking end-to-end invocation latency

Pre-generated TensorRT evaluation results are in [`jetson_deploy/artifacts/`](jetson_deploy/artifacts/).

---

## Scientific Standards & Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full NeurIPS/Pineau Reproducibility Checklist.

| Property | Value |
|:---|:---|
| **Seeds** | 10 random seeds (1–10) |
| **Batch size** | 8 (matches original CFT paper) |
| **Training epochs** | 30 |
| **Optimizer** | SGD, momentum=0.937, weight_decay=5e-4 |
| **Learning rate schedule** | Linear warmup (3 epochs) → Cosine decay |
| **Input resolution** | 640 × 640 |
| **Validation** | M3FD deterministic 20% hold-out (seed=42) |
| **Hardware** | 10× NVIDIA A100 (HPC cluster, 1 GPU/seed) |
| **FP16 Training** | Enabled (automatic mixed precision) |

---

## Citation

```bibtex
@inproceedings{cmafm2027wacv,
  title     = {CMAFM: Cross-Modal Attention Fusion Module for Efficient
               Real-Time RGB-Thermal Object Detection on Edge Systems},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027},
  note      = {Anonymous submission, ID 3068}
}
```

---

## Disclaimer

*The views expressed in this research are those of the authors and do not reflect the official policy or position of any government agency or institution.*

*For double-blind anonymity, all institutional affiliations, author names, and internal cluster identifiers have been removed from this supplementary release.*
