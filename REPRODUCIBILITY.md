# ML Reproducibility Checklist

> This document addresses the [NeurIPS/Pineau Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf) for the WACV 2027 submission:  
> *"CMAFM: Cross-Modal Attention Fusion Module for Efficient Real-Time RGB-Thermal Object Detection on Edge Systems"*

---

## A. Computational Resources

**A1. For each model and experiment, do you report the number of parameters, the total computational budget (e.g. GPU hours), and computing infrastructure used?**

✅ **Yes.**

- **CMAFM-YOLO**: 105.74 M parameters, 190.3 GFLOPs (17.2 ms / 58 FPS on RTX 4070 Laptop GPU).
- **CFT Baseline**: 206.56 M parameters, 13,682.5 GFLOPs (23.0 ms / 43.5 FPS).
- **10-Seed Training**: Each seed runs 30 epochs on 1 × NVIDIA A100 GPU, ≈3 hours each. Total: ≈30 GPU-hours for the primary CMAFM-YOLO experiment, ≈60 GPU-hours for all ablations and baselines.
- **Hardware**: HPC cluster with NVIDIA A100 (80 GB) GPUs, 8 CPU cores per job, 32 GB RAM.
- **Edge Deployment**: Jetson Orin Nano 8 GB, TensorRT 10, FP16.

GFLOPs profiling can be reproduced via:
```bash
python tools/get_efficiency.py
```

---

## B. Code and Model Release

**B1. Do you release code, and if so, does it include all scripts sufficient to reproduce the main results?**

✅ **Yes.** This supplementary package contains:
- Full CMAFM module implementation (`src/fusion/model.py`, `src/engine/engine_fixes/cmafm.py`)
- Training scripts (`src/fusion/train.py`, `src/fusion/train_yolo.py`)
- Evaluation scripts (`src/fusion/evaluate.py`)
- Dataset preparation scripts (`scripts/prepare_all.py`, `scripts/prepare_flir.py`)
- HPC SLURM job scripts (`slurm/`)
- Docker container for dependency isolation (`Dockerfile`, `docker-compose.yml`)

**B2. Do you include a file specifying all dependencies?**

✅ **Yes.** See [`requirements.txt`](requirements.txt). The Docker container (`Dockerfile`) pins all dependencies including PyTorch 2.6.0+cu124.

**B3. Do you include model weights or a download script for them?**

⚠️ **Partial.** Model weights are not included in the supplementary package to stay within the 200 MB size limit. The initialization weights (YOLOv5l mapped to dual-stream) are reproduced via:
```bash
python tools/load_pretrained_partial.py
```
Trained CMAFM-YOLO weights will be released on a public repository upon paper acceptance (de-anonymized).

---

## C. Experimental Setup and Code

**C1. Did you run ablation studies to assess the impact of different components?**

✅ **Yes.**

- **Channel-only**: CMAFM with channel cross-attention gate disabled.
- **Spatial-only**: CMAFM with spatial cross-gating disabled.
- **Full CMAFM**: Both gates active.

Ablation runner: `src/fusion/run_ablation.py`. Results in Table 2 of the paper.

**C2. Did you use the same experimental setup for all compared methods?**

✅ **Yes.**
- Same base engine (CFT / YOLOv5-based dual-stream architecture).
- Identical hyperparameters: batch size 8, 30 epochs, SGD optimizer, cosine LR schedule.
- Same train/val split (M3FD deterministic 20% hold-out, seed=42).

**C3. Do you describe the method for creating train/test splits?**

✅ **Yes.**
- **M3FD**: Deterministic random split with `random.seed(42)` applied to 4,200 registered image pairs. 80% train (3,360 pairs), 20% validation (840 pairs). Same split reused across all 10 seeds.
- **FLIR ADAS v2**: Pre-defined official test split (1,013 images). FLIR is used as supplementary training data only; official test split is used for evaluation.
- **No data leakage**: Validation images were never used to select hyperparameters; the validation split is held-out from the beginning of the experimental protocol.

**C4. Did you run experiments multiple times?**

✅ **Yes.** 10 independent random seeds for all primary results. Mean and population standard deviation reported:
- mAP@0.5: **85.75% ± 0.28%**
- mAP@0.5:0.95: **56.71% ± 0.26%**

**C5. Do you report results on a test set (not validation set)?**

✅ **Yes (for FLIR).** FLIR ADAS v2 TensorRT evaluation uses the official test split. M3FD does not have a published official test split; we use the deterministic 20% hold-out as described above.

---

## D. Datasets

**D1. Do you use a publicly available dataset? If so, do you provide a link?**

✅ **Yes.**

| Dataset | Reference | Link |
|:---|:---|:---|
| M3FD | [Liu et al., CVPR 2022] | https://github.com/dlsrbgg33/M3FD |
| FLIR ADAS v2 (aligned) | [Zhang et al., ICIP 2020] | https://huggingface.co/datasets/jsonhash/FLIR_aligned |

**D2. Do you describe all the preprocessing steps needed to use the data?**

✅ **Yes.** See [Dataset Preparation](README.md#dataset-preparation) in the README. The script `scripts/prepare_all.py` performs:
1. Extraction of compressed archives (`.7z` for FLIR, `.zip` for M3FD).
2. VOC XML → YOLO label format conversion for M3FD.
3. COCO JSON → YOLO format conversion for FLIR.
4. Frame-level ID matching to ensure RGB and thermal image pairs are correctly registered.
5. Creation of unified train/val text manifests (`train_rgb_unified.txt`, `val_rgb_unified.txt`, etc.).

**D3. If data is not available, do you describe how the data was collected?**

N/A — Both datasets are publicly available. See links above.

---

## E. Analysis

**E1. Does the paper include a full description of the analysis method(s)?**

✅ **Yes.** The paper contains:
- Full module architecture diagram and mathematical formulation of channel cross-attention and spatial cross-gating.
- Complexity analysis (FLOPs breakdown per scale).
- Comparison with 5 alternative detector families to demonstrate plug-and-play generalization.
- Quantitative ablation table isolating each component's contribution.

**E2. Do you describe how you selected the reported quantitative results?**

✅ **Yes.** We report the mean and standard deviation across **all** 10 seeds. No seed selection or cherry-picking. The best single-seed result (86.27%) is separately labeled as "peak" in the paper.

**E3. Did you report error bars?**

✅ **Yes.** Standard deviation across 10 seeds reported for all primary metrics (Table 1, Table 3 of the paper).

**E4. For each experiment, do you provide the exact number of evaluation runs?**

✅ **Yes.** 10 seeds for all primary results, 3 seeds for ablation variants (matching prior work on computational budget).

---

## F. Model Details

**F1. Did you describe the architecture in enough detail that someone could re-implement it?**

✅ **Yes.** The CMAFM module is fully specified in the paper (Figure 2, Section 3.2) and implemented in:
- `src/engine/engine_fixes/cmafm.py` — PyTorch module
- `src/fusion/model.py` — Full Faster R-CNN CMAFM integration
- `src/fusion/model_yolo.py` — CMAFM-YOLO adapter

**F2. Did you describe the optimizer and training procedure?**

✅ **Yes.**

| Hyperparameter | Value |
|:---|:---|
| Optimizer | SGD |
| Momentum | 0.937 |
| Weight decay | 5 × 10⁻⁴ |
| Initial LR | 0.01 |
| LR schedule | Linear warmup (3 epochs) → Cosine annealing |
| Batch size | 8 (locked to match CFT baseline paper) |
| Input resolution | 640 × 640 |
| Epochs | 30 |
| FP16 | Enabled (PyTorch AMP) |
| Data augmentation | Mosaic, HSV jitter, horizontal flip, scale/translate (standard YOLOv5 augmentation) |

**F3. Did you describe the hyperparameter search procedure?**

✅ **Yes.** No automated hyperparameter search was performed. Hyperparameters were inherited directly from the CFT baseline paper to ensure a fair comparison on the same optimization trajectory. See `data/hyp.scratch.yaml` for the complete hyperparameter file.

**F4. Did you describe the final architecture/hyperparameter configuration in enough detail that others could replicate your training?**

✅ **Yes.** See `data/yolov5l_cmafm_M3FD.yaml` (model architecture) and `data/hyp.scratch.yaml` (hyperparameters). Both files are fully annotated and included in this supplementary package.

---

## G. Runtime Environment

**G1. Did you specify which version of the computing infrastructure was used?**

✅ **Yes.**

| Component | Version |
|:---|:---|
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| cuDNN | 8 |
| TensorRT (Jetson) | 10.x |
| Base engine | YOLOv5 (DocF/multispectral-object-detection, pinned via `git clone`) |

**G2. Do you provide a containerized environment?**

✅ **Yes.** See [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml). The container is based on `pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime` and includes all dependencies.

```bash
docker compose up --build
```

---

## H. Significance and Broader Impact

**H1. Are results statistically significant?**

✅ **Yes.** The improvement over the CFT baseline (85.75% vs. 80.48% mAP@0.5, +5.27 pp) is reported with standard deviations across 10 seeds. The improvement exceeds 10× the standard deviation of either method, indicating a statistically robust finding.

**H2. Does the paper include a broader impact statement?**

✅ **Yes.** The paper discusses the potential applications (search and rescue, autonomous driving, infrastructure monitoring) and notes limitations (labeled multispectral data requirements, domain-specific calibration needs).

---

*Checklist version: NeurIPS 2022 / Pineau et al. 2021*  
*Last updated: WACV 2027 Round 2 submission*
