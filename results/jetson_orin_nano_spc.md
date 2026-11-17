# Jetson Orin Nano SPC Benchmark & TensorRT FP16 Parity Report

- **Date**: 2026-11-16
- **Device**: NVIDIA Jetson Orin Nano Developer Kit Super (8 GB)
- **OS / L4T / JetPack**: Ubuntu 24.04.5 LTS, L4T `R39.2.1` (`nvidia-l4t-core 39.2.1-20260806224157`, `r39.2/main` repo corresponding to JetPack `7.2.1`)
- **TensorRT / CUDA / cuDNN**: TensorRT `10.16.2.10` (`10.16.2.10-1+cuda13.2`), CUDA `13.2`, cuDNN `9.20.0.46`
- **Checkpoints**: Oct 29 SPC batch, Seed 1 `best.pt` (`cmafm_full` and `dual_add`)
- **Benchmark Setup**: TensorRT FP16 (`trtexec --fp16`), batch size `1`, `640×640`, all `840` real M3FD test images, `5` real-input warm-up passes, `tegrastats --interval 100` (`VDD_IN` averaged strictly over the 840-image timed loop).
- **Stage Definitions**:
  - **Pre-process**: `preprocess_pair` (BGR→RGB, thermal→3ch, bilinear letterbox `640×640`, CHW FP32 `/255.0`; disk PNG decode excluded).
  - **Inference**: TensorRT `execute_async_v3` + H2D inputs + D2H output + `cudaStreamSynchronize`.
  - **Post-process (incl. NMS)**: H2D logits + PyTorch CUDA `non_max_suppression` (`conf=0.001`, `IoU=0.60`) + box scaling to native resolution + `torch.cuda.synchronize`.
  - **End-to-End**: `Pre-process + Inference + Post-process` per frame (`FPS = 1000 / End-to-End ms`).

## 1. Latency, Throughput, and Board Power (M3FD Test, 840 Real Images)

| Model | Power Mode | Pre-process (ms) | Inference (ms) | Post-process incl. NMS (ms) | End-to-End (ms) | End-to-End FPS | Inference FPS | Avg Board Power `VDD_IN` (W) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **CMAFM (full)** | `15W` (`nvpmodel -m 0`) | 26.70 | 63.40 | 9.78 | **99.88** | **10.01** | 15.77 | **8.22** (933 samples) |
| **Dual + Add** | `15W` (`nvpmodel -m 0`) | 25.58 | 52.62 | 9.38 | **87.58** | **11.42** | 19.00 | **7.93** (810 samples) |
| **CMAFM (full)** | `MAXN_SUPER` (`nvpmodel -m 2`) | 23.56 | 43.41 | 8.23 | **75.20** | **13.30** | 23.04 | **11.85** (696 samples) |

## 2. Accuracy Parity: Before (PyTorch FP32) vs. After (TensorRT FP16) — `CMAFM (full)` Seed 1

Evaluated at `640×640`, batch size `1`, `conf=0.001`, `IoU=0.60` on the deterministic M3FD test split (`840` images, `7,083` instances) and FLIR effective test split (`1,011` images, `8,214` instances; matching `results/eval.csv` and `results/dataset_stats.md`).

| Dataset | Images / Instances | Backend | mAP@0.5 (%) | mAP@[.5:.95] (%) |
|---|---|---|---:|---:|
| **M3FD test** | 840 / 7,083 | PyTorch FP32 (`640×640`, batch 1) | 87.85 | 59.20 |
| **M3FD test** | 840 / 7,083 | TensorRT FP16 (Orin Nano) | **87.78** | **58.99** |
| **FLIR test** | 1,011 / 8,214 | PyTorch FP32 (`640×640`, batch 1) | 88.12 | 51.84 |
| **FLIR test** | 1,011 / 8,214 | TensorRT FP16 (Orin Nano) | **88.09** | **51.80** |

*(For reference, `results/eval.csv` reports `cmafm_full` seed 1 with `cft_engine/test.py` rectangular batching `rect=True, pad=0.5` as `88.0% / 59.5%` on M3FD and `88.2% / 52.2%` on FLIR.)*
