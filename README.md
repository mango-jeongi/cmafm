# CMAFM: Cross-Modal Attention Fusion Module

This repository contains the evaluation framework, HPC deployment scripts, and architectural patches for the BMVC 2026 submission: **"Efficient RGB-Thermal Cross-Modal Attention Fusion for Object Detection in Low-Illumination Environments"**.

## 📂 Repository Structure

- **`src/`**: Modular source code.
  - **`fusion/`**: Core implementation of the Cross-Modal Attention Fusion Module.
  - **`engine/`**: Stability patches and runtime fixes.
    - **`cft_engine_patches/`**: Drop-in files that overwrite original files in the cloned `cft_engine`.
    - **`engine_fixes/`**: Runtime scripts including the novel `cmafm.py` module.
- **`scripts/`**: Automation for data preparation and environment setup.
- **`tools/`**: Dataset alignment verification and metric aggregation tools.
- **`slurm/`**: Configurations for the HPC cluster 10-seed job array.
- **`cft_engine/`**: *(Gitignored)* The base multispectral engine (YOLOv5 based) cloned during setup.

## 🚀 Cluster Deployment

Follow these steps for a scientifically accurate 30-epoch reproduction:

### 1. Environment & Engine Setup
```bash
# 1. Create venv
uv venv .venv --python 3.10
source .venv/bin/activate

# 2. Install stabilized dependencies (CUDA support)
uv pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match

# 3. Setup Environment and Engine Patches
# This script will automatically clone the base cft_engine, copy .env.example to .env, 
# and safely apply all architectural patches to the multispectral engine.
bash scripts/setup.sh
```

### 2. Dataset Alignment (Critical)
Spatial alignment is required for CMAFM attention gates. We use the **Aligned Research Subset** (MSCOCO format), which ensures pixel-perfect correspondence between RGB and Thermal modalities.

*   **Source:** [Hugging Face: jsonhash/FLIR_aligned](https://huggingface.co/datasets/jsonhash/FLIR_aligned) (`flir_align.7z`)
*   **Methodology:** Manually aligned and time-synced using the protocol from **Zhang et al. (ICIP 2020)**.
*   **Requirement:** Ensure `flir_align.7z` and `M3FD_Detection.zip` are in a parallel `../data/` folder.

```bash
# Extract and unify datasets (Scenario A: Val on M3FD ONLY)
python scripts/prepare_all.py

# Verify alignment visually
python tools/verify_flir.py
```

### 3. Pretrained Weight Mapping
Map standard YOLOv5l weights into the dual-stream architecture:
```bash
python tools/load_pretrained_partial.py
```

### 4. Parallel Training
Launch the 10-seed experiment array using the global environment wrapper:
```bash
# This reads your .env configuration (e.g., SLURM_PARTITION) dynamically
./scripts/submit.sh slurm/submit_yolov5_eval.sh
```

### 5. Packaging for Submission
To build the strictly anonymized, <100MB `Supplementary.zip` package for double-blind review:
```bash
bash scripts/build_supplementary.sh
```

## 📉 Scientific Standards

- **Parity:** We lock `Batch Size = 8` to match the original author's optimization statistics.
- **Stability:** All reported metrics are averaged across 10 random seeds with standard deviation.
- **Benchmark:** Our official target was **85.9% mAP@0.5**.
- **Final Result:** Reached **85.75% ± 0.28%** (Peak **86.27%**).
- **Efficiency:** Measured at **190.3 GFLOPs** and **58 FPS** (RTX 4070 Laptop).

## 📊 Final Results (10-Seed Average)

| Metric | Mean ± StdDev |
| :--- | :--- |
| **mAP @ 0.5** | **85.75% ± 0.28%** |
| **mAP @ [.5:.95]** | **56.71% ± 0.26%** |
| **Recall** | **80.21% ± 1.06%** |

## ⚖️ Disclaimer

*The views expressed in this research are those of the authors and do not reflect the official policy or position of any sponsoring agency.*

---
© 2026 BMVC Submission Team (ID 1669). For institutional privacy, institutional names and internal cluster identifiers are scrubbed from public commits.
