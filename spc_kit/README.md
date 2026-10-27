# SPC experiment kit (CMAFM-YOLO, response to WACV #3068 reviews)

Date: 2026-10-20 · Target code: `github.com/mango-jeongi/cmafm`, branch `wacv-2027`

Copy this folder to the repo root as `spc_kit/`, build `cft_engine` with the existing `scripts/setup.sh`, then run the steps below. Existing code and results are not modified; new outputs go only to `runs/spc/` and `results/`.

## 0. Problems found in the code (they affect both the paper and the experiments)

| # | Problem | Where | What the kit does |
|---|---|---|---|
| A | **`--seed` is ignored.** `train.py` calls `init_seeds(2 + rank)`, so SLURM seeds 1–10 all trained with the same seed. The reported "10-seed ±0.28" is only GPU nondeterminism | `cft_engine/train.py` lines 66 and 504 | Changes it to `init_seeds(opt.seed + max(rank, 0))` (single-process rank is -1) |
| B | **Initialization differs per model.** CMAFM started from a hand-written layer map (`CMAFM_Pretrained.pt`), while CFT loaded `yolov5l.pt` by name, so only the first few layers of stream one got pretrained weights | `slurm/submit_*.sh`, `tools/load_pretrained_partial.py` | Every model gets COCO YOLOv5l weights by the same rule (`init_weights.py`); fusion modules and Detect are freshly initialized per seed |
| C | The main CMAFM run used **M3FD-only data, 30 epochs, fine-tune hyp**; CFT used **M3FD+FLIR, 100 epochs, scratch hyp** | `slurm/submit_cmafm_yolo_m3fd.sh`, `submit_cft_m3fd_flir.sh` | One protocol for all (section 2) |
| D | best.pt is **selected on the reported val (= test) set** | `train.py` fitness | 10% of M3FD train is held out as `sel` for selection only; the 840 images are test only |
| E | Faster R-CNN and YOLO **use different 840-image M3FD sets** (numpy vs python random) | `src/fusion/dataset.py`, `scripts/prepare_all.py` | Same stem list for both (`patch_frcnn.py`) |
| F | Faster R-CNN ablations use StepLR step 8, the full model step 10, and report the best of several test-set evaluations | `src/fusion/run_ablation.py` | Step 10 for all, last epoch reported |
| G | 13,682.5 GFLOPs comes from thop at 32×32 × 400. CFT pools to 8×8 tokens, so its cost does not scale with image area | `tools/get_efficiency.py` | Counted directly at 640×640 (`profile_models.py`) |
| H | Two CMAFM implementations exist (`cmafm.py` without alpha, `cmafm_checkpoint.py` with alpha). **Training used the one with alpha** (imported by `yolo_test.py`) | `models/yolo_test.py` line 1 | Paper describes the alpha version; deleting `cmafm.py` is recommended |
| I | Pretrained transfer is **1044/1190 tensors (87.7%)**, not 99.7%; the rest are fusion modules and Detect | `init_weights.py` output | Number corrected in the paper |

## 1. Already done: FLOPs counted directly at 640×640 (CPU, with this kit)

`results/flops_640_cpu.json` · torch FlopCounter, both inputs 640×640, nc=6

| Model | Params (M) | GFLOPs | Fusion-module GFLOPs | Old method (32×32×400) |
|---|---|---|---|---|
| RGB-only / Thermal-only (YOLOv5l) | 46.66 | 114.0 | 0 | 114.6 |
| Dual+Add (no fusion) | 73.73 | 189.7 | 0 | 190.8 |
| Dual+Concat (1×1 projection) | 88.88 | 217.3 | 27.7 | 218.5 |
| SE / CBAM controls | 86.47 | 212.3 | 22.7 | 214 |
| **CFT** | 206.27 | **224.4** | 34.8 | **13,732.6** |
| **CMAFM-YOLO (full)** | 105.45 | **232.6** | 42.9 (14.3 per scale) | 240.4 |

CMAFM has slightly more FLOPs than CFT and half the parameters. The equal 14.3 GFLOPs per scale shows the dominant term is HWC² (dense 1×1 and 3×3 convolutions), consistent with the measured per-scale latencies (1.57 / 1.97 / 1.90 ms).

## 2. Unified protocol

- Data: M3FD train (3,360 images minus 10% held out as `sel`) + FLIR train for training / M3FD `sel` for best.pt selection / the 840 M3FD images as test / the FLIR official test split as in-domain test
- Initialization: COCO YOLOv5l v5.0 (`weights/yolov5l.pt`), same rule for every model
- Training: `data/hyp.scratch.yaml` (lr0 0.005, warmup 3, mosaic 1.0), 100 epochs, batch 8, 640, SGD
- Seeds: 1–5 (10 if possible)
- Evaluation: conf 0.001, NMS IoU 0.6, 640, the same subset files for every model

## 3. Steps

```bash
# repo root, cft_engine already prepared by setup.sh
cp -r /path/to/spc_kit ./spc_kit
python spc_kit/tools/apply_kit.py --engine cft_engine          # seed fix + register CMAFMVar + write 12 yamls
wget -P weights https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt   # only if missing
python spc_kit/tools/init_weights.py --engine cft_engine --src weights/yolov5l.pt --out weights/spc_init
python spc_kit/tools/make_splits.py --merged data/merged_txt --out data/spc_split
python spc_kit/tools/make_subsets.py --split data/spc_split --out data/spc_subsets  # day/night subsets (--scene-csv if you have scene labels)

# training: 6 models x 5 seeds = 30 jobs (required), then 30 ablation jobs
sbatch --array=0-29 spc_kit/slurm/run_matrix.sh spc_kit/slurm/matrix_main.txt 5
sbatch --array=0-29 spc_kit/slurm/run_matrix.sh spc_kit/slurm/matrix_ablation.txt 5

# evaluation and tables (after all training finishes)
sbatch spc_kit/slurm/eval_all.sh       # -> results/tables.md, results/eval.csv, results/flops_640.json (with latency), results/alpha_seed*.json
```

To check the pipeline first, run a single short job such as `EPOCHS=1 sbatch --array=29 spc_kit/slurm/run_matrix.sh spc_kit/slurm/matrix_main.txt 5`.

### Faster R-CNN family (optional)

```bash
python spc_kit/tools/patch_frcnn.py
for s in 1 2 3; do
  CMAFM_TEST_STEMS=data/spc_split/m3fd_test_stems.txt CMAFM_SEL_STEMS=data/spc_split/m3fd_sel_stems.txt \
  python src/fusion/run_ablation.py --seed $s --retrain-full --variants rgb_only thermal_only early_fusion dual_no_attn full
done
```

### Jetson Orin Nano (optional)

Use the newly trained `cmafm_full` seed 1 best.pt with the existing `jetson_deploy` procedure, and:
1. Measure in two power modes (default 15 W and MAXN_SUPER) and record the mode (`sudo nvpmodel -q`).
2. Use **real data** instead of synthetic input: run `src/evaluate_tensorrt.py` on the 840 M3FD test images, recording pre-processing, inference and post-processing time.
3. Log power with `tegrastats --interval 100` (average VDD_IN).
4. Compare PyTorch FP32 mAP and TensorRT FP16 mAP for the same checkpoint (not against a multi-seed mean).
5. If possible, convert CFT and Dual+Add the same way and put them in the same table.
6. Report INT8 only if actually measured.

## 4. Compute budget (L40S, about 7.5 h per 100-epoch run from earlier logs)

| Group | Jobs | GPU hours | Notes |
|---|---|---|---|
| Main (6 models × 5 seeds) | 30 | ~225 h | **Required** |
| Ablation (6 variants × 5 seeds) | 30 | ~225 h | ~135 h with 3 seeds |
| Evaluation, latency, alpha statistics | 1 | a few hours | |
| Faster R-CNN (5 variants × 3 seeds, 30 epochs) | 15 | ~30 h | Optional |

With 8 GPUs in parallel, the main set takes about 1.5 days, about 3 days including ablations.

## 5. What was verified in a container

- All 12 configs build, run forward and are profiled for FLOPs (CPU)
- On fake data: `cmafm_full` trained for 2 epochs → best.pt → subset evaluation → alpha statistics → table aggregation, end to end (CPU; accuracy values meaningless)
- `init_weights.py` produces exactly the same CMAFM layer map as the earlier hand-written `load_pretrained_partial.py`
- Not done: GPU training and real-data runs (no GPU or datasets in the container)

## Hamming operational workflow (2026-10-27)

Use the repository root on Hamming. `scripts/submit.sh` loads the existing `.env`
partition setting (`genai`); the SPC scripts also default to that partition.
`genai` contains the L40S nodes `compute-8-10` and `compute-8-12`.
Python commands in the launchers use `uv` and the existing repository `.venv`.
Checkpoint selection uses 0.1 � AP50 + 0.9 � AP50:95 on the selection split;
the added AP75 output is excluded from that fitness input.
Setup automatically applies the SPC kit when this directory is present and saves
engine changes in `src/engine/cft_engine_patches/` for both setup scripts.

The original `weights/yolov5l.pt` on Hamming was a later SPPF model. Keep it intact;
preparation uses the separate official v5.0 checkpoint `weights/yolov5l_v5.0.pt`.
Preparation and all data/model checks run inside a Slurm allocation:

```bash
mkdir -p logs results
bash scripts/submit.sh spc_kit/slurm/prepare.sh
# After preflight passes, one real-data epoch of CMAFM-full, seed 1:
EPOCHS=1 bash scripts/submit.sh --array=5 spc_kit/slurm/run_matrix.sh spc_kit/slurm/matrix_main.txt 1
# After the smoke test passes, 60 runs, at most eight GPUs in total:
EPOCHS=100 bash scripts/submit.sh --array=0-29%4 spc_kit/slurm/run_matrix.sh spc_kit/slurm/matrix_main.txt 5
EPOCHS=100 bash scripts/submit.sh --array=0-29%4 spc_kit/slurm/run_matrix.sh spc_kit/slurm/matrix_ablation.txt 5
# Attach afterok dependencies on both training array IDs when submitting evaluation:
bash scripts/submit.sh --dependency=afterok:MAIN_ID:ABLATION_ID spc_kit/slurm/eval_all.sh
```

Short runs default to `runs/spc_smoke/job_JOBID/`, separate from production
`runs/spc/`. The launcher refuses to overwrite an existing training directory.
Create `logs/` before calling `sbatch`, since Slurm opens its log before the script
starts. Matrix files may use Windows or Unix line endings.

`results/spc_preflight.json` records all 12 CUDA forwards, transferred tensor
counts, split checks, software/device versions, and input hashes. Evaluation
requires all 60 checkpoints and fails on an unsuccessful test subprocess rather
than silently reporting incomplete results. Per-subset process logs are kept in
`results/eval_logs/`.

The existing data contain 3,024 M3FD + 4,126 FLIR training pairs, 336 M3FD selection
pairs, 840 M3FD test pairs, and 1,013 aligned FLIR test pairs. Verify the FLIR source
provenance before describing that subset as the full official ADAS v2 test set.
The fixed luminance thresholds produce 58/782 night/day M3FD test pairs and
20/993 night/day FLIR test pairs; these are brightness-defined subsets.

Final evaluation now audits all 60 training logs for epochs 0�99 and checks the
100-epoch/seed/batch/image-size protocol before loading checkpoints. After
aggregation, `results/audit_all.json` records complete 360-row model/seed/subset
coverage, identical evaluated samples across models, all five alpha analyses,
and hashes of the final paper artifacts. The audit rejects incomplete results.
