# CMAFM-YOLO deployment for Jetson Orin Nano 8 GB

This directory is an isolated deployment overlay. It does not edit the research,
training, dataset, or configuration files in the parent repository. The upstream
DocF engine is cloned and patched under `vendor/`, while checkpoints and generated
ONNX/TensorRT files stay under `artifacts/`.

The supported deployment target is:

- Jetson Orin Nano 8 GB / Orin Nano Super
- aarch64 Linux
- JetPack 6.2.x (6.2.3 recommended)
- TensorRT 10.3
- Fixed batch size 1
- Two `1x3x640x640` inputs named `rgb` and `thermal`
- TensorRT FP16 inference on the Ampere GPU

## What is generated

```text
trusted FP16 best.pt
        |
        | load through isolated patched engine; upcast in memory for export
        v
fixed-shape FP32 ONNX
        |
        | TensorRT --fp16 on the Orin Nano
        v
mixed-precision TensorRT engine optimized for the target device
```

The original `.pt` checkpoint is never rewritten. TensorRT may retain individual
layers in FP32 when that is required or faster; the `--fp16` builder flag enables
FP16 kernels wherever TensorRT considers them suitable.

## 1. Install JetPack components

Flash a JetPack 6.2.x Super-capable image and install the NVIDIA compute stack:

```bash
sudo apt update
sudo apt install nvidia-jetpack
sudo apt install git python3-venv python3-pip python3-opencv python3-libnvinfer
```

Create an environment that can see the TensorRT and OpenCV packages installed by
JetPack:

```bash
cd jetson_deploy
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
python -m pip install -r requirements-export.txt
```

Install NVIDIA's aarch64 PyTorch and torchvision packages that match the installed
JetPack version. Do not install the desktop CUDA packages in `../requirements.txt`.
Use NVIDIA's compatibility instructions:

https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/

`onnxruntime` is used only by `tests/verify_onnx.py`. It can be omitted from a
runtime-only environment after deployment validation is complete.

Check the board and software stack:

```bash
bash scripts/check_jetson.sh
```

For sustained benchmarking, use active cooling and select `MAXN_SUPER` from the
Jetson power menu. Query the supported modes instead of assuming a numeric mode:

```bash
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show
```

## 2. Prepare the isolated model engine

The preparation script clones the upstream engine into `vendor/cft_engine`, copies
the CMAFM module and YAML from the parent repository, and applies the existing
parser fixes only to the vendored copy. It also installs the historical
`models.common.CMAFM_Fusion` and `models.common._CMAFM` classes required by the
supplied `best.pt` checkpoint:

```bash
bash scripts/prepare_engine.sh
```

No file outside `jetson_deploy/` is written by this script.

## 3. Add the trusted checkpoint

Copy the trained checkpoint to:

```text
artifacts/best.pt
```

Legacy YOLO `.pt` checkpoints contain pickled Python objects. The exporter uses
`weights_only=False`, so only use a checkpoint you created or obtained from a
trusted source.

## 4. Export fixed-shape ONNX

Run the export from the `jetson_deploy` directory:

```bash
python -m src.export_onnx \
  --weights artifacts/best.pt \
  --output artifacts/cmafm_yolo_640.onnx
```

The exporter performs a model forward pass, checks that the decoded output is
`[1, N, 11]` for six classes, validates the resulting ONNX model, and writes a
JSON manifest containing hashes and tool versions.

Use `--device cuda` if CPU export encounters an operation unavailable in the
Jetson PyTorch CPU build. The graph remains FP32 either way.

## 5. Verify ONNX parity

Place one aligned pair under `samples/rgb/` and `samples/thermal/`, then run:

```bash
python tests/verify_onnx.py \
  --weights artifacts/best.pt \
  --onnx artifacts/cmafm_yolo_640.onnx \
  --rgb samples/rgb/example.jpg \
  --thermal samples/thermal/example.jpg
```

This compares raw PyTorch and ONNX outputs and reports detection counts.

## 6. Build the TensorRT FP16 engine

Always build the serialized engine on the target Jetson:

```bash
bash scripts/build_fp16_engine.sh
```

Optional positional paths are accepted:

```bash
bash scripts/build_fp16_engine.sh model.onnx model_fp16.engine
```

The build uses a 2 GiB TensorRT workspace, builder optimization level 5, detailed
layer profiling metadata, and a reusable timing cache. Override the workspace if
needed:

```bash
WORKSPACE_MIB=1024 bash scripts/build_fp16_engine.sh
```

Do not copy a serialized TensorRT engine from another GPU or TensorRT release.
Copy the ONNX file and rebuild the engine on the Orin Nano instead.

## 7. Verify TensorRT parity

```bash
python tests/verify_tensorrt.py \
  --weights artifacts/best.pt \
  --engine artifacts/cmafm_yolo_640_fp16.engine \
  --rgb samples/rgb/example.jpg \
  --thermal samples/thermal/example.jpg
```

The test reports post-NMS detection counts, mean best matched IoU, and the maximum
matched confidence difference between PyTorch FP16 and TensorRT FP16.

For publication-quality validation, run the full validation set through both
backends and compare mAP; a single image is only a conversion smoke test.

## 8. Run inference

```bash
python -m src.infer_tensorrt \
  --engine artifacts/cmafm_yolo_640_fp16.engine \
  --rgb samples/rgb/example.jpg \
  --thermal samples/thermal/example.jpg \
  --output artifacts/detection.jpg \
  --json-output artifacts/detection.json \
  --warmup 10 \
  --iterations 50
```

The runner loads TensorRT once, allocates persistent CUDA buffers, applies paired
YOLO letterboxing, executes asynchronously, performs class-aware NumPy NMS, and
rescales detections to the original RGB image.

Benchmark the engine independently of image processing:

```bash
bash scripts/benchmark_engine.sh
```

The first benchmark excludes transfers and the second includes them. Neither
includes camera capture, image decoding, preprocessing, or NMS.

Reproduce the fixed-iteration invocation benchmark used for the paper metrics:

```bash
python -m jetson_deploy.src.benchmark_tensorrt \
  --engine jetson_deploy/artifacts/cmafm_yolo_640_fp16.engine \
  --warmup 5 \
  --iterations 100 \
  --json-output jetson_deploy/artifacts/jetson_trt_benchmark.json \
  --csv-output jetson_deploy/artifacts/jetson_trt_latencies.csv
```

This protocol performs five discarded passes immediately before 100 timed
invocations and saves every measured latency. Its timed region includes both
input H2D copies, TensorRT execution, all output D2H copies, and stream
synchronization. It excludes preprocessing, NMS, visualization, video I/O, and
the runner's optional second host-side output copy. Add `--rgb` and `--thermal`
to use a real aligned image pair; otherwise, the script uses deterministic
synthetic tensors with seed 42.

## Input requirements

- RGB and thermal frames must have identical dimensions and be spatially aligned.
- Paired frames must represent the same capture time.
- RGB is read as BGR by OpenCV and converted to RGB for the model.
- Eight-bit thermal input is used unchanged and repeated to three channels.
- Higher-bit-depth thermal files are normalized per image as a convenience. A live
  radiometric camera should use a fixed, sensor-specific conversion calibrated to
  the representation used during training.
- Both modalities receive identical resize and padding geometry.

The supplied runtime handles paired image files. Live CSI/USB camera capture should
be added as a separate producer that supplies synchronized arrays to
`preprocess_pair`; it should not be mixed into the TensorRT execution class.

## Troubleshooting

### Checkpoint cannot be unpickled

Run `scripts/prepare_engine.sh` first. The checkpoint depends on Python classes in
the patched DocF engine, including `models.cmafm.CMAFM_Fusion`.

### ONNX output is a list of feature maps

Confirm the model is in evaluation mode and the legacy Detect layer is not forced
into training/export mode. The supplied wrapper expects decoded predictions, not
three raw detection heads.

### TensorRT parser rejects an operation

Save the complete `trtexec` build log. CMAFM uses TensorRT-supported primitive
operations, so a rejection is most likely caused by legacy YOLO graph construction
or an ONNX exporter/version mismatch. Do not add a custom plugin until the failing
node and parser message are identified.

### Out of memory while building

Close desktop applications and reduce the workspace:

```bash
WORKSPACE_MIB=1024 bash scripts/build_fp16_engine.sh
```

If inference memory, rather than build memory, is the problem, export a separate
fixed 512x512 model and validate its accuracy before deployment.
