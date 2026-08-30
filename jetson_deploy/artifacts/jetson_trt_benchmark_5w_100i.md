# Jetson TensorRT Invocation Benchmark — 5 Warmups / 100 Timed Runs

Run date: 2026-08-28  
Device: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super  
Power mode: `MAXN_SUPER`  
TensorRT: 10.3.0  
Engine: `cmafm_yolo_640_fp16.engine`

## Protocol

- One fixed TensorRT execution context and CUDA stream
- Deterministic synthetic RGB and thermal tensors (`seed=42`)
- Input shapes: RGB `[1, 3, 640, 640]`; thermal `[1, 3, 640, 640]`
- Engine I/O dtype: FP32 for both inputs and the prediction output
- Internal engine precision: TensorRT FP16 deployment engine
- Five immediately preceding warmup invocations, discarded
- 100 sequential timed invocations
- Every individual latency saved to CSV and JSON

The timer includes two asynchronous host-to-device input copies, TensorRT
`execute_async_v3`, the output device-to-host copy, and CUDA stream
synchronization. It excludes preprocessing, NMS, box scaling, visualization,
video decoding/encoding, and the runner's additional host-side output copy.

## Results

| Metric | Result |
|---|---:|
| Mean latency | **36.95 ms** |
| Population standard deviation | **1.48 ms** |
| Median latency | **36.31 ms** |
| p95 latency | **40.43 ms** |
| p99 latency | **40.57 ms** |
| Minimum latency | **35.66 ms** |
| Maximum latency | **40.57 ms** |
| Reciprocal inference rate | **27.06 FPS** |

This is invocation latency, not measured end-to-end video throughput.

## Exploratory power telemetry

`tegrastats` was sampled every 200 ms while the benchmark process loaded the
engine, warmed up, and ran the timed loop. During the sustained high-GPU-load
portion identified after the run (`GR3D_FREQ >= 80%` and `VDD_IN >= 15 W`), 15
samples had:

| Metric | Exploratory value |
|---|---:|
| Mean total board input power (`VDD_IN`) | 18.63 W |
| Observed total board input-power range | 15.98–19.20 W |
| Mean GPU temperature | 47.0 °C |
| Approximate board energy per invocation | 0.69 J |

These power values are exploratory. The telemetry interval was not synchronized
to each inference, the active interval was selected after the run, and
`jetson_clocks --show` could not be verified without root privileges. Therefore,
they should not yet be presented as publication-quality model power or energy
measurements. `VDD_IN` is total board input power, not GPU-only power.

## Artifacts

- Raw result and all 100 latencies: `jetson_trt_benchmark_5w_100i.json`
- Per-invocation CSV: `jetson_trt_latencies_5w_100i.csv`
- Raw 200 ms telemetry: `jetson_trt_benchmark_tegrastats.log`

## Reproduction command

```bash
python -m jetson_deploy.src.benchmark_tensorrt \
  --engine jetson_deploy/artifacts/cmafm_yolo_640_fp16.engine \
  --warmup 5 \
  --iterations 100 \
  --json-output jetson_deploy/artifacts/jetson_trt_benchmark_5w_100i.json \
  --csv-output jetson_deploy/artifacts/jetson_trt_latencies_5w_100i.csv
```
