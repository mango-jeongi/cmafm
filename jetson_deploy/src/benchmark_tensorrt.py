"""Benchmark the deployed CMAFM TensorRT engine on Jetson.

The timed region matches the supplied reference protocol: two asynchronous
host-to-device input copies, TensorRT execution, asynchronous device-to-host
output copies, and stream synchronization. Preprocessing, NMS, visualization,
video I/O, and a second host-side output copy are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .infer_tensorrt import (
    DEFAULT_BENCHMARK_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS,
    TensorRTSession,
)
from .preprocess import load_pair, preprocess_pair


def parse_args() -> argparse.Namespace:
    deploy_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        type=Path,
        default=deploy_dir / "artifacts" / "cmafm_yolo_640_fp16.engine",
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        help="Optional RGB image. Supply --thermal as well to benchmark real inputs.",
    )
    parser.add_argument(
        "--thermal",
        type=Path,
        help="Optional aligned thermal image. Supply --rgb as well.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_BENCHMARK_ITERATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=deploy_dir / "artifacts" / "jetson_trt_benchmark.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=deploy_dir / "artifacts" / "jetson_trt_latencies.csv",
    )
    return parser.parse_args()


def prepare_inputs(
    args: argparse.Namespace, session: TensorRTSession
) -> tuple[np.ndarray, np.ndarray, str]:
    if (args.rgb is None) != (args.thermal is None):
        raise SystemExit("--rgb and --thermal must be supplied together")

    if args.rgb is not None and args.thermal is not None:
        rgb_bgr, thermal_gray = load_pair(args.rgb, args.thermal)
        rgb, thermal, _ = preprocess_pair(
            rgb_bgr,
            thermal_gray,
            image_size=session.image_size,
            dtype=session.input_dtype,
        )
        return rgb, thermal, "preprocessed real RGB/thermal pair"

    rng = np.random.default_rng(args.seed)
    rgb_record = session.inputs["rgb"]
    thermal_record = session.inputs["thermal"]
    rgb = rng.random(rgb_record["shape"], dtype=np.float32).astype(
        rgb_record["dtype"], copy=False
    )
    thermal = rng.random(thermal_record["shape"], dtype=np.float32).astype(
        thermal_record["dtype"], copy=False
    )
    return rgb, thermal, f"seeded synthetic tensors (seed={args.seed})"


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile))


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise SystemExit("--warmup must be nonnegative")
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    if not args.engine.is_file():
        raise SystemExit(f"TensorRT engine not found: {args.engine}")

    with TensorRTSession(args.engine) as session:
        rgb, thermal, input_source = prepare_inputs(args, session)

        print(f"[*] Input: {input_source}")
        print(f"[*] Running {args.warmup} discarded warmup passes...")
        for _ in range(args.warmup):
            session.infer(rgb, thermal, copy_outputs=False)

        print(f"[*] Benchmarking {args.iterations} sequential invocations...")
        latencies_ms: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            session.infer(rgb, thermal, copy_outputs=False)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

        tensor_metadata = {
            "inputs": {
                name: {
                    "shape": list(record["shape"]),
                    "dtype": str(record["dtype"]),
                }
                for name, record in session.inputs.items()
            },
            "outputs": {
                name: {
                    "shape": list(record["shape"]),
                    "dtype": str(record["dtype"]),
                }
                for name, record in session.outputs.items()
            },
            "tensorrt_version": session.trt.__version__,
        }

    values = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": str(args.engine.resolve()),
        "input_source": input_source,
        "warmup_passes": args.warmup,
        "timed_invocations": args.iterations,
        "timing_scope": (
            "two H2D input copies + TensorRT execute_async_v3 + all D2H output "
            "copies + CUDA stream synchronization"
        ),
        "excluded": [
            "image/video decoding",
            "preprocessing",
            "NMS",
            "box scaling",
            "drawing/rendering",
            "video encoding",
            "additional host-side output copy",
        ],
        "mean_ms": mean_ms,
        "population_std_ms": float(np.std(values, ddof=0)),
        "median_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "reciprocal_fps": 1000.0 / mean_ms,
        "latencies_ms": latencies_ms,
        "tensor_metadata": tensor_metadata,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["iteration", "latency_ms"])
        writer.writeheader()
        writer.writerows(
            {"iteration": index, "latency_ms": latency}
            for index, latency in enumerate(latencies_ms, start=1)
        )

    print("=" * 60)
    print(
        f"Latency:    {summary['mean_ms']:.2f} ± "
        f"{summary['population_std_ms']:.2f} ms (population SD)"
    )
    print(f"Median:     {summary['median_ms']:.2f} ms")
    print(f"p95 / p99: {summary['p95_ms']:.2f} ms / {summary['p99_ms']:.2f} ms")
    print(f"Min / Max: {summary['min_ms']:.2f} ms / {summary['max_ms']:.2f} ms")
    print(f"Rate:       {summary['reciprocal_fps']:.2f} FPS (reciprocal)")
    print(f"JSON:       {args.json_output}")
    print(f"CSV:        {args.csv_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
