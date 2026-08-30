"""Run paired RGB/thermal CMAFM-YOLO inference with TensorRT 10 on Jetson."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .postprocess import DEFAULT_NAMES, draw_detections, non_max_suppression
from .preprocess import load_pair, preprocess_pair, scale_boxes


DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_BENCHMARK_ITERATIONS = 100


def _load_cudart():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError as exc:
            raise RuntimeError(
                "cuda-python is required. Install requirements-runtime.txt in "
                "the Jetson deployment environment."
            ) from exc
    return cudart


class TensorRTSession:
    """Fixed-shape TensorRT 10 session with persistent device buffers."""

    def __init__(self, engine_path: Path, logger_severity: int | None = None):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT Python bindings are unavailable. Install them through JetPack."
            ) from exc

        self.trt = trt
        self.cudart = _load_cudart()
        severity = trt.Logger.WARNING if logger_severity is None else logger_severity
        self.logger = trt.Logger(severity)
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        self.stream = self._cuda(self.cudart.cudaStreamCreate())
        self.inputs: dict[str, dict[str, Any]] = {}
        self.outputs: dict[str, dict[str, Any]] = {}
        self.device_allocations: list[Any] = []
        self._allocate_static_tensors()

    def _cuda(self, result):
        error, *values = result if isinstance(result, tuple) else (result,)
        if error != self.cudart.cudaError_t.cudaSuccess:
            try:
                _, message = self.cudart.cudaGetErrorString(error)
                if isinstance(message, bytes):
                    message = message.decode("utf-8", errors="replace")
            except Exception:
                message = str(error)
            raise RuntimeError(f"CUDA runtime error: {message}")
        if not values:
            return None
        return values[0] if len(values) == 1 else tuple(values)

    def _allocate_static_tensors(self) -> None:
        trt = self.trt
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            if any(value < 0 for value in shape):
                raise RuntimeError(
                    f"Tensor {name!r} has dynamic shape {shape}. Re-export the model "
                    "with fixed batch-one dimensions."
                )
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            device_ptr = self._cuda(self.cudart.cudaMalloc(nbytes))
            self.device_allocations.append(device_ptr)
            if not self.context.set_tensor_address(name, int(device_ptr)):
                raise RuntimeError(f"TensorRT rejected the device address for tensor {name!r}")
            record = {
                "shape": shape,
                "dtype": dtype,
                "nbytes": nbytes,
                "device": device_ptr,
            }
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = record
            else:
                record["host"] = np.empty(shape, dtype=dtype)
                self.outputs[name] = record

        if set(self.inputs) != {"rgb", "thermal"}:
            raise RuntimeError(
                "Expected TensorRT input names {'rgb', 'thermal'}, received "
                f"{sorted(self.inputs)}. Re-export with the supplied exporter."
            )
        if "predictions" not in self.outputs and len(self.outputs) != 1:
            raise RuntimeError(
                "Expected one output named 'predictions', received "
                f"{sorted(self.outputs)}."
            )

    @property
    def input_dtype(self) -> np.dtype:
        rgb_dtype = self.inputs["rgb"]["dtype"]
        thermal_dtype = self.inputs["thermal"]["dtype"]
        if rgb_dtype != thermal_dtype:
            raise RuntimeError("RGB and thermal TensorRT input dtypes differ")
        return rgb_dtype

    @property
    def image_size(self) -> int:
        rgb_shape = self.inputs["rgb"]["shape"]
        if len(rgb_shape) != 4 or rgb_shape[0] != 1 or rgb_shape[1] != 3:
            raise RuntimeError(f"Unexpected RGB input shape: {rgb_shape}")
        if rgb_shape[2] != rgb_shape[3]:
            raise RuntimeError("Only square fixed-shape inputs are supported")
        return rgb_shape[2]

    def infer(
        self,
        rgb: np.ndarray,
        thermal: np.ndarray,
        *,
        copy_outputs: bool = True,
    ) -> dict[str, np.ndarray]:
        arrays = {"rgb": rgb, "thermal": thermal}
        # Keep converted host arrays alive until the asynchronous copies finish.
        input_hosts: dict[str, np.ndarray] = {}
        for name, array in arrays.items():
            record = self.inputs[name]
            contiguous = np.ascontiguousarray(array, dtype=record["dtype"])
            if contiguous.shape != record["shape"]:
                raise ValueError(
                    f"Input {name!r} has shape {contiguous.shape}; expected {record['shape']}"
                )
            self._cuda(
                self.cudart.cudaMemcpyAsync(
                    int(record["device"]),
                    contiguous.ctypes.data,
                    record["nbytes"],
                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            )
            input_hosts[name] = contiguous

        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

        for record in self.outputs.values():
            host = record["host"]
            self._cuda(
                self.cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    int(record["device"]),
                    record["nbytes"],
                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
            )
        self._cuda(self.cudart.cudaStreamSynchronize(self.stream))
        return {
            name: record["host"].copy() if copy_outputs else record["host"]
            for name, record in self.outputs.items()
        }

    def prediction_output(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        if "predictions" in outputs:
            return outputs["predictions"]
        return next(iter(outputs.values()))

    def close(self) -> None:
        for allocation in self.device_allocations:
            try:
                self._cuda(self.cudart.cudaFree(allocation))
            except Exception:
                pass
        self.device_allocations.clear()
        if getattr(self, "stream", None) is not None:
            try:
                self._cuda(self.cudart.cudaStreamDestroy(self.stream))
            except Exception:
                pass
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    deploy_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        type=Path,
        default=deploy_dir / "artifacts" / "cmafm_yolo_640_fp16.engine",
    )
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=deploy_dir / "artifacts" / "detection.jpg"
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_BENCHMARK_ITERATIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        raise SystemExit("--confidence and --iou must be between zero and one")
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be nonnegative and --iterations must be positive")

    rgb_bgr, thermal_gray = load_pair(args.rgb, args.thermal)
    with TensorRTSession(args.engine) as session:
        start_preprocess = time.perf_counter()
        rgb, thermal, meta = preprocess_pair(
            rgb_bgr,
            thermal_gray,
            image_size=session.image_size,
            dtype=session.input_dtype,
        )
        preprocess_ms = (time.perf_counter() - start_preprocess) * 1000

        for _ in range(args.warmup):
            session.infer(rgb, thermal)

        times_ms = []
        outputs = None
        for _ in range(args.iterations):
            start = time.perf_counter()
            outputs = session.infer(rgb, thermal)
            times_ms.append((time.perf_counter() - start) * 1000)
        assert outputs is not None
        predictions = session.prediction_output(outputs)

    start_postprocess = time.perf_counter()
    detections = non_max_suppression(
        predictions,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
    )
    detections[:, :4] = scale_boxes(detections[:, :4], meta)
    annotated = draw_detections(rgb_bgr, detections, DEFAULT_NAMES)
    postprocess_ms = (time.perf_counter() - start_postprocess) * 1000

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"Failed to write output image: {args.output}")

    records = [
        {
            "box_xyxy": [round(float(value), 2) for value in row[:4]],
            "confidence": round(float(row[4]), 6),
            "class_id": int(row[5]),
            "class_name": (
                DEFAULT_NAMES[int(row[5])]
                if 0 <= int(row[5]) < len(DEFAULT_NAMES)
                else f"class_{int(row[5])}"
            ),
        }
        for row in detections
    ]
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    mean_inference_ms = float(np.mean(times_ms))
    print(f"Detections: {len(detections)}")
    print(f"Preprocess: {preprocess_ms:.2f} ms")
    print(f"TensorRT inference: {mean_inference_ms:.2f} ms ({1000 / mean_inference_ms:.2f} FPS)")
    print(f"Postprocess: {postprocess_ms:.2f} ms")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
