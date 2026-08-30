"""Compare one paired sample between the PyTorch checkpoint and ONNX Runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from src.checkpoint import load_trusted_model, prediction_tensor
from src.postprocess import non_max_suppression
from src.preprocess import load_pair, preprocess_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--onnx", type=Path, default=DEPLOY_DIR / "artifacts" / "cmafm_yolo_640.onnx"
    )
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument(
        "--engine-dir", type=Path, default=DEPLOY_DIR / "vendor" / "cft_engine"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEPLOY_DIR / "vendor" / "cft_engine" / "models" / "yolov5l_cmafm_M3FD.yaml",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Install requirements-export.txt and the JetPack-matched PyTorch package first"
        ) from exc

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_shapes = {item.name: item.shape for item in session.get_inputs()}
    expected = {"rgb", "thermal"}
    if set(input_shapes) != expected:
        raise RuntimeError(f"Unexpected ONNX inputs: {input_shapes}")
    image_size = int(input_shapes["rgb"][-1])

    rgb_bgr, thermal_gray = load_pair(args.rgb, args.thermal)
    rgb, thermal, _ = preprocess_pair(
        rgb_bgr, thermal_gray, image_size=image_size, dtype=np.float32
    )

    model = load_trusted_model(
        args.weights,
        args.engine_dir,
        device="cpu",
        model_config=args.model_config,
    )
    with torch.inference_mode():
        pytorch_output = prediction_tensor(
            model(torch.from_numpy(rgb), torch.from_numpy(thermal), augment=False)
        ).cpu().numpy()
    onnx_output = session.run(["predictions"], {"rgb": rgb, "thermal": thermal})[0]

    if pytorch_output.shape != onnx_output.shape:
        raise RuntimeError(
            f"Output shape mismatch: PyTorch {pytorch_output.shape}, ONNX {onnx_output.shape}"
        )
    difference = np.abs(pytorch_output.astype(np.float32) - onnx_output.astype(np.float32))
    pytorch_detections = non_max_suppression(
        pytorch_output, confidence_threshold=args.confidence
    )
    onnx_detections = non_max_suppression(onnx_output, confidence_threshold=args.confidence)

    print(f"Output shape: {pytorch_output.shape}")
    print(f"Maximum absolute raw-output difference: {difference.max():.6g}")
    print(f"Mean absolute raw-output difference: {difference.mean():.6g}")
    print(f"PyTorch detections: {len(pytorch_detections)}")
    print(f"ONNX detections: {len(onnx_detections)}")


if __name__ == "__main__":
    main()
