"""Compare one paired sample between PyTorch FP16 and TensorRT FP16."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from src.checkpoint import load_trusted_model, prediction_tensor
from src.infer_tensorrt import TensorRTSession
from src.postprocess import box_iou_one_to_many, non_max_suppression
from src.preprocess import load_pair, preprocess_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--engine",
        type=Path,
        default=DEPLOY_DIR / "artifacts" / "cmafm_yolo_640_fp16.engine",
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


def matched_detection_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    best_ious = []
    confidence_deltas = []
    for row in reference:
        same_class = candidate[candidate[:, 5] == row[5]]
        if not len(same_class):
            best_ious.append(0.0)
            continue
        ious = box_iou_one_to_many(row[:4], same_class[:, :4])
        best_index = int(ious.argmax())
        best_ious.append(float(ious[best_index]))
        confidence_deltas.append(abs(float(row[4] - same_class[best_index, 4])))
    mean_iou = float(np.mean(best_ious)) if best_ious else 1.0
    max_confidence_delta = max(confidence_deltas, default=0.0)
    return mean_iou, max_confidence_delta


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the JetPack-matched NVIDIA PyTorch package first") from exc
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is required for FP16 parity validation")

    rgb_bgr, thermal_gray = load_pair(args.rgb, args.thermal)
    model = load_trusted_model(
        args.weights,
        args.engine_dir,
        device="cuda",
        model_config=args.model_config,
    ).half().eval()


    with TensorRTSession(args.engine) as session:
        rgb_trt, thermal_trt, _ = preprocess_pair(
            rgb_bgr,
            thermal_gray,
            image_size=session.image_size,
            dtype=session.input_dtype,
        )
        rgb_pt = torch.from_numpy(rgb_trt.astype(np.float16, copy=False)).cuda()
        thermal_pt = torch.from_numpy(thermal_trt.astype(np.float16, copy=False)).cuda()
        with torch.inference_mode():
            pytorch_output = prediction_tensor(
                model(rgb_pt, thermal_pt, augment=False)
            ).float().cpu().numpy()
        trt_outputs = session.infer(rgb_trt, thermal_trt)
        tensorrt_output = session.prediction_output(trt_outputs).astype(np.float32)

    if pytorch_output.shape != tensorrt_output.shape:
        raise RuntimeError(
            f"Output shape mismatch: PyTorch {pytorch_output.shape}, "
            f"TensorRT {tensorrt_output.shape}"
        )
    pytorch_detections = non_max_suppression(
        pytorch_output, confidence_threshold=args.confidence
    )
    tensorrt_detections = non_max_suppression(
        tensorrt_output, confidence_threshold=args.confidence
    )
    mean_iou, max_conf_delta = matched_detection_metrics(
        pytorch_detections, tensorrt_detections
    )

    print(f"Output shape: {pytorch_output.shape}")
    print(f"PyTorch FP16 detections: {len(pytorch_detections)}")
    print(f"TensorRT FP16 detections: {len(tensorrt_detections)}")
    print(f"Mean best matched IoU: {mean_iou:.6f}")
    print(f"Maximum matched confidence difference: {max_conf_delta:.6f}")


if __name__ == "__main__":
    main()
