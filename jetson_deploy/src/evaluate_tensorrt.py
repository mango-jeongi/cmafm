"""Evaluate the fixed-batch CMAFM TensorRT engine on a paired M3FD manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .infer_tensorrt import DEFAULT_WARMUP_ITERATIONS, TensorRTSession
from .preprocess import load_pair, preprocess_pair


def parse_args() -> argparse.Namespace:
    deploy_dir = Path(__file__).resolve().parents[1]
    repo_root = deploy_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=Path,
        default=deploy_dir / "artifacts" / "cmafm_yolo_640_fp16.engine",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, default=repo_root / "cft_engine")
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=deploy_dir / "artifacts" / "m3fd_tensorrt_map.json"
    )
    return parser.parse_args()


def scale_predictions(predictions: torch.Tensor, meta) -> torch.Tensor:
    scaled = predictions.clone()
    scaled[:, [0, 2]] -= meta.pad[0]
    scaled[:, [1, 3]] -= meta.pad[1]
    scaled[:, [0, 2]] /= meta.ratio[0]
    scaled[:, [1, 3]] /= meta.ratio[1]
    height, width = meta.original_shape
    scaled[:, [0, 2]].clamp_(0, width)
    scaled[:, [1, 3]].clamp_(0, height)
    return scaled


def evaluate() -> dict:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--confidence and --iou must be between zero and one")
    if args.max_detections != 300:
        raise SystemExit("The installed CFT evaluator fixes --max-detections at 300")

    engine_dir = args.engine_dir.resolve()
    if not (engine_dir / "utils" / "metrics.py").is_file():
        raise SystemExit(f"CFT evaluator not found under {engine_dir}")
    sys.path.insert(0, str(engine_dir))
    from utils.general import box_iou, non_max_suppression
    from utils.metrics import ap_per_class

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    if args.limit:
        records = records[: args.limit]
    class_names = tuple(manifest["class_names"])
    device = torch.device("cuda:0")
    iou_vector = torch.linspace(0.5, 0.95, 10, device=device)
    stats = []
    inference_seconds = 0.0
    nms_seconds = 0.0
    target_count = 0
    detection_count = 0
    started = time.perf_counter()

    with TensorRTSession(args.engine) as session:
        dummy = np.zeros(
            (1, 3, session.image_size, session.image_size), dtype=session.input_dtype
        )
        for _ in range(args.warmup):
            session.infer(dummy, dummy)

        for image_index, record in enumerate(records, start=1):
            rgb_path = manifest_path.parent / record["rgb"]
            thermal_path = manifest_path.parent / record["thermal"]
            rgb_bgr, thermal_gray = load_pair(rgb_path, thermal_path)
            if (rgb_bgr.shape[1], rgb_bgr.shape[0]) != (record["width"], record["height"]):
                raise ValueError(f"Manifest image dimensions do not match {record['id']}")
            rgb, thermal, meta = preprocess_pair(
                rgb_bgr,
                thermal_gray,
                image_size=session.image_size,
                dtype=session.input_dtype,
            )

            inference_start = time.perf_counter()
            outputs = session.infer(rgb, thermal)
            inference_seconds += time.perf_counter() - inference_start
            raw = session.prediction_output(outputs).astype(np.float32, copy=False)

            nms_start = time.perf_counter()
            prediction_tensor = torch.from_numpy(raw).to(device)
            detections = non_max_suppression(
                prediction_tensor,
                args.confidence,
                args.iou,
                multi_label=True,
                agnostic=False,
            )[0]
            nms_seconds += time.perf_counter() - nms_start
            detections_native = scale_predictions(detections, meta)
            detection_count += len(detections_native)

            annotations = record["annotations"]
            target_classes = [int(item["class_id"]) for item in annotations]
            target_count += len(target_classes)
            labels = torch.tensor(
                [[item["class_id"], *item["bbox_xyxy"]] for item in annotations],
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 5)

            correct = torch.zeros(
                detections_native.shape[0], len(iou_vector), dtype=torch.bool, device=device
            )
            if len(labels) and len(detections_native):
                detected_targets: set[int] = set()
                target_class_tensor = labels[:, 0]
                for class_id in torch.unique(target_class_tensor):
                    target_indices = (class_id == target_class_tensor).nonzero(as_tuple=False).view(-1)
                    prediction_indices = (
                        class_id == detections_native[:, 5]
                    ).nonzero(as_tuple=False).view(-1)
                    if prediction_indices.numel():
                        ious, matched = box_iou(
                            detections_native[prediction_indices, :4], labels[target_indices, 1:5]
                        ).max(1)
                        for match_index in (ious > iou_vector[0]).nonzero(as_tuple=False):
                            target_index = int(target_indices[matched[match_index]].item())
                            if target_index not in detected_targets:
                                detected_targets.add(target_index)
                                correct[prediction_indices[match_index]] = ious[match_index] > iou_vector
                                if len(detected_targets) == len(labels):
                                    break

            stats.append((
                correct.cpu().numpy(),
                detections_native[:, 4].detach().cpu().numpy(),
                detections_native[:, 5].detach().cpu().numpy(),
                np.asarray(target_classes, dtype=np.float32),
            ))

            if image_index == 1 or image_index % 25 == 0 or image_index == len(records):
                elapsed = time.perf_counter() - started
                rate = image_index / elapsed
                eta = (len(records) - image_index) / rate if rate else 0.0
                print(
                    f"PROGRESS {image_index}/{len(records)} "
                    f"({100 * image_index / len(records):.1f}%) ETA={eta / 60:.1f}min",
                    flush=True,
                )

    combined = [np.concatenate(items, axis=0) for items in zip(*stats)]
    precision, recall, average_precision, f1, ap_classes = ap_per_class(
        *combined, plot=False, names={i: name for i, name in enumerate(class_names)}
    )
    ap50 = average_precision[:, 0]
    ap50_95 = average_precision.mean(axis=1)
    per_class = {}
    for result_index, class_id in enumerate(ap_classes.astype(int)):
        per_class[class_names[class_id]] = {
            "class_id": int(class_id),
            "ground_truth_boxes": int((combined[3] == class_id).sum()),
            "precision": float(precision[result_index]),
            "recall": float(recall[result_index]),
            "ap50": float(ap50[result_index]),
            "ap50_95": float(ap50_95[result_index]),
        }

    result = {
        "backend": "TensorRT FP16",
        "engine": str(args.engine.resolve()),
        "dataset": manifest["dataset"],
        "manifest": str(manifest_path),
        "split_seed": manifest["seed"],
        "image_size": session.image_size,
        "images": len(records),
        "ground_truth_boxes": target_count,
        "detections_after_nms": detection_count,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.iou,
        "map50": float(ap50.mean()),
        "map50_95": float(ap50_95.mean()),
        "mean_precision": float(precision.mean()),
        "mean_recall": float(recall.mean()),
        "mean_inference_ms": 1000.0 * inference_seconds / len(records),
        "mean_nms_ms": 1000.0 * nms_seconds / len(records),
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"RESULT_JSON={args.output.resolve()}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
