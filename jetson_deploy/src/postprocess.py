"""NumPy YOLOv5 decoding, nonmaximum suppression, and rendering."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


DEFAULT_NAMES = ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")
DEFAULT_COLORS = (
    (56, 189, 248),
    (244, 63, 94),
    (167, 139, 250),
    (250, 204, 21),
    (52, 211, 153),
    (251, 146, 60),
)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.clip(right_bottom - left_top, 0, None), axis=1)
    area_a = np.prod(np.clip(box[2:] - box[:2], 0, None))
    area_b = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0, None), axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        iou = box_iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def non_max_suppression(
    prediction: np.ndarray,
    *,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> np.ndarray:
    """Decode one YOLOv5 prediction matrix into ``xyxy, score, class`` rows."""
    prediction = np.asarray(prediction)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError("This deployment runtime supports batch size one only")
        prediction = prediction[0]
    if prediction.ndim != 2 or prediction.shape[1] < 6:
        raise ValueError(
            "Expected decoded YOLO output shaped [1, N, 5 + classes], received "
            f"{prediction.shape}."
        )

    prediction = prediction.astype(np.float32, copy=False)
    objectness = prediction[:, 4]
    candidates = prediction[objectness > confidence_threshold]
    if not len(candidates):
        return np.empty((0, 6), dtype=np.float32)

    class_scores = candidates[:, 5:] * candidates[:, 4:5]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    selected = scores > confidence_threshold
    if not selected.any():
        return np.empty((0, 6), dtype=np.float32)

    boxes = xywh_to_xyxy(candidates[selected, :4])
    scores = scores[selected]
    class_ids = class_ids[selected]

    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        class_keep = nms(boxes[indices], scores[indices], iou_threshold)
        kept.extend(indices[class_keep].tolist())
    if not kept:
        return np.empty((0, 6), dtype=np.float32)

    kept_array = np.asarray(kept, dtype=np.int64)
    kept_array = kept_array[np.argsort(scores[kept_array])[::-1]][:max_detections]
    return np.column_stack(
        (boxes[kept_array], scores[kept_array], class_ids[kept_array].astype(np.float32))
    )


def draw_detections(
    rgb_bgr: np.ndarray,
    detections: np.ndarray,
    names: Sequence[str] = DEFAULT_NAMES,
) -> np.ndarray:
    image = rgb_bgr.copy()
    for x1, y1, x2, y2, score, class_id_float in detections:
        class_id = int(class_id_float)
        color = DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]
        label_name = names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
        label = f"{label_name} {score:.2f}"
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(image, p1, p2, color, 2, cv2.LINE_AA)
        (width, height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_top = max(p1[1] - height - baseline - 4, 0)
        cv2.rectangle(
            image,
            (p1[0], text_top),
            (p1[0] + width + 4, text_top + height + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (p1[0] + 2, text_top + height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image

