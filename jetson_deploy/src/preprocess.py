"""Paired RGB/thermal loading and YOLO-compatible letterboxing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessMeta:
    original_shape: Tuple[int, int]
    input_shape: Tuple[int, int]
    ratio: Tuple[float, float]
    pad: Tuple[float, float]


def _thermal_to_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Unsupported thermal dtype: {image.dtype}")
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("Thermal image contains no finite values")
    low = float(image[finite].min())
    high = float(image[finite].max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def load_pair(rgb_path: Path, thermal_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    thermal = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
    if thermal is None:
        raise FileNotFoundError(f"Could not read thermal image: {thermal_path}")

    if thermal.ndim == 3:
        if thermal.shape[2] == 4:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGRA2GRAY)
        else:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    thermal = _thermal_to_u8(thermal)

    if rgb_bgr.shape[:2] != thermal.shape[:2]:
        raise ValueError(
            "RGB and thermal frames must already be spatially aligned and have "
            f"the same dimensions; received {rgb_bgr.shape[:2]} and {thermal.shape[:2]}."
        )
    return rgb_bgr, thermal


def _letterbox_geometry(
    shape: Tuple[int, int], target: Tuple[int, int], stride: int
) -> tuple[Tuple[int, int], Tuple[float, float], Tuple[float, float]]:
    height, width = shape
    target_h, target_w = target
    ratio = min(target_h / height, target_w / width)
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    if target_h % stride or target_w % stride:
        raise ValueError(f"Input size {target} must be divisible by model stride {stride}")
    return (resized_w, resized_h), (ratio, ratio), (pad_w / 2, pad_h / 2)


def _resize_and_pad(
    image: np.ndarray,
    resized: Tuple[int, int],
    pad: Tuple[float, float],
    color,
) -> np.ndarray:
    if (image.shape[1], image.shape[0]) != resized:
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = pad
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


def preprocess_pair(
    rgb_bgr: np.ndarray,
    thermal_gray: np.ndarray,
    *,
    image_size: int = 640,
    stride: int = 32,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray, PreprocessMeta]:
    if rgb_bgr.shape[:2] != thermal_gray.shape[:2]:
        raise ValueError("RGB and thermal inputs must have identical dimensions")

    original_shape = rgb_bgr.shape[:2]
    target = (image_size, image_size)
    resized, ratio, pad = _letterbox_geometry(original_shape, target, stride)

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    thermal = cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2RGB)
    rgb = _resize_and_pad(rgb, resized, pad, (114, 114, 114))
    thermal = _resize_and_pad(thermal, resized, pad, (114, 114, 114))

    rgb_tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=dtype)
    thermal_tensor = np.ascontiguousarray(
        thermal.transpose(2, 0, 1)[None], dtype=dtype
    )
    rgb_tensor /= np.asarray(255.0, dtype=dtype)
    thermal_tensor /= np.asarray(255.0, dtype=dtype)

    meta = PreprocessMeta(
        original_shape=original_shape,
        input_shape=target,
        ratio=ratio,
        pad=pad,
    )
    return rgb_tensor, thermal_tensor, meta


def scale_boxes(boxes: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    scaled = boxes.astype(np.float32, copy=True)
    scaled[:, [0, 2]] -= meta.pad[0]
    scaled[:, [1, 3]] -= meta.pad[1]
    scaled[:, [0, 2]] /= meta.ratio[0]
    scaled[:, [1, 3]] /= meta.ratio[1]
    height, width = meta.original_shape
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clip(0, width)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clip(0, height)
    return scaled

