"""Trusted CMAFM-YOLO checkpoint loading for export and parity tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def add_vendored_engine(engine_dir: Path) -> None:
    """Make the isolated DocF engine importable before unpickling a checkpoint."""
    engine_dir = engine_dir.resolve()
    if not (engine_dir / "models").is_dir():
        raise FileNotFoundError(
            f"Vendored engine not found at {engine_dir}. "
            "Run scripts/prepare_engine.sh first."
        )
    engine_str = str(engine_dir)
    if engine_str not in sys.path:
        sys.path.insert(0, engine_str)

    # Fail early with a useful message if the supplied-checkpoint compatibility
    # classes were not installed into models.common.
    common = importlib.import_module("models.common")
    missing = [name for name in ("CMAFM_Fusion", "_CMAFM") if not hasattr(common, name)]
    if missing:
        raise ImportError(
            "Vendored engine lacks checkpoint compatibility classes "
            f"{missing}. Run scripts/prepare_engine.sh again."
        )


def _extract_model(checkpoint: Any, model_config: Path | None, num_classes: int):
    import torch

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint type. Expected a PyTorch module or a YOLO "
            f"checkpoint dictionary, received {type(checkpoint).__name__}."
        )

    candidate = checkpoint.get("ema")
    if candidate is None:
        candidate = checkpoint.get("model")
    if isinstance(candidate, torch.nn.Module):
        return candidate

    state_dict = None
    if isinstance(candidate, dict):
        state_dict = candidate
    elif isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]

    if state_dict is None:
        raise ValueError(
            "The checkpoint does not contain a loadable 'ema', 'model', or "
            "'state_dict' entry."
        )
    if model_config is None:
        raise ValueError(
            "A state-dict-only checkpoint requires --model-config so the model "
            "can be reconstructed."
        )

    from models.yolo_test import Model

    model = Model(str(model_config), ch=3, nc=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "State dictionary does not match the CMAFM model. "
            f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}."
        )
    return model


def load_trusted_model(
    weights: Path,
    engine_dir: Path,
    *,
    device: str = "cpu",
    model_config: Path | None = None,
    num_classes: int = 6,
    fuse: bool = False,
):
    """Load a trusted YOLO checkpoint without modifying the vendored engine.

    ``weights_only=False`` is required for legacy YOLO checkpoints that pickle a
    complete model. It must only be used with a checkpoint from a trusted source.
    The model is deliberately converted to FP32 in memory for reliable ONNX export.
    The checkpoint file itself is never changed.
    """
    import torch

    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    add_vendored_engine(engine_dir)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = _extract_model(checkpoint, model_config, num_classes)

    if hasattr(model, "module"):
        model = model.module
    model = model.float().eval()
    if fuse and hasattr(model, "fuse"):
        model = model.fuse().eval()
    model = model.to(torch.device(device))
    return model


def prediction_tensor(output):
    """Extract the decoded YOLO prediction tensor from legacy model output."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        return prediction_tensor(output[0])
    if isinstance(output, list) and len(output) == 1:
        return prediction_tensor(output[0])
    raise TypeError(
        "Expected one decoded YOLO prediction tensor, but the model returned "
        f"{type(output).__name__}. Ensure it is in eval mode and its Detect layer "
        "is not configured for raw training outputs."
    )
