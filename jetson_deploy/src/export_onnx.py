"""Export a trusted legacy CMAFM-YOLO checkpoint to fixed-shape ONNX."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .checkpoint import load_trusted_model, prediction_tensor


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        return prediction_tensor(self.model(rgb, thermal, augment=False))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    deploy_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--engine-dir", type=Path, default=deploy_dir / "vendor" / "cft_engine"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=deploy_dir / "vendor" / "cft_engine" / "models" / "yolov5l_cmafm_M3FD.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=deploy_dir / "artifacts" / "cmafm_yolo_640.onnx"
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fuse", action="store_true", help="Fuse Conv/BN in memory before export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.image_size % 32:
        raise SystemExit("--image-size must be a positive multiple of 32")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA export requested, but torch.cuda.is_available() is false")

    model = load_trusted_model(
        args.weights,
        args.engine_dir,
        device=args.device,
        model_config=args.model_config,
        fuse=args.fuse,
    )
    wrapper = ExportWrapper(model).eval()
    device = torch.device(args.device)
    rgb = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
    thermal = torch.zeros_like(rgb)

    with torch.inference_mode():
        sample_output = wrapper(rgb, thermal)
    expected_columns = 5 + 6
    if (
        sample_output.ndim != 3
        or sample_output.shape[0] != 1
        or sample_output.shape[2] != expected_columns
    ):
        raise RuntimeError(
            f"Expected decoded predictions shaped [1, N, {expected_columns}], received "
            f"{tuple(sample_output.shape)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        input_names=["rgb", "thermal"],
        output_names=["predictions"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.inference_mode():
        torch.onnx.export(wrapper, (rgb, thermal), str(args.output), **export_kwargs)

    import onnx

    onnx_model = onnx.load(str(args.output))
    onnx.checker.check_model(onnx_model)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.weights.resolve()),
        "checkpoint_sha256": sha256(args.weights.resolve()),
        "onnx": str(args.output.resolve()),
        "onnx_sha256": sha256(args.output.resolve()),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "opset": args.opset,
        "inputs": {
            "rgb": [1, 3, args.image_size, args.image_size],
            "thermal": [1, 3, args.image_size, args.image_size],
        },
        "output": {"predictions": list(sample_output.shape)},
        "export_device": args.device,
        "export_graph_dtype": "FP32",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"ONNX export validated: {args.output}")
    print(f"Prediction shape: {tuple(sample_output.shape)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
