#!/usr/bin/env bash
set -euo pipefail

echo "CMAFM Jetson deployment environment check"

ARCH="$(uname -m)"
echo "Architecture: ${ARCH}"
if [[ "${ARCH}" != "aarch64" ]]; then
    echo "ERROR: expected aarch64. Run this script on the Jetson Orin Nano." >&2
    exit 1
fi

if [[ -r /etc/nv_tegra_release ]]; then
    echo "Jetson Linux: $(head -n 1 /etc/nv_tegra_release)"
else
    echo "ERROR: /etc/nv_tegra_release was not found; JetPack may not be installed." >&2
    exit 1
fi

python3 --version

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1
else
    echo "WARNING: nvcc is not on PATH. Runtime CUDA may still be installed."
fi

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ -x "${TRTEXEC}" ]]; then
    "${TRTEXEC}" --version
else
    echo "ERROR: trtexec not found at ${TRTEXEC}. Install the JetPack TensorRT components." >&2
    exit 1
fi

python3 - <<'PY'
modules = ("tensorrt", "cv2", "numpy")
for name in modules:
    try:
        module = __import__(name)
        print(f"{name}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"ERROR: cannot import {name}: {exc}")
        raise SystemExit(1)

try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch GPU: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"WARNING: PyTorch is unavailable ({exc}). It is needed for export, not TensorRT runtime.")
PY

if command -v nvpmodel >/dev/null 2>&1; then
    nvpmodel -q || true
fi

echo "Environment check complete."

