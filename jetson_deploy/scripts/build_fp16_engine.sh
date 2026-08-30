#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ONNX_PATH="${1:-${DEPLOY_DIR}/artifacts/cmafm_yolo_640.onnx}"
ENGINE_PATH="${2:-${DEPLOY_DIR}/artifacts/cmafm_yolo_640_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"

if [[ ! -f "${ONNX_PATH}" ]]; then
    echo "ERROR: ONNX model not found: ${ONNX_PATH}" >&2
    exit 1
fi
if [[ ! -x "${TRTEXEC}" ]]; then
    echo "ERROR: trtexec not found or not executable: ${TRTEXEC}" >&2
    exit 1
fi

mkdir -p "$(dirname "${ENGINE_PATH}")"

"${TRTEXEC}" \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --fp16 \
    --memPoolSize="workspace:${WORKSPACE_MIB}" \
    --builderOptimizationLevel=5 \
    --profilingVerbosity=detailed \
    --timingCacheFile="${ENGINE_PATH}.timing.cache" \
    --skipInference

echo "TensorRT FP16 engine written to ${ENGINE_PATH}"
