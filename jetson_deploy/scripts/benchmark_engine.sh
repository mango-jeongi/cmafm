#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENGINE_PATH="${1:-${DEPLOY_DIR}/artifacts/cmafm_yolo_640_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [[ ! -f "${ENGINE_PATH}" ]]; then
    echo "ERROR: TensorRT engine not found: ${ENGINE_PATH}" >&2
    exit 1
fi

echo "Compute-only benchmark (excludes host/device transfers)"
"${TRTEXEC}" \
    --loadEngine="${ENGINE_PATH}" \
    --warmUp=1000 \
    --duration=30 \
    --useCudaGraph \
    --noDataTransfers

echo "Benchmark including host/device transfers"
"${TRTEXEC}" \
    --loadEngine="${ENGINE_PATH}" \
    --warmUp=1000 \
    --duration=30 \
    --useCudaGraph

