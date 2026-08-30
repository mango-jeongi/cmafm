#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8501}"
ENGINE_PATH="${ENGINE_PATH:-${DEPLOY_DIR}/artifacts/cmafm_yolo_640_fp16.engine}"

cd "${REPO_ROOT}"
export CMAFM_TENSORRT_ENGINE="${ENGINE_PATH}"
export WEIGHTS_CMAFM_YOLO="${ENGINE_PATH}"
exec "${PYTHON_BIN}" -m streamlit run src/fusion/dashboard.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false
