#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/.." && pwd)"
VENDOR_DIR="${DEPLOY_DIR}/vendor"
ENGINE_DIR="${VENDOR_DIR}/cft_engine"

mkdir -p "${VENDOR_DIR}"

if [[ ! -d "${ENGINE_DIR}/.git" ]]; then
    git clone https://github.com/DocF/multispectral-object-detection.git "${ENGINE_DIR}"
else
    echo "Using existing vendored engine: ${ENGINE_DIR}"
fi

cp -a "${REPO_ROOT}/src/engine/cft_engine_patches/." "${ENGINE_DIR}/"
cp "${REPO_ROOT}/src/engine/engine_fixes/cmafm.py" "${ENGINE_DIR}/models/cmafm.py"
cp "${REPO_ROOT}/data/M3FD_FLIR.yaml" "${ENGINE_DIR}/data/M3FD_FLIR.yaml"
cp "${REPO_ROOT}/data/m3fd_rgbt.yaml" "${ENGINE_DIR}/data/m3fd_rgbt.yaml"
cp "${REPO_ROOT}/data/mini.yaml" "${ENGINE_DIR}/data/mini.yaml"
cp "${REPO_ROOT}/data/yolov5l_cmafm_M3FD.yaml" "${ENGINE_DIR}/models/yolov5l_cmafm_M3FD.yaml"
mkdir -p "${ENGINE_DIR}/models/transformer"
cp "${REPO_ROOT}/data/yolov5l_fusion_transformerx3_M3FD.yaml" \
   "${ENGINE_DIR}/models/transformer/yolov5l_fusion_transformerx3_M3FD.yaml"

# The repository patcher uses a relative cft_engine path. Running it from the
# vendor directory confines every write to jetson_deploy/vendor/cft_engine.
(
    cd "${VENDOR_DIR}"
    python3 "${REPO_ROOT}/src/engine/engine_fixes/patch_parser.py"
)

# The supplied best.pt was trained with the historical implementation serialized
# as models.common.CMAFM_Fusion and models.common._CMAFM. Install those exact
# classes and adjust only the vendored parser to their constructor signature.
python3 "${SCRIPT_DIR}/install_checkpoint_compat.py" \
    --engine-dir "${ENGINE_DIR}" \
    --compat-source "${DEPLOY_DIR}/compat/cmafm_checkpoint.py"

echo "Vendored engine prepared at ${ENGINE_DIR}"
