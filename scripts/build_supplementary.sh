#!/bin/bash
# scripts/build_supplementary.sh
# Packages the CMAFM repository for WACV 2027 double-blind supplementary submission.
# Produces an anonymized, self-contained zip under 200MB.
#
# Usage:
#   bash scripts/build_supplementary.sh [--with-videos] [--with-figures]
#
# Options:
#   --with-videos    Include benchmark driving sequence videos (~99MB)
#   --with-figures   Include high-resolution publication figures into docs/figures/

set -euo pipefail

WITH_VIDEOS=false
WITH_FIGURES=false

for arg in "$@"; do
    case $arg in
        --with-videos)
            WITH_VIDEOS=true
            ;;
        --with-figures)
            WITH_FIGURES=true
            ;;
    esac
done

cd "$(dirname "$0")/.."   # Move to repo root

ZIP_NAME="CMAFM_Supplementary_WACV2027.zip"
ZIP_PATH="$(pwd)/../${ZIP_NAME}"
echo "=== CMAFM WACV 2027 Supplementary Packager ==="
echo "Target: ${ZIP_PATH}"
echo "Include videos: ${WITH_VIDEOS}"
echo "Include high-res figures: ${WITH_FIGURES}"

# Remove old zip if exists
rm -f "${ZIP_PATH}"

# Create a temporary build directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

cp -r . "${TEMP_DIR}/code"

cd "${TEMP_DIR}"

# ── Remove git internals ────────────────────────────────────────────────────
find . -name ".git" -exec rm -rf {} + 2>/dev/null || true
find . -name ".gitmodules" -exec rm -f {} + 2>/dev/null || true

# ── Remove caches and compiled artifacts ───────────────────────────────────
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -exec rm -f {} + 2>/dev/null || true
find . -name "*.pyo" -exec rm -f {} + 2>/dev/null || true
find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# ── Remove local environment files ─────────────────────────────────────────
find . -name ".venv" -exec rm -rf {} + 2>/dev/null || true
find . -name ".env" -exec rm -f {} + 2>/dev/null || true

# ── Remove model weights (download separately per README) ──────────────────
find . -name "*.pt" -exec rm -f {} + 2>/dev/null || true
find . -name "*.pth" -exec rm -f {} + 2>/dev/null || true
find . -name "*.ckpt" -exec rm -f {} + 2>/dev/null || true
find . -name "*.engine" -exec rm -f {} + 2>/dev/null || true
find . -name "*.onnx" -exec rm -f {} + 2>/dev/null || true

# ── Clean cft_engine internal git metadata ──────────────────────────────────
rm -rf code/cft_engine/.git 2>/dev/null || true

# ── Remove HPC logs and training run artifacts ─────────────────────────────
rm -rf code/logs

# ── Optionally exclude large driving sequence videos ───────────────────────
if [[ "${WITH_VIDEOS}" == "false" ]]; then
    rm -f code/runs/flir_v1_rgb.mp4
    rm -f code/runs/flir_v1_thermal.mp4
    rm -f code/video_demo.mp4
    echo "Note: Video files excluded (use --with-videos to include them)."
fi

# ── Optionally bundle high-resolution paper figures ─────────────────────────
if [[ "${WITH_FIGURES}" == "true" ]]; then
    FIGURES_SRC="$(pwd)/../../wacv-2027-author-kit-template/images"
    if [ -d "${FIGURES_SRC}" ]; then
        echo "Bundling high-resolution figures into docs/figures/..."
        mkdir -p code/docs/figures
        cp -r "${FIGURES_SRC}"/* code/docs/figures/
    fi
fi

# ── Remove any residual desktop.ini files from Google Drive ────────────────
find . -name "desktop.ini" -exec rm -f {} + 2>/dev/null || true

# ── Zip the sanitized code ─────────────────────────────────────────────────
zip -rq "${ZIP_NAME}" code/
mv "${ZIP_NAME}" "${ZIP_PATH}"

cd - > /dev/null

# ── Report ─────────────────────────────────────────────────────────────────
echo ""
echo "Done! Final size:"
du -sh "${ZIP_PATH}"
echo ""
echo "Verify anonymity before submission:"
echo "  unzip -l '${ZIP_PATH}' | grep -iE '(nps|naval|postgraduate|mingu|owenk)'"
