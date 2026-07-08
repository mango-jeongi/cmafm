#!/bin/bash
# scripts/setup.sh

echo "Setting up CMAFM..."

# 1. Setup .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example. Please adjust values if necessary."
fi

# 2. Clone base engine if missing
if [ ! -d "cft_engine" ]; then
    echo "Cloning base multispectral engine..."
    git clone https://github.com/DocF/multispectral-object-detection.git cft_engine
fi

# 3. Apply baseline patches
echo "Applying baseline patches..."
if [ -d "src/engine/cft_engine_patches" ]; then
    cp -r src/engine/cft_engine_patches/* cft_engine/ 2>/dev/null || true
fi
cp src/engine/engine_fixes/cmafm.py cft_engine/models/ 2>/dev/null || true
cp data/M3FD_FLIR.yaml data/m3fd_rgbt.yaml data/mini.yaml cft_engine/data/ 2>/dev/null || true
cp data/yolov5l_cmafm_M3FD.yaml cft_engine/models/ 2>/dev/null || true
cp data/yolov5l_fusion_transformerx3_M3FD.yaml cft_engine/models/transformer/ 2>/dev/null || true

# 4. Run automated ultimate repair (patch_parser)
echo "Running ultimate repair parser to inject automated fixes into cft_engine..."
python3 src/engine/engine_fixes/patch_parser.py

echo "Setup complete."
