# setup.ps1: High-performance SSD environment setup & engine patching for Windows
# Target Path: $env:USERPROFILE\.venvs\bmvc

$ErrorActionPreference = "Stop"

$VENV_PATH = Join-Path $env:USERPROFILE ".venvs\bmvc"
$PYTHON_EXE = "$VENV_PATH\Scripts\python.exe"

Write-Host "── [1/5] Checking for 'uv' installation ──" -ForegroundColor Cyan
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' is not installed. Please install it with: curl -LsSf https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Red
    exit 1
}

Write-Host "── [2/5] Creating Virtual Environment on SSD (Python 3.10) ──" -ForegroundColor Cyan
$parent_dir = Join-Path $env:USERPROFILE ".venvs"
if (!(Test-Path $parent_dir)) {
    New-Item -ItemType Directory -Path $parent_dir -Force | Out-Null
}

if (Test-Path $VENV_PATH) {
    $current_version = & $PYTHON_EXE --version 2>$null
    if ($current_version -notlike "*3.10*") {
        Write-Host "Existing venv is not Python 3.10. Recreating..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VENV_PATH
    }
}

if (!(Test-Path $VENV_PATH)) {
    uv venv $VENV_PATH --python 3.10
}

Write-Host "── [3/5] Installing Dependencies (CUDA Enabled) ──" -ForegroundColor Cyan
uv pip install --python $PYTHON_EXE -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match

Write-Host "── [4/5] Setting up .env and Cloning Base Engine ──" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Please adjust values if necessary."
}

if (-not (Test-Path "cft_engine")) {
    Write-Host "Cloning base multispectral engine..."
    git clone https://github.com/DocF/multispectral-object-detection.git cft_engine
}

Write-Host "── [5/5] Applying Patches and Running ultimate repair ──" -ForegroundColor Cyan
if (Test-Path "src/engine/cft_engine_patches") {
    Copy-Item -Recurse -Force "src\engine\cft_engine_patches\*" "cft_engine\" -ErrorAction SilentlyContinue
}
Copy-Item -Force "src\engine\engine_fixes\cmafm.py" "cft_engine\models\" -ErrorAction SilentlyContinue
Copy-Item -Force "data\M3FD_FLIR.yaml", "data\m3fd_rgbt.yaml", "data\mini.yaml" "cft_engine\data\" -ErrorAction SilentlyContinue
Copy-Item -Force "data\yolov5l_cmafm_M3FD.yaml" "cft_engine\models\" -ErrorAction SilentlyContinue

& $PYTHON_EXE src/engine/engine_fixes/patch_parser.py

Write-Host "`n✅ Setup Complete!" -ForegroundColor Green
Write-Host "To activate, use: & $VENV_PATH\Scripts\Activate.ps1"
