# scripts/build_supplementary.ps1
# Packages the CMAFM repository for WACV 2027 double-blind supplementary submission.
# Produces an anonymized, self-contained zip under 200MB.

param(
    [switch]$WithVideos = $false,
    [switch]$WithFigures = $false
)

$ErrorActionPreference = "Stop"

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$zipName = "CMAFM_Supplementary_WACV2027.zip"
$zipPath = Join-Path (Split-Path $repoRoot -Parent) $zipName

Write-Host "=== CMAFM WACV 2027 Supplementary Packager (PowerShell) ===" -ForegroundColor Cyan
Write-Host "Source: $repoRoot"
Write-Host "Target: $zipPath"
Write-Host "Include videos: $WithVideos"
Write-Host "Include high-res figures: $WithFigures"

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

$tempBase = [System.IO.Path]::GetTempPath()
$tempFolder = Join-Path $tempBase ("cmafm_supp_" + [System.Guid]::NewGuid().ToString("N"))
$codeDir = Join-Path $tempFolder "code"

New-Item -ItemType Directory -Path $codeDir -Force | Out-Null

try {
    Write-Host "`n[*] Copying repository to temp directory..." -ForegroundColor Yellow
    
    # Robocopy to temp excluding git, caches, weights, and build dirs
    $excludeDirs = @(".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints", "cft_engine", "logs", "build", "dist", ".idea", ".vscode")
    $excludeFiles = @(".env", "*.pt", "*.pth", "*.ckpt", "*.engine", "*.onnx", "*.zip", "desktop.ini", "Thumbs.db", ".DS_Store", "ssh_tunnel_watchdog.log")
    
    if (-not $WithVideos) {
        $excludeFiles += @("flir_v1_rgb.mp4", "flir_v1_thermal.mp4", "video_demo.mp4")
    }

    $cmd = "robocopy `"$repoRoot`" `"$codeDir`" /E /XD $($excludeDirs -join ' ') /XF $($excludeFiles -join ' ') /NFL /NDL /NJH /NJS"
    Invoke-Expression $cmd | Out-Null

    # If WithFigures flag is set, bundle high-res figures from wacv-2027-author-kit-template/images/
    if ($WithFigures) {
        $figuresSource = Join-Path (Split-Path $repoRoot -Parent) "wacv-2027-author-kit-template\images"
        if (Test-Path $figuresSource) {
            Write-Host "[*] Bundling high-resolution paper figures into docs/figures/..." -ForegroundColor Yellow
            $figuresDest = Join-Path $codeDir "docs\figures"
            New-Item -ItemType Directory -Path $figuresDest -Force | Out-Null
            Copy-Item -Path "$figuresSource\*" -Destination $figuresDest -Recurse -Force
        }
    }

    # Clean any stray files
    Get-ChildItem -Path $codeDir -Filter "desktop.ini" -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Path $codeDir -Filter "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

    Write-Host "[*] Compressing archive to $zipName..." -ForegroundColor Yellow
    Compress-Archive -Path "$codeDir" -DestinationPath $zipPath -CompressionLevel Optimal -Force

    $sizeBytes = (Get-Item $zipPath).Length
    $sizeMB = [Math]::Round($sizeBytes / 1MB, 2)
    Write-Host "`n[SUCCESS] Supplementary archive created!" -ForegroundColor Green
    Write-Host "Location: $zipPath" -ForegroundColor Green
    Write-Host "Size: $sizeMB MB" -ForegroundColor Green

    # Anonymity verification
    Write-Host "`n[*] Running automated anonymity scan on archive entries..." -ForegroundColor Yellow
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    $piiHits = $zip.Entries | Where-Object { $_.FullName -match '(nps|naval|postgraduate|mingu|owenk)' }
    $zip.Dispose()

    if ($piiHits) {
        Write-Warning "Potential PII found in archive filenames:"
        $piiHits | ForEach-Object { Write-Warning $_.FullName }
    } else {
        Write-Host "[PASS] No author/institutional PII detected in archive paths!" -ForegroundColor Green
    }
}
finally {
    if (Test-Path $tempFolder) {
        Remove-Item -Recurse -Force $tempFolder -ErrorAction SilentlyContinue
    }
}
