<#
.SYNOPSIS
Installs the GroundingDINO + SAM2 backends pcseg's segmentation pipeline needs, on native Windows.

.DESCRIPTION
NOT independently verified end-to-end - this project was developed and tested
on Linux. SAM2's own installation docs explicitly recommend WSL2 over native
Windows ("If you are installing on Windows, it's strongly recommended to use
Windows Subsystem for Linux (WSL) with Ubuntu") - see README.md's "Windows"
section for that path, which mirrors install_segmentation_backends.sh
unchanged inside WSL2. This script is the native-Windows alternative for
anyone who wants to try that anyway.

SAM2 is built from source and includes an optional CUDA C++ extension, which
needs a working MSVC + CUDA Toolkit setup to compile. If it fails to build,
the install still completes: SAM2 falls back to a pure-Python path that only
skips a minor mask post-processing step (removing small holes/sprinkles),
which "shouldn't affect the results in most cases" per SAM2's own docs.

.PARAMETER DinoSize
tiny (default, smallest/fastest) or base.

.PARAMETER SamSize
tiny (default, smallest/fastest), small, base_plus, or large.

.EXAMPLE
.\scripts\install_segmentation_backends.ps1
.\scripts\install_segmentation_backends.ps1 -DinoSize base -SamSize large
#>
param(
    [ValidateSet("tiny", "base")]
    [string]$DinoSize = "tiny",
    [ValidateSet("tiny", "small", "base_plus", "large")]
    [string]$SamSize = "tiny"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

# Invoke-WebRequest's default progress-bar rendering is known to make large
# downloads dramatically slower on Windows PowerShell 5.1 (sometimes 10-100x) -
# this has nothing to do with actual network speed, it's UI overhead.
$ProgressPreference = 'SilentlyContinue'

switch ($DinoSize) {
    "tiny" {
        $DinoCfgName = "GroundingDINO_SwinT_OGC.py"
        $DinoCkptName = "groundingdino_swint_ogc.pth"
        $DinoCkptUrl = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    }
    "base" {
        $DinoCfgName = "GroundingDINO_SwinB_cfg.py"
        $DinoCkptName = "groundingdino_swinb_cogcoor.pth"
        $DinoCkptUrl = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    }
}

switch ($SamSize) {
    "tiny"      { $SamCfgName = "sam2.1_hiera_t.yaml";  $SamCkptName = "sam2.1_hiera_tiny.pt" }
    "small"     { $SamCfgName = "sam2.1_hiera_s.yaml";  $SamCkptName = "sam2.1_hiera_small.pt" }
    "base_plus" { $SamCfgName = "sam2.1_hiera_b+.yaml"; $SamCkptName = "sam2.1_hiera_base_plus.pt" }
    "large"     { $SamCfgName = "sam2.1_hiera_l.yaml";  $SamCkptName = "sam2.1_hiera_large.pt" }
}
$SamCkptUrl = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/$SamCkptName"

# Same pins as install_segmentation_backends.sh - keep both in sync if these ever change.
$Sam2Commit = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
$DinoCommit = "856dde20aee659246248e20734ef9ba5214f5e44"
$Sam2Dir = "external/sam2-external"
$DinoDir = "external/GroundingDINO-external"

foreach ($cmd in @("python", "git")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "'$cmd' not found on PATH. Install it and re-run this script."
        exit 1
    }
}

Write-Host "[1/5] Installing pip-installable segmentation dependencies (torch, pillow, groundingdino-py)..."
python -m pip install -e ".[segmentation]"

Write-Host "[2/5] Pre-fetching GroundingDINO's text encoder (bert-base-uncased) so it never needs the network again..."
Write-Host "      GroundingDINO always loads this via transformers, even for cached weights, to" -ForegroundColor Yellow
Write-Host "      check for updates - pcseg forces offline mode at runtime (see dino_service.py)" -ForegroundColor Yellow
Write-Host "      once this is cached, so a flaky connection or an SSL-inspecting proxy can't turn" -ForegroundColor Yellow
Write-Host "      every run into a retry loop or an outright failure. If this step itself fails" -ForegroundColor Yellow
Write-Host "      (e.g. the same SSL/proxy issue blocking huggingface.co), the rest of the install" -ForegroundColor Yellow
Write-Host "      still completes - just re-run this script once you have a plain connection." -ForegroundColor Yellow
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased'); print('bert-base-uncased cached.')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  (bert-base-uncased pre-fetch failed - see message above; continuing with the rest of the install)" -ForegroundColor Yellow
}

Write-Host "[3/5] Fetching GroundingDINO ($DinoSize) config + checkpoint..."
New-Item -ItemType Directory -Force -Path "$DinoDir/groundingdino/config" | Out-Null
New-Item -ItemType Directory -Force -Path "$DinoDir/weights" | Out-Null

$DinoCfgPath = Join-Path $DinoDir "groundingdino/config/$DinoCfgName"
if (-not (Test-Path $DinoCfgPath)) {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/$DinoCommit/groundingdino/config/$DinoCfgName" -OutFile $DinoCfgPath
}

$DinoCkptPath = Join-Path $DinoDir "weights/$DinoCkptName"
if (-not (Test-Path $DinoCkptPath)) {
    Invoke-WebRequest -Uri $DinoCkptUrl -OutFile $DinoCkptPath
}

Write-Host "[4/5] Cloning + installing SAM2 from source (pinned commit $Sam2Commit)..."
Write-Host "      If this step fails to compile SAM2's CUDA extension (needs MSVC Build Tools" -ForegroundColor Yellow
Write-Host "      + a matching CUDA Toolkit), the install still completes with a slower," -ForegroundColor Yellow
Write-Host "      pure-Python fallback - see this script's header comment for details." -ForegroundColor Yellow
if (-not (Test-Path $Sam2Dir)) {
    git clone --quiet https://github.com/facebookresearch/sam2.git $Sam2Dir
}
Push-Location $Sam2Dir
git checkout --quiet $Sam2Commit
Pop-Location
python -m pip install --no-build-isolation -e $Sam2Dir

Write-Host "[5/5] Fetching SAM2 ($SamSize) checkpoint..."
New-Item -ItemType Directory -Force -Path "$Sam2Dir/checkpoints" | Out-Null
$SamCkptPath = Join-Path $Sam2Dir "checkpoints/$SamCkptName"
if (-not (Test-Path $SamCkptPath)) {
    Invoke-WebRequest -Uri $SamCkptUrl -OutFile $SamCkptPath
}

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  python -c `"from sam2.build_sam import build_sam2; from groundingdino.util.inference import load_model; print('OK')`""
Write-Host ""
if ($DinoSize -ne "tiny" -or $SamSize -ne "tiny") {
    Write-Host "Non-default size(s) fetched - pass these to pcseg to actually use them:"
    if ($DinoSize -ne "tiny") {
        Write-Host "  --dino-config $DinoDir/groundingdino/config/$DinoCfgName --dino-checkpoint $DinoDir/weights/$DinoCkptName"
    }
    if ($SamSize -ne "tiny") {
        Write-Host "  --sam-config configs/sam2.1/$SamCfgName --sam-checkpoint $Sam2Dir/checkpoints/$SamCkptName"
    }
}
