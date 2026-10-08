#!/usr/bin/env bash
# Installs the GroundingDINO + SAM2 backends pcseg's segmentation pipeline needs.
# Verified end-to-end against a real scan on 2026-09-06 (Linux, CUDA GPU).
#
# GroundingDINO is a normal pip package (groundingdino-py); it only needs one
# extra file per model size - its model config - fetched directly, no repo
# clone required. SAM2 is not published on PyPI and must be built from source,
# but its repo already ships every size's config, so only the checkpoint
# needs to be fetched per size.
#
# Usage:
#   ./install_segmentation_backends.sh [--dino-size tiny|base] [--sam-size tiny|small|base_plus|large]
#
# Both default to "tiny" (the smallest/fastest of each, and pcseg's own
# defaults - see DEFAULT_CONFIG_PATH/DEFAULT_CHECKPOINT_PATH in
# src/pcseg/services/{dino,sam}_service.py). Larger sizes are slower and use
# more GPU memory in exchange for better detection/segmentation quality;
# there's no way to know which is "enough" without trying it on your own
# scans. Pass the printed --dino-config/--dino-checkpoint/--sam-config/
# --sam-checkpoint flags to `pcseg run` / `auto-segment` / `pick-segment` to
# actually use a non-default size - this script only fetches the weights.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DINO_SIZE="tiny"
SAM_SIZE="tiny"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dino-size) DINO_SIZE="$2"; shift 2 ;;
        --sam-size) SAM_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "$DINO_SIZE" in
    tiny) DINO_CFG_NAME="GroundingDINO_SwinT_OGC.py"
          DINO_CKPT_NAME="groundingdino_swint_ogc.pth"
          DINO_CKPT_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" ;;
    base) DINO_CFG_NAME="GroundingDINO_SwinB_cfg.py"
          DINO_CKPT_NAME="groundingdino_swinb_cogcoor.pth"
          DINO_CKPT_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" ;;
    *) echo "Unknown --dino-size '$DINO_SIZE' (expected tiny|base)" >&2; exit 1 ;;
esac

case "$SAM_SIZE" in
    tiny)      SAM_CFG_NAME="sam2.1_hiera_t.yaml";  SAM_CKPT_NAME="sam2.1_hiera_tiny.pt" ;;
    small)     SAM_CFG_NAME="sam2.1_hiera_s.yaml";  SAM_CKPT_NAME="sam2.1_hiera_small.pt" ;;
    base_plus) SAM_CFG_NAME="sam2.1_hiera_b+.yaml"; SAM_CKPT_NAME="sam2.1_hiera_base_plus.pt" ;;
    large)     SAM_CFG_NAME="sam2.1_hiera_l.yaml";  SAM_CKPT_NAME="sam2.1_hiera_large.pt" ;;
    *) echo "Unknown --sam-size '$SAM_SIZE' (expected tiny|small|base_plus|large)" >&2; exit 1 ;;
esac
SAM_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM_CKPT_NAME}"

SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
# Pinned commit to fetch GroundingDINO's model config files from (raw.githubusercontent.com
# has no tag/release archive for single files, so a commit SHA is the closest equivalent to
# SAM2_COMMIT above). The repo's own early "v0.1.0-alpha[2]" release tags predate the current
# groundingdino/config/ layout (config fetch 404s there), so this pins to a specific recent
# commit instead of the moving `main` branch - verified 2026-09-06 to serve both the
# SwinT and SwinB config files pcseg supports.
DINO_COMMIT="856dde20aee659246248e20734ef9ba5214f5e44"
SAM2_DIR="external/sam2-external"
DINO_DIR="external/GroundingDINO-external"

echo "[1/5] Installing pip-installable segmentation dependencies (torch, pillow, groundingdino-py)..."
python3 -m pip install -e ".[segmentation]"

echo "[2/5] Pre-fetching GroundingDINO's text encoder (bert-base-uncased) so it never needs the network again..."
# GroundingDINO always loads this via `transformers`, even for cached weights, to check
# for updates - pcseg forces offline mode at runtime (see dino_service.py) once this is
# cached, so a flaky connection or SSL-inspecting proxy can't turn every run into a
# multi-second retry loop or an outright failure. If this step itself fails (e.g. the
# same SSL/proxy issue blocking huggingface.co), the rest of the install still completes -
# just re-run this script once you have a plain connection to huggingface.co.
python3 -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased')
print('bert-base-uncased cached.')
" || echo "  (bert-base-uncased pre-fetch failed - see comment above; continuing with the rest of the install)"

echo "[3/5] Fetching GroundingDINO ($DINO_SIZE) config + checkpoint..."
mkdir -p "$DINO_DIR/groundingdino/config" "$DINO_DIR/weights"

if [ ! -f "$DINO_DIR/groundingdino/config/$DINO_CFG_NAME" ]; then
    curl -sL -o "$DINO_DIR/groundingdino/config/$DINO_CFG_NAME" \
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/$DINO_COMMIT/groundingdino/config/$DINO_CFG_NAME"
fi

if [ ! -f "$DINO_DIR/weights/$DINO_CKPT_NAME" ]; then
    curl -sL -o "$DINO_DIR/weights/$DINO_CKPT_NAME" "$DINO_CKPT_URL"
fi

echo "[4/5] Cloning + installing SAM2 from source (pinned commit ${SAM2_COMMIT})..."
if [ ! -d "$SAM2_DIR" ]; then
    git clone --quiet https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
fi
(cd "$SAM2_DIR" && git checkout --quiet "$SAM2_COMMIT")
python3 -m pip install --no-build-isolation -e "$SAM2_DIR"

echo "[5/5] Fetching SAM2 ($SAM_SIZE) checkpoint..."
mkdir -p "$SAM2_DIR/checkpoints"
if [ ! -f "$SAM2_DIR/checkpoints/$SAM_CKPT_NAME" ]; then
    curl -sL -o "$SAM2_DIR/checkpoints/$SAM_CKPT_NAME" "$SAM_CKPT_URL"
fi

echo
echo "Done. Verify with:"
echo "  python3 -c \"from sam2.build_sam import build_sam2; from groundingdino.util.inference import load_model; print('OK')\""
echo
if [ "$DINO_SIZE" != "tiny" ] || [ "$SAM_SIZE" != "tiny" ]; then
    echo "Non-default size(s) fetched - pass these to pcseg to actually use them:"
    [ "$DINO_SIZE" != "tiny" ] && echo "  --dino-config $DINO_DIR/groundingdino/config/$DINO_CFG_NAME --dino-checkpoint $DINO_DIR/weights/$DINO_CKPT_NAME"
    [ "$SAM_SIZE" != "tiny" ] && echo "  --sam-config configs/sam2.1/$SAM_CFG_NAME --sam-checkpoint $SAM2_DIR/checkpoints/$SAM_CKPT_NAME"
fi
