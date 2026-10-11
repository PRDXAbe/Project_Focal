#!/usr/bin/env bash
# Download the pretrained MinkUNet-34 (spconv) SemanticKITTI checkpoint (~150 MB, too big for git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL=https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti_20230512_233817-72b200d8.pth
mkdir -p "$ROOT/checkpoints"
curl -L --fail -o "$ROOT/checkpoints/minkunet.pth" "$URL"
echo "saved $ROOT/checkpoints/minkunet.pth"
