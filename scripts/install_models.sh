#!/usr/bin/env bash
# Copy built models into the app assets (app/src/main/assets/models/<lang>/).
# Usage: scripts/install_models.sh <work_dir>   (expects <work_dir>/out/{en,zh}/ from tools/convert/build_model.sh)
# encoder_layout.txt is taken from models/<lang>/ unless the model directory has its own (see tools/device/calibrate.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
W=$1
for lang in en zh; do
  src=$W/out/$lang
  [ -d "$src" ] || { echo "skip $lang (no $src)"; continue; }
  dst=app/src/main/assets/models/$lang
  mkdir -p "$dst"
  cp "$src/encoder.rknn" "$src/decoder.rknn" "$src/joiner.rknn" "$src/tokens.txt" "$dst/"
  if [ -f "$src/encoder_layout.txt" ]; then cp "$src/encoder_layout.txt" "$dst/"; else cp "models/$lang/encoder_layout.txt" "$dst/"; fi
  echo "installed $lang"
done
