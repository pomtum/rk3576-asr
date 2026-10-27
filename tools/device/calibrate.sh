#!/usr/bin/env bash
# Generate encoder_layout.txt for a model on a connected RK3576 device (adb, root not required).
# The layout describes how each encoder state is laid out in NPU memory (native output layout); the app
# copies states between chunks with it. It depends on the model and on librknnrt, so re-run it after
# rebuilding either one.
# Usage: tools/device/calibrate.sh <model_dir> <some_16k_mono.wav>   (set ANDROID_SERIAL to pick a device)
#        PYTHON=python3 by default; numpy is required
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=$1
WAV=$2
D=/data/local/tmp/rk3576-asr
OUT=tools/device/out
"${PYTHON:-python3}" tools/device/make_feats.py "$WAV" "$OUT/feats.f32" 77 64 > /dev/null
adb shell mkdir -p $D
# one file per push: some adb builds truncate file names when pushing several files at once
adb push "$OUT/zcbench" $D/zcbench > /dev/null
adb push "$OUT/librknnrt.so" $D/librknnrt.so > /dev/null
adb push "$MODEL/encoder.rknn" $D/encoder.rknn > /dev/null
adb push "$OUT/feats.f32.unit.f32" $D/feats.f32 > /dev/null
adb shell "cd $D && chmod 755 zcbench && LD_LIBRARY_PATH=$D ./zcbench encoder.rknn feats.f32 8"
adb pull $D/encoder_layout.txt "$MODEL/encoder_layout.txt" > /dev/null
echo "wrote $MODEL/encoder_layout.txt"
