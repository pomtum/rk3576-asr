#!/usr/bin/env bash
# Build the on-device tools (Android arm64) with the NDK. Run scripts/fetch_rknn_runtime.sh first.
# Usage: ANDROID_NDK=/path/to/ndk tools/device/build.sh   -> tools/device/out/
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${ANDROID_NDK:?set ANDROID_NDK to the NDK directory (tested with r27c)}"
HOST=$(ls "$ANDROID_NDK/toolchains/llvm/prebuilt" | head -1)
CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$HOST/bin/aarch64-linux-android29-clang"
[ -x "$CC" ] || CC="$CC.cmd"   # Windows NDK
CPP=app/src/main/cpp
RT=app/src/main/jniLibs/arm64-v8a
OUT=tools/device/out
mkdir -p "$OUT"
FLAGS=(-O2 -Wall -I "$CPP/include" -I "$CPP" -L "$RT" -lrknnrt -lm)
"$CC" tools/device/zcbench.c -o "$OUT/zcbench" "${FLAGS[@]}"
"$CC" tools/device/perfdetail.c -o "$OUT/perfdetail" "${FLAGS[@]}"
"$CC" tools/device/npucheck.c -o "$OUT/npucheck" "${FLAGS[@]}"
"$CC" -DASR_PROFILE tools/device/asrcli.c "$CPP/fbank.c" "$CPP/encoder_zc.c" "$CPP/recognizer.c" -o "$OUT/asrcli" "${FLAGS[@]}"
cp "$RT/librknnrt.so" "$OUT/"
echo "built into $OUT"
