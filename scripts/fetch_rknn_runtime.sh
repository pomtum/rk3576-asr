#!/usr/bin/env bash
# Download the Rockchip RKNN runtime (librknnrt.so, Android arm64) and its header.
# They are not redistributed in this repository; this pins the exact files the app was tested with
# (RKNN SDK 2.5.0, airockchip/rknn-toolkit2 commit 2c723177) and verifies their SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMIT=2c723177a94fac65cbc036d722907ddbc91980cf
BASE=https://raw.githubusercontent.com/airockchip/rknn-toolkit2/$COMMIT/rknpu2/runtime/Android/librknn_api

fetch() {  # url dest sha256
  mkdir -p "$(dirname "$2")"
  curl -fsSL -o "$2" "$1"
  echo "$3  $2" | sha256sum -c -
}

fetch "$BASE/arm64-v8a/librknnrt.so" app/src/main/jniLibs/arm64-v8a/librknnrt.so \
  7ec1f56dd3f19981283a225a348ece974664e3678c50d47b254b012e56bd5e21
fetch "$BASE/include/rknn_api.h" app/src/main/cpp/include/rknn_api.h \
  2d4160cd868efd7b83a72587e411c16d6972da5876e067bd73665cea778ce666
