#!/usr/bin/env bash
# Create the two Python environments used to build the models (tested on Ubuntu 24.04 aarch64, Python 3.12):
#   venv-export : PyTorch 2.11 (CPU) + k2 + icefall, to export the checkpoints to ONNX
#   venv-rknn   : rknn-toolkit2 2.3.2, to convert ONNX to RKNN for RK3576
# Usage: tools/convert/setup_env.sh [work_dir]   (default: ./work)
set -euo pipefail
W=$(realpath -m "${1:-work}")
mkdir -p "$W"

case "$(uname -m)" in
  aarch64) K2_ARCH=linux-arm64; K2_PLAT=manylinux_2_27_aarch64.manylinux_2_28_aarch64 ;;
  x86_64)  K2_ARCH=linux-x64;   K2_PLAT=manylinux_2_27_x86_64.manylinux_2_28_x86_64 ;;
  *) echo "unsupported architecture"; exit 1 ;;
esac
# k2 CPU wheels: https://k2-fsa.github.io/k2/cpu.html
K2_WHEEL="https://huggingface.co/csukuangfj2/k2/resolve/main/cpu/1.24.4.dev20260625/$K2_ARCH/k2-1.24.4.dev20260625+cpu.torch2.11.0-cp312-cp312-$K2_PLAT.whl"

if [ ! -x "$W/venv-export/bin/python" ]; then
  python3 -m venv "$W/venv-export"
  "$W/venv-export/bin/pip" install -q --upgrade pip
  "$W/venv-export/bin/pip" install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
  "$W/venv-export/bin/pip" install -q "$K2_WHEEL"
  [ -d "$W/icefall" ] || git clone -q --depth 1 https://github.com/k2-fsa/icefall.git "$W/icefall"
  grep -v -i -E "^(torch|k2|kaldifeat|#)|^$" "$W/icefall/requirements.txt" > "$W/icefall-req.txt"
  "$W/venv-export/bin/pip" install -q -r "$W/icefall-req.txt" lhotse urllib3 requests onnxsim
fi

if [ ! -x "$W/venv-rknn/bin/python" ]; then
  python3 -m venv "$W/venv-rknn"
  "$W/venv-rknn/bin/pip" install -q --upgrade pip
  "$W/venv-rknn/bin/pip" install -q rknn-toolkit2==2.3.2 "setuptools<81"
fi
echo "environments ready in $W"
