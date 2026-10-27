#!/usr/bin/env bash
# Build an RK3576 RKNN streaming zipformer model (encoder/decoder/joiner + tokens) from an icefall checkpoint:
#   export ONNX (chunk 32, left context 128, causal) -> onnxsim
#   -> rewrite 1: bool-mask Expand + Where      -> float multiply-add   (Expand of a bool mask hangs/overflows on the RK3576 NPU)
#   -> rewrite 2: Max(0,x), log(1+exp(x)) guard -> Relu, stable softplus (they otherwise fall back to the CPU)
#   -> rewrite 3: remaining bool Where           -> float multiply-add   (CPU fallback)
#   -> rknn-toolkit2 2.3.2, fp16, target rk3576
# Usage: tools/convert/build_model.sh <en|zh> [work_dir]   (run setup_env.sh first)
# Output: <work_dir>/out/<lang>/{encoder,decoder,joiner}.rknn tokens.txt
set -euo pipefail
LANG_ID=$1
W=$(realpath -m "${2:-work}")
HERE=$(cd "$(dirname "$0")" && pwd)
N=epoch-99-avg-1-chunk-32-left-128

case "$LANG_ID" in
  en) RECIPE=librispeech
      HF=https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17/resolve/main
      TOKENS=data/lang_bpe_500/tokens.txt ;;
  zh) RECIPE=multi_zh-hans
      HF=https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05/resolve/main
      TOKENS=data/lang_bpe_2000/tokens.txt ;;
  *) echo "usage: $0 <en|zh> [work_dir]"; exit 2 ;;
esac

ASR=$W/icefall/egs/$RECIPE/ASR
D=$ASR/exp_$LANG_ID
S=$W/stage/$LANG_ID
OUT=$W/out/$LANG_ID
mkdir -p "$D" "$S" "$OUT"
[ -s "$D/epoch-99.pt" ] || curl -fsSL -o "$D/epoch-99.pt" "$HF/exp/pretrained.pt"
[ -s "$D/tokens.txt" ] || curl -fsSL -o "$D/tokens.txt" "$HF/$TOKENS"

PY=$W/venv-export/bin/python
if [ ! -s "$D/encoder-$N.onnx" ]; then
  (cd "$ASR" && PYTHONPATH="$W/icefall" "$PY" ./zipformer/export-onnx-streaming.py \
    --dynamic-batch 0 --enable-int8-quantization 0 \
    --tokens "$D/tokens.txt" --use-averaged-model 0 --epoch 99 --avg 1 --exp-dir "$D" \
    --use-ctc 0 --use-transducer 1 \
    --chunk-size 32 --left-context-frames 128 --causal 1)
fi

"$PY" -s - "$D/encoder-$N.onnx" "$S/sim.onnx" <<'EOF'
import sys, onnx, onnxsim
m = onnx.load(sys.argv[1])
s, ok = onnxsim.simplify(m)
assert ok
onnx.save(s, sys.argv[2])
EOF
"$PY" -s "$HERE/mask_where_to_mul.py" "$S/sim.onnx" "$S/mask.onnx"
"$PY" -s "$HERE/cpu_fallback_rewrite.py" "$S/mask.onnx" "$S/npu.onnx"
mkdir -p "$S/final"
"$PY" -s "$HERE/where_to_arith.py" "$S/npu.onnx" "$S/final/encoder-$N.onnx"
cp "$D/decoder-$N.onnx" "$D/joiner-$N.onnx" "$D/tokens.txt" "$S/final/"

# The rewritten encoder must match the original (zero state, random-ish input)
"$PY" -s - "$D/encoder-$N.onnx" "$S/final/encoder-$N.onnx" <<'EOF'
import sys, numpy as np, onnxruntime as ort
def run(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    feeds = {}
    for i in s.get_inputs():
        if i.name == "x":
            feeds[i.name] = rng.uniform(-12, 2, size=i.shape).astype(np.float32)
        else:
            feeds[i.name] = np.zeros(i.shape, np.int64 if "int64" in i.type else np.float32)
    return s.run(None, feeds)[0]
a, b = run(sys.argv[1]), run(sys.argv[2])
err = float(np.linalg.norm(a - b) / np.linalg.norm(a))
print("rewritten encoder vs original: rel_err", err)
assert err < 1e-4
EOF

(cd "$W" && "$W/venv-rknn/bin/python" -s "$HERE/convert_rknn.py" "$S/final" "$OUT")
echo "model ready in $OUT"
