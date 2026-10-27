# Streaming zipformer on the RK3576 NPU: what went wrong and how it was fixed

Device used for all measurements: RK3576 (4× Cortex-A72 + 4× Cortex-A53), Android 16 (vendor Android 14),
kernel 6.1, RKNPU driver 0.9.8, NPU at 950 MHz.

## 1. The published RK3576 encoders give wrong results on the device

sherpa-onnx (tested 1.12.21 and 2.0.1, `sherpa-onnx-*-rknn.aar`) with the published models
`sherpa-onnx-rk3576-streaming-zipformer-en-2023-06-26` and
`sherpa-onnx-rk3576-streaming-zipformer-small-bilingual-zh-en-2023-02-16` produces garbage on the device:
the English model repeats the same tokens for every input (`UN COUNT WENT COUNT COUNT ...`), the bilingual
model returns empty text. The NPU itself is fine (rknn-toolkit2's mobilenet example gives the expected top-1),
and the CPU path (sherpa-onnx int8 ONNX) is correct.

Checking each stage separately with small C programs that call the RKNN API directly (no sherpa-onnx):

| stage | result |
|---|---|
| fbank features | correct (the ONNX models decode them correctly) |
| decoder.rknn | matches ONNX, rel. error 0.0002 |
| joiner.rknn | matches ONNX, rel. error 0.001, same argmax |
| state passing between chunks | not involved: the first chunk with zero states is already wrong |
| **encoder.rknn (en)** | **all outputs are the same `inf` values for any input** (librknnrt 2.2.0, 2.3.2 and 2.5.0 alike) |
| **encoder.rknn (bilingual)** | finite but wrong: the ONNX decoder/joiner cannot decode it either |

So sherpa-onnx's code is fine; the encoder `.rknn` files are not. They were built with rknn-toolkit2 2.2.0
and, as far as we can tell, only checked on the toolkit simulator.

## 2. Root cause: Expand of a bool mask

Re-exporting the same checkpoint with the icefall CI settings and converting with rknn-toolkit2 2.3.2:

| build | simulator | RK3576 device |
|---|---|---|
| toolkit 2.3.2 | matches ONNX (rel. error 0.0011) | NPU job timeout at op `Expand`, driver soft reset, `rknn_run` fails |
| + onnxsim | same | same |
| + Expand replaced by Tile | same | same (the toolkit turns Tile back into Expand) |
| **+ mask rewrite** | matches ONNX | **correct** (cosine 0.999 vs ONNX fp32) |

The conv modules of zipformer2 apply the padding mask with `masked_fill`, which ONNX export turns into
`Where(Expand(mask, [1, C, T]), 0, x)` where `mask` is a small bool tensor `[1, 1, T]`. Executing that bool
Expand on the RK3576 NPU goes wrong (inf with the 2.2.0 compiler, a hang with 2.3.2).

Fix (`tools/convert/mask_where_to_mul.py`): `Where(Expand(m), a, b)` → `b + (a - b) * Cast(m, float)`,
keeping `m` in its small shape and letting `Mul` broadcast. The rewritten ONNX matches the original to 5e-7.

## 3. CPU fallback ops

With the fix above the encoder runs, but RKNN's per-op profile shows 318 of 2698 ops on the CPU
(119 ms per chunk on an A72):

| op | count | where | rewrite |
|---|---|---|---|
| `Equal` + `Where` | 80 | SwooshL fallback `where(log(1+exp(x)) == inf, x, ...)` in the feed-forward modules | stable softplus `Relu(x) + log(1 + exp(-|x|))` |
| `Max(0, x)` | 4 | `logaddexp` in the conv embed | `Relu(x)` |
| `Where(mask, -1000, scores)` | 16 | attention masks | `scores + (-1000 - scores) * float(mask)` |

After `cpu_fallback_rewrite.py` and `where_to_arith.py` only the input/output operators remain on the CPU
(2.4 ms). `rknn_run` went from 193 ms wall / 114 ms CPU to about 100 ms wall per 640 ms chunk.

## 4. Zero-copy state passing

The encoder has 98 state tensors. With `rknn_inputs_set` / `rknn_outputs_get` the runtime converts every
state from the NPU's native output layout to float32 and back each chunk: 80 ms of CPU per chunk.

The native layouts differ between outputs and inputs: some outputs are NC1HWC2 with 8× channel padding,
some 3-D outputs are NC1HWC2 with the last dimension as the channel, some are plain with tail padding, and
one input (`embed_states`) is NC1HWC2 and requires `pass_through = 1`. The same byte size can correspond to
different layouts, so the mapping is not guessed: `tools/device/zcbench.c` runs the regular API as a reference,
picks for every state the formula that matches it element by element, and writes `encoder_layout.txt`.
The app (`app/src/main/cpp/encoder_zc.c`) binds all tensors with `rknn_create_mem` / `rknn_set_io_mem` and
copies states with tight loops (3.8 ms per chunk). Over 25 chunks the output is bit-identical to the regular API.

## 5. Runtime version

On a 67 s recording, pinned to the A72 cores, CPU time of `rknn_run` per second of audio:
librknnrt 2.2.0 66.5 ms, 2.3.2 35.9 ms, **2.5.0 24.9 ms**. 2.5.0 loads the 2.3.2-compiled models and gives
bit-identical results. Note that sherpa-onnx's AAR links against symbols exported by its bundled 2.2.0
runtime (`__cxa_thread_atexit`) and fails to load with 2.5.0; sherpa-onnx with 2.3.2 crashes in `decode`.

## 6. Result

Whole pipeline (fbank, zero-copy encoder, decoder and joiner on the NPU, greedy search), librknnrt 2.5.0:

| | English (LibriSpeech test clips) | Chinese (multi-zh-hans test clips) |
|---|---|---|
| RTF (A72) | 0.11 | 0.11 |
| CPU | 4.1% of one core | 3.8% of one core |
| text | both clips correct | 3 clips correct except one character |
| model load (from APK, mmap) | 0.7 s | 0.7 s |

For comparison, sherpa-onnx int8 ONNX on one CPU thread: RTF 0.31, about 44% of one core, 14 s to load.

Running the recognizer thread on the A53 cores instead roughly doubles the CPU time spent inside `rknn_run`.
