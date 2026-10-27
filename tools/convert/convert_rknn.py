"""Convert an icefall streaming zipformer (encoder/decoder/joiner ONNX, chunk-32 / left-128 export)
to RKNN for RK3576 with rknn-toolkit2: fp16, no quantization (same settings as icefall CI).

The encoder gets the icefall metadata as the RKNN custom string (decode_chunk_len, T, ...).
As a sanity check the encoder is also run on the toolkit simulator for one chunk with zero
states and compared against ONNX Runtime.

Usage: python convert_rknn.py <onnx_dir> <out_dir> [features.f32]
  features.f32: optional, T*80 float32 fbank frames for the check (random input if omitted)
"""
import glob
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN

KEYS = ['model_type', 'decode_chunk_len', 'T', 'num_encoder_layers', 'encoder_dims', 'cnn_module_kernels',
        'left_context_len', 'query_head_dims', 'value_head_dims', 'num_heads']


def find(d, prefix):
    return sorted(f for f in glob.glob(f'{d}/{prefix}-*chunk-32-left-128.onnx') if '.int8.' not in f)[0]


def build(onnx_path, custom_string=None):
    r = RKNN(verbose=False)
    r.config(target_platform='rk3576', custom_string=custom_string)
    assert r.load_onnx(model=onnx_path) == 0, onnx_path
    assert r.build(do_quantization=False) == 0, onnx_path
    return r


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    sess = ort.InferenceSession(find(src, 'encoder'), providers=['CPUExecutionProvider'])
    meta = sess.get_modelmeta().custom_metadata_map
    custom = ';'.join(f'{k}={meta[k]}' for k in KEYS)
    T = int(meta['T'])
    print('custom_string:', custom)

    if len(sys.argv) > 3:
        x = np.fromfile(sys.argv[3], np.float32).reshape(-1, T, 80)[:1]
    else:
        x = np.random.default_rng(0).uniform(-12, 2, size=(1, T, 80)).astype(np.float32)
    feeds = {'x': x}
    for i in sess.get_inputs()[1:]:
        feeds[i.name] = np.zeros(i.shape, np.int64 if 'int64' in i.type else np.float32)
    ref = sess.run(None, feeds)[0]

    for name in ('encoder', 'decoder', 'joiner'):
        r = build(find(src, name), custom if name == 'encoder' else None)
        assert r.export_rknn(f'{dst}/{name}.rknn') == 0
        if name == 'encoder' and r.init_runtime() == 0:
            # the simulator expects 4-D inputs in NHWC
            ins = [feeds[i.name].transpose(0, 2, 3, 1) if feeds[i.name].ndim == 4 else feeds[i.name]
                   for i in sess.get_inputs()]
            sim = np.asarray(r.inference(inputs=ins)[0], np.float32)
            err = np.linalg.norm(sim.ravel() - ref.ravel()) / np.linalg.norm(ref)
            print(f'simulator vs onnx (encoder, chunk 0): rel_err={err:.4f} finite={bool(np.isfinite(sim).all())}')
        r.release()
        print('exported', name)
    shutil.copy(f'{src}/tokens.txt', f'{dst}/tokens.txt')


if __name__ == '__main__':
    main()
