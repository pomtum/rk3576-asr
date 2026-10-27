"""Compute 80-dim fbank (kaldi-native-fbank defaults, samples in [-1, 1]) from a 16 kHz wav and cut it into
streaming encoder chunks.

Usage: python make_feats.py <in.wav> <out.f32> [T] [shift]
  T      frames per chunk (77 for the chunk-32 models)
  shift  frames between chunks (decode_chunk_len, 64)
Writes <out> (samples scaled to int16 range), <out>.unit.f32 (samples in [-1, 1]; this is what the models expect)
and <out>.zero.f32 (two all-zero chunks).
"""

import sys
import wave

import numpy as np


def mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def fbank(samples, sr=16000, num_bins=80, low=20.0, high=-400.0):
    flen, fshift, nfft = 400, 160, 512
    # snip_edges=False：两端反射补齐，帧中心对齐 k*shift
    n = len(samples)
    num_frames = (n + fshift // 2) // fshift
    pad_l = flen // 2 - fshift // 2
    x = np.concatenate([samples[:pad_l][::-1], samples, samples[::-1][: flen]])
    frames = np.stack([x[i * fshift: i * fshift + flen] for i in range(num_frames)]).astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    pre = frames.copy()
    pre[:, 1:] -= 0.97 * frames[:, :-1]
    pre[:, 0] -= 0.97 * frames[:, 0]
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(flen) / (flen - 1))) ** 0.85
    spec = np.abs(np.fft.rfft(pre * win, nfft)) ** 2

    high = sr / 2 + high if high <= 0 else high
    mlo, mhi = mel(low), mel(high)
    centers = np.linspace(mlo, mhi, num_bins + 2)
    fft_mel = mel(np.arange(nfft // 2 + 1) * sr / nfft)
    banks = np.zeros((num_bins, nfft // 2 + 1))
    for b in range(num_bins):
        l, c, r = centers[b], centers[b + 1], centers[b + 2]
        up = (fft_mel - l) / (c - l)
        down = (r - fft_mel) / (r - c)
        banks[b] = np.clip(np.minimum(up, down), 0, None)
        banks[b][(fft_mel <= l) | (fft_mel >= r)] = 0
    e = spec @ banks.T
    return np.log(np.maximum(e, np.finfo(np.float32).eps)).astype(np.float32)


def main():
    src, out = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 77
    shift = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    w = wave.open(src)
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    # sherpa-onnx 的 fbank 输入是 [-1,1] 浮点，Kaldi 本身按 int16 量级；两者 log 能量差常数 2*ln(32768)
    for scale, path in ((32768.0, out), (1.0, out + '.unit.f32')):
        feats = fbank(pcm * scale)
        chunks = []
        i = 0
        while i + T <= len(feats):
            chunks.append(feats[i: i + T])
            i += shift
        np.stack(chunks).tofile(path)
        print(f'{path}: scale={scale:g} frames={len(feats)} chunks={len(chunks)} '
              f'range=[{feats.min():.2f},{feats.max():.2f}] mean={feats.mean():.2f}')
    np.zeros((2, T, 80), np.float32).tofile(out + '.zero.f32')


if __name__ == '__main__':
    main()
