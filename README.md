# PomTum ASR for RK3576

Offline streaming speech recognition (English and Chinese) for Android on the Rockchip RK3576, running
on the NPU. Built for blind users: start and stop with a hardware key, screen-reader-first interface.

- Streaming zipformer transducer models from [icefall](https://github.com/k2-fsa/icefall), converted to RKNN
  with fixes for the RK3576 NPU (see [docs/rk3576-rknn-notes.md](docs/rk3576-rknn-notes.md))
- Own small C recognizer (`app/src/main/cpp`, about 60 KB): streaming fbank, zero-copy RKNN encoder,
  decoder/joiner greedy search, endpointing. No sherpa-onnx or ONNX Runtime in the app.
- RTF about 0.11 and about 4% of one CPU core on the device; models load in 0.7 s straight from the APK

## Repository layout

| path | contents |
|---|---|
| `app/` | Android app (Kotlin) and the native recognizer (C, `app/src/main/cpp`) |
| `tools/convert/` | export icefall checkpoints to ONNX, rewrite the graph for the RK3576 NPU, convert to RKNN |
| `tools/device/` | on-device tools: layout calibration, command-line recognizer, NPU profiling |
| `models/<lang>/encoder_layout.txt` | state layouts for the models built by `build_model.sh` with librknnrt 2.5.0 |
| `scripts/` | fetch the RKNN runtime, install models into the app |
| `docs/` | what went wrong with the published RK3576 models and how it was fixed |

## Build

Requirements: Android SDK (platform 36, build-tools 36), NDK 27.2.12479018, CMake 3.22.1, JDK 17.
The models need Linux (tested on Ubuntu 24.04 aarch64) with Python 3.12.

```bash
scripts/fetch_rknn_runtime.sh                 # librknnrt.so 2.5.0 + rknn_api.h (not redistributed here)
tools/convert/setup_env.sh work               # PyTorch/k2/icefall and rknn-toolkit2 environments
tools/convert/build_model.sh en work          # -> work/out/en
tools/convert/build_model.sh zh work          # -> work/out/zh
scripts/install_models.sh work                # copy into app/src/main/assets/models/
./gradlew assembleDebug                       # or assembleRelease with a keystore.properties
```

`encoder_layout.txt` depends on the model and the runtime. The files in `models/` match models built with
these scripts and librknnrt 2.5.0; if you change either, regenerate them on a device:

```bash
ANDROID_NDK=/path/to/ndk tools/device/build.sh
tools/device/calibrate.sh work/out/en some_16k_mono.wav
```

For a release build, create `keystore.properties` next to `settings.gradle.kts`:

```properties
storeFile=/path/to/release.jks
storePassword=...
keyAlias=...
keyPassword=...
```

## Using the app

- Press the AI key on the left side to start listening; press it again to stop. Start and stop are confirmed
  with a vibration and a short tone. (On the target device the key is the `adc-keys-ai` input device, reported
  as `KEY_A`; the app accepts `KEYCODE_A` from non-alphabetic input devices only, so an external keyboard does
  not trigger it.)
- With TalkBack on, the status and each finished sentence are announced; the partial sentence is not.
- Switch between English and 中文 at the top. The transcript is kept on the device and can be copied,
  shared or cleared.

## Models

| language | checkpoint | license |
|---|---|---|
| English | [Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17](https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17) | see the model page |
| Chinese | [zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05](https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05) | see the model page |

## License

Apache License 2.0, see [LICENSE](LICENSE). The Rockchip RKNN runtime and header are downloaded from
[airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) and are under their own terms.

## Acknowledgements

[k2-fsa](https://github.com/k2-fsa) (icefall, sherpa-onnx, k2) for the models and the reference
implementation, and Rockchip for rknn-toolkit2.
