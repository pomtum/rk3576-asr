// Kotlin 接口：org.pomtum.asr.NpuAsr。模型从 APK assets 的 models/<lang>/ 内存映射加载。
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "recognizer.h"

#define FN(name) Java_org_pomtum_asr_NpuAsr_##name

typedef struct {
    Recognizer *rec;
    AAsset *assets[3];   // encoder / decoder / joiner，识别器存活期间保持打开
    char *tokens, *layout;
} Handle;

static void throw_state(JNIEnv *env, const char *msg) {
    jclass ex = (*env)->FindClass(env, "java/lang/IllegalStateException");
    (*env)->ThrowNew(env, ex, msg);
}

static AAsset *open_buffer(AAssetManager *am, const char *dir, const char *name, const void **buf, size_t *len) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    AAsset *a = AAssetManager_open(am, path, AASSET_MODE_BUFFER);
    if (!a) return NULL;
    *buf = AAsset_getBuffer(a);
    *len = (size_t)AAsset_getLength(a);
    if (!*buf) { AAsset_close(a); return NULL; }
    return a;
}

static char *read_text(AAssetManager *am, const char *dir, const char *name) {
    const void *buf;
    size_t len;
    AAsset *a = open_buffer(am, dir, name, &buf, &len);
    if (!a) return NULL;
    char *s = malloc(len + 1);
    memcpy(s, buf, len);
    s[len] = 0;
    AAsset_close(a);
    return s;
}

static void handle_free(Handle *h) {
    if (!h) return;
    recognizer_destroy(h->rec);
    for (int i = 0; i < 3; i++) if (h->assets[i]) AAsset_close(h->assets[i]);
    free(h->tokens);
    free(h->layout);
    free(h);
}

JNIEXPORT jlong JNICALL FN(nativeCreate)(JNIEnv *env, jclass cls, jobject asset_manager, jstring dir) {
    AAssetManager *am = AAssetManager_fromJava(env, asset_manager);
    const char *d = (*env)->GetStringUTFChars(env, dir, NULL);
    Handle *h = calloc(1, sizeof(Handle));
    ModelData m;
    memset(&m, 0, sizeof(m));
    char err[256] = "model files missing";
    h->assets[0] = open_buffer(am, d, "encoder.rknn", &m.encoder, &m.encoder_len);
    h->assets[1] = open_buffer(am, d, "decoder.rknn", &m.decoder, &m.decoder_len);
    h->assets[2] = open_buffer(am, d, "joiner.rknn", &m.joiner, &m.joiner_len);
    h->tokens = read_text(am, d, "tokens.txt");
    h->layout = read_text(am, d, "encoder_layout.txt");
    (*env)->ReleaseStringUTFChars(env, dir, d);
    if (h->assets[0] && h->assets[1] && h->assets[2] && h->tokens && h->layout) {
        m.tokens = h->tokens;
        m.layout = h->layout;
        h->rec = recognizer_create(&m, err, sizeof(err));
    }
    if (!h->rec) {
        handle_free(h);
        throw_state(env, err);
        return 0;
    }
    return (jlong)(intptr_t)h;
}

#define REC(h) (((Handle *)(intptr_t)(h))->rec)

JNIEXPORT void JNICALL FN(nativeDestroy)(JNIEnv *env, jclass cls, jlong h) {
    handle_free((Handle *)(intptr_t)h);
}

JNIEXPORT jboolean JNICALL FN(nativeAccept)(JNIEnv *env, jclass cls, jlong h, jfloatArray samples, jint n) {
    jfloat *s = (*env)->GetPrimitiveArrayCritical(env, samples, NULL);
    int ret = recognizer_accept(REC(h), s, (size_t)n);
    (*env)->ReleasePrimitiveArrayCritical(env, samples, s, JNI_ABORT);
    return ret == 0;
}

JNIEXPORT jstring JNICALL FN(nativeText)(JNIEnv *env, jclass cls, jlong h) {
    return (*env)->NewStringUTF(env, recognizer_text(REC(h)));
}

JNIEXPORT jboolean JNICALL FN(nativeIsEndpoint)(JNIEnv *env, jclass cls, jlong h) {
    return recognizer_is_endpoint(REC(h)) != 0;
}

JNIEXPORT void JNICALL FN(nativeReset)(JNIEnv *env, jclass cls, jlong h) {
    recognizer_reset_utterance(REC(h));
}
