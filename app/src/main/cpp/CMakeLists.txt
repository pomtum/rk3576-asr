cmake_minimum_required(VERSION 3.22)
project(npuasr C)

set(CMAKE_C_STANDARD 11)

# Rockchip RKNN 运行时（随 APK 分发，版本见 jniLibs）
add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/librknnrt.so)

add_library(npuasr SHARED
    fbank.c
    encoder_zc.c
    recognizer.c
    asr_jni.c)
target_include_directories(npuasr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(npuasr PRIVATE -O2 -Wall)
target_link_libraries(npuasr rknnrt android m)
