#!/bin/bash
set -e  # Salir si hay un error

# === Configuración de rutas ===
TOOLCHAIN=/toolchain/Projects/arm-rockchip830-linux-uclibcgnueabihf
PROJECT_DIR=/toolchain/Projects
SRC=main.c
OUTPUT=/board/mobilenet_run
TF_BUILD=${PROJECT_DIR}/tf-build

CXX=${TOOLCHAIN}/bin/arm-rockchip830-linux-uclibcgnueabihf-g++

INCLUDES="-I${PROJECT_DIR}/tensorflow"

# === Flags de compilador ===
CXXFLAGS="-O3 -flto -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -static"

# === Librerías principales ===
LIBS=(
  "${TF_BUILD}/libtensorflow-lite.a"
  -Wl,--start-group
  ${TF_BUILD}/_deps/ruy-build/ruy/libruy_*.a
  "${TF_BUILD}/_deps/cpuinfo-build/libcpuinfo.a"
  "${TF_BUILD}/_deps/fft2d-build/libfft2d_fftsg.a"
  "${TF_BUILD}/_deps/fft2d-build/libfft2d_fftsg2d.a"
  "${TF_BUILD}/_deps/flatbuffers-build/libflatbuffers.a"
  "${TF_BUILD}/_deps/farmhash-build/libfarmhash.a"
  "${TF_BUILD}/_deps/gemmlowp-build/libeight_bit_int_gemm.a"
  ${TF_BUILD}/_deps/abseil-cpp-build/absl/**/libabsl_*.a
  -Wl,--end-group
  -lpthread -ldl -lm
)

# === Compilación ===
echo "[INFO] Compiling..."
$CXX $INCLUDES $CXXFLAGS "$SRC" "${LIBS[@]}" -o "$OUTPUT"

echo "[SUCCESS] Binary in: $OUTPUT"
