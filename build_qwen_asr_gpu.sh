#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-qwen-asr-vulkan}"
JOBS="${JOBS:-$(nproc)}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_VULKAN=ON \
    -DNCNN_SIMPLEVK=ON \
    -DNCNN_SIMPLEOCV=ON \
    -DNCNN_BUILD_EXAMPLES=ON \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TOOLS=OFF

cmake --build "${BUILD_DIR}" --target qwen_asr_0_6_b qwen_asr_1_7_b -j"${JOBS}"

echo "Built:"
echo "  ${BUILD_DIR}/examples/qwen_asr_0_6_b"
echo "  ${BUILD_DIR}/examples/qwen_asr_1_7_b"
