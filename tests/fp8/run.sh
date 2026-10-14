#!/usr/bin/env bash
# Build the ggml CPU libraries and run the F8_E4M3_B128 unit test.
# Run from the repository root on the Linux host, inside the dev image:
#   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/src -w /src drama3-lowmem-dev tests/fp8/run.sh
set -euo pipefail
build_dir=${BUILD_DIR:-build/linux-cuda-release}
cmake --build "$build_dir" --target ggml-cpu ggml-base -j "$(nproc)" > /dev/null
mkdir -p "$build_dir/tests"
g++ -std=c++17 -O2 -march=native -fopenmp \
    -I external/ggml/include \
    tests/fp8/test_f8_e4m3_b128.cpp \
    "$build_dir/ggml/src/libggml-cpu.a" "$build_dir/ggml/src/libggml-base.a" \
    -lpthread -lm -o "$build_dir/tests/test_f8_e4m3_b128"
"$build_dir/tests/test_f8_e4m3_b128"
