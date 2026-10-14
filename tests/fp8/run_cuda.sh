#!/usr/bin/env bash
# Build ggml with CUDA (shared, standalone) and run the F8_E4M3_B128 CUDA test.
# Pass "bench" to also run the MUL_MAT_ID microbenchmark (F16 vs F8 experts).
# Run from the repository root on the Linux host, inside the dev image:
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/src -w /src drama3-lowmem-dev tests/fp8/run_cuda.sh [bench]
set -euo pipefail
build_dir=${BUILD_DIR:-build/ggml-cuda-tests}
cmake -S external/ggml -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-120a}" -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF > /dev/null
cmake --build "$build_dir" --target ggml ggml-cuda -j "$(nproc)" > /dev/null
mkdir -p "$build_dir/fp8"
for name in test_f8_e4m3_b128_cuda bench_mul_mat_id_cuda; do
    g++ -std=c++17 -O2 -I external/ggml/include "tests/fp8/$name.cpp" \
        -L "$build_dir/bin" -L "$build_dir/src" -L "$build_dir/src/ggml-cuda" \
        -lggml -lggml-base -lggml-cuda -Wl,-rpath,"$PWD/$build_dir/bin:$PWD/$build_dir/src:$PWD/$build_dir/src/ggml-cuda" \
        -o "$build_dir/fp8/$name"
done
"$build_dir/fp8/test_f8_e4m3_b128_cuda"
if [[ "${1:-}" == "bench" ]]; then
    for shape in up down; do
        for tokens in 1 4 8 32 128; do
            for type in f16 f8; do
                "$build_dir/fp8/bench_mul_mat_id_cuda" "$type" "$shape" "$tokens"
            done
        done
    done
fi
