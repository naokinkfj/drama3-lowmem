#!/usr/bin/env bash
# Build and run the Drama 3 graph unit tests against the ggml CPU backend.
# Run from the repository root inside the dev image (see tests/fp8/run.sh).
set -euo pipefail
build_dir=${BUILD_DIR:-build/linux-cuda-release}
cmake --build "$build_dir" --target ggml-cpu ggml-base -j "$(nproc)" > /dev/null
mkdir -p "$build_dir/tests"
g++ -std=c++17 -O2 -march=native -fopenmp \
    -I external/ggml/include -I include \
    tests/drama3/test_sampler_graph.cpp src/community_models/fish_drama3/sampler_graph.cpp \
    "$build_dir/ggml/src/libggml-cpu.a" "$build_dir/ggml/src/libggml-base.a" \
    -lpthread -lm -o "$build_dir/tests/test_sampler_graph"
"$build_dir/tests/test_sampler_graph"
