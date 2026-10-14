# Toolchain image for building and running drama3-lowmem on a Linux CUDA host
# without installing the CUDA toolkit or CMake on the host.
#
#   docker build -f docker/dev.Dockerfile -t drama3-lowmem-dev docker/
#   docker run --rm --gpus all -v "$PWD":/src -w /src drama3-lowmem-dev \
#     scripts/build_linux.sh --backend cuda --cuda-arch 120 --build-type Release \
#       --model-set custom --models fish_drama3
#
# The source tree is bind-mounted, so build outputs land in ./build on the host.
ARG CUDA_VERSION=12.9.1
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake ninja-build git ca-certificates libgomp1 ffmpeg python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
