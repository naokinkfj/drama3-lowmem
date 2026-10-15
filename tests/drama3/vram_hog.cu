// Holds GPU memory so that only the requested amount stays free, to emulate a
// GPU with less memory. Runs until it is stopped (SIGTERM / SIGINT).
//
//   vram_hog <GiB to leave free>
//
// A headless card leaves about (its nvidia-smi total - 0.48 GiB) free to a new
// process: 23.5 GiB for a 24 GB card, 15.44 for 16 GB, 11.5 for 12 GB.
#include <cuda_runtime.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

static volatile std::sig_atomic_t stop = 0;

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <GiB to leave free>\n", argv[0]);
        return 2;
    }
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const size_t leave = static_cast<size_t>(std::atof(argv[1]) * gib);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemGetInfo failed\n");
        return 1;
    }
    if (free_bytes <= leave) {
        std::fprintf(stderr, "only %.2f GiB free, nothing to hold\n", free_bytes / gib);
        return 1;
    }
    void * held = nullptr;
    const size_t hold = free_bytes - leave;
    if (cudaMalloc(&held, hold) != cudaSuccess || cudaMemset(held, 0, hold) != cudaSuccess) {
        std::fprintf(stderr, "could not hold %.2f GiB\n", hold / gib);
        return 1;
    }
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::printf("holding %.2f GiB; %.2f GiB of %.2f GiB free\n", hold / gib, free_bytes / gib, total_bytes / gib);
    std::fflush(stdout);
    std::signal(SIGTERM, [](int) { stop = 1; });
    std::signal(SIGINT, [](int) { stop = 1; });
    while (!stop) pause();
    cudaFree(held);
    return 0;
}
