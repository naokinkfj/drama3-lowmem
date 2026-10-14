// Times one Drama 3 MoE layer's routed experts on the CPU backend:
// gate+up (2048 -> 512) and down (512 -> 2048) for 8 of 256 experts, one token.
#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

int main(int argc, char ** argv) {
    const ggml_type type = argc > 1 && std::strcmp(argv[1], "f16") == 0 ? GGML_TYPE_F16 : GGML_TYPE_F8_E4M3_B128;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 16;
    const int layers = 12, experts = 256, used = 8;
    ggml_init_params params{(size_t) 12ull * 1024 * 1024 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    std::mt19937 rng(1);
    std::vector<ggml_tensor *> up(layers), down(layers);
    for (int l = 0; l < layers; ++l) {
        up[l] = ggml_new_tensor_3d(ctx, type, 2048, 1024, experts);  // gate and up stacked
        down[l] = ggml_new_tensor_3d(ctx, type, 512, 2048, experts);
        for (ggml_tensor * t : {up[l], down[l]}) {
            auto * bytes = static_cast<uint8_t *>(t->data);
            for (size_t i = 0; i < ggml_nbytes(t); ++i) bytes[i] = static_cast<uint8_t>(rng() & 0x77);
        }
    }
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, 1);
    for (int i = 0; i < 2048; ++i) static_cast<float *>(x->data)[i] = 0.01f * (i % 7);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    for (int l = 0; l < layers; ++l) {
        ggml_tensor * h = ggml_mul_mat_id(ctx, up[l], x, ids);                      // [1024, 8, 1]
        ggml_tensor * g = ggml_view_3d(ctx, h, 512, used, 1, h->nb[1], h->nb[2], 0);
        ggml_tensor * d = ggml_mul_mat_id(ctx, down[l], ggml_cont(ctx, g), ids);    // [2048, 8, 1]
        ggml_build_forward_expand(graph, d);
    }
    double best = 1e9;
    for (int rep = 0; rep < 30; ++rep) {
        for (int i = 0; i < used; ++i) static_cast<int32_t *>(ids->data)[i] = static_cast<int32_t>(rng() % experts);
        const auto start = std::chrono::steady_clock::now();
        ggml_graph_compute_with_ctx(ctx, graph, threads);
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    const double bytes = layers * used * (ggml_row_size(type, 2048) * 1024.0 + ggml_row_size(type, 512) * 2048.0);
    std::printf("%s threads=%d: %.2f ms for %d layers (%.2f ms/layer), %.1f GB/s\n",
                ggml_type_name(type), threads, best, layers, best / layers, bytes / best / 1e6);
    return 0;
}
