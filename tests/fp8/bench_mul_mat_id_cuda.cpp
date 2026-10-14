// MUL_MAT_ID microbenchmark for Drama 3 expert shapes on CUDA: 256 experts,
// top-8, F16 vs F8_E4M3_B128 weights.
//
// One graph holds `nodes` MUL_MAT_ID ops over the same expert bank. For a single
// token, node g uses experts 8g..8g+7, so 32 nodes walk the whole bank once and
// successive evaluations cannot be served from L2. Reported bandwidth counts the
// distinct expert weights each node reads.
//
// usage: bench_mul_mat_id_cuda <f16|f8> <up|down> [tokens=1] [nodes=32] [iters=50]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <f16|f8> <up|down> [tokens=1] [nodes=32] [iters=50]\n", argv[0]);
        return 1;
    }
    const ggml_type type = std::string(argv[1]) == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F8_E4M3_B128;
    const bool up = std::string(argv[2]) == "up";
    const int tokens = argc > 3 ? std::atoi(argv[3]) : 1;
    const int nodes = argc > 4 ? std::atoi(argv[4]) : 32;
    const int iters = argc > 5 ? std::atoi(argv[5]) : 50;

    const int64_t n_expert = 256, n_used = 8;
    const int64_t k = up ? 2048 : 512;
    const int64_t n = up ? 512 : 2048;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }

    ggml_init_params params = {(4 + 3 * size_t(nodes)) * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * bank = ggml_new_tensor_3d(ctx, type, k, n, n_expert);
    std::vector<ggml_tensor *> inputs, ids, outs;
    for (int g = 0; g < nodes; ++g) {
        // gate/up read the normed hidden state once per token, down reads one row per expert slot.
        inputs.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, up ? 1 : n_used, tokens));
        ids.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, tokens));
        outs.push_back(ggml_mul_mat_id(ctx, bank, inputs.back(), ids.back()));
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(1);
    {
        std::vector<uint8_t> data(ggml_nbytes(bank));
        if (type == GGML_TYPE_F16) {
            std::normal_distribution<float> dist(0.0f, 0.02f);
            ggml_fp16_t * h = reinterpret_cast<ggml_fp16_t *>(data.data());
            for (int64_t i = 0; i < ggml_nelements(bank); ++i) {
                h[i] = ggml_fp32_to_fp16(dist(rng));
            }
        } else {
            const size_t block_bytes = ggml_type_size(type);
            for (size_t b = 0; b < data.size() / block_bytes; ++b) {
                uint8_t * block = data.data() + b * block_bytes;
                const float scale = 1e-4f;
                std::memcpy(block, &scale, sizeof(float));
                for (size_t i = sizeof(float); i < block_bytes; ++i) {
                    uint8_t code = uint8_t(rng());
                    block[i] = (code & 0x7F) == 0x7F ? 0 : code;
                }
            }
        }
        ggml_backend_tensor_set(bank, data.data(), 0, data.size());
    }

    std::normal_distribution<float> act(0.0f, 1.0f);
    double distinct_bytes = 0.0;
    const double expert_bytes = double(ggml_nbytes(bank)) / n_expert;
    for (int g = 0; g < nodes; ++g) {
        std::vector<float> x(ggml_nelements(inputs[g]));
        for (float & v : x) {
            v = act(rng);
        }
        ggml_backend_tensor_set(inputs[g], x.data(), 0, x.size() * sizeof(float));

        std::vector<int32_t> sel(n_used * tokens);
        std::set<int32_t> distinct;
        for (int t = 0; t < tokens; ++t) {
            std::set<int32_t> chosen;
            for (int u = 0; u < n_used; ++u) {
                int32_t e = tokens == 1 ? int32_t((g * n_used + u) % n_expert) : int32_t(rng() % n_expert);
                while (chosen.count(e)) {
                    e = int32_t((e + 1) % n_expert);
                }
                chosen.insert(e);
                distinct.insert(e);
                sel[t * n_used + u] = e;
            }
        }
        distinct_bytes += distinct.size() * expert_bytes;
        ggml_backend_tensor_set(ids[g], sel.data(), 0, sel.size() * sizeof(int32_t));
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (ggml_tensor * out : outs) {
        ggml_build_forward_expand(graph, out);
    }
    for (int i = 0; i < 5; ++i) {
        ggml_backend_graph_compute(backend, graph);
    }

    // The GPU may be shared, so report the best of several trials.
    double seconds = 1e30;
    for (int trial = 0; trial < 10; ++trial) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(backend, graph);
        }
        seconds = std::min(seconds, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }

    const double per_node_us = seconds / (double(iters) * nodes) * 1e6;
    const double gbps = distinct_bytes * iters / seconds / 1e9;
    std::printf("%-12s %-4s tokens=%-4d nodes=%d: %8.2f us/op, %7.1f MB distinct weights/op, %7.1f GB/s\n",
        ggml_type_name(type), up ? "up" : "down", tokens, nodes, per_node_us, distinct_bytes / nodes / 1e6, gbps);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
