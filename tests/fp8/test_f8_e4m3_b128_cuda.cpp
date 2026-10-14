// CUDA tests for GGML_TYPE_F8_E4M3_B128.
//
// What this protects:
//   - CUDA dequantization equals the CPU dequantize_row bit for bit for every
//     one of the 256 codes (0x7F/0xFF included) and a spread of block scales
//   - the mul_mat vector kernel (<= 8 columns), the dequantize + cuBLAS path
//     (> 8 columns) and both MUL_MAT_ID kernels (per token, per expert) read
//     every weight exactly: with one-hot activations each output is a single
//     dequantized weight, so any rounding of the weights (FP16/TF32 GEMM) or a
//     mis-addressed block, row or expert shows up
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            ++failures;                                   \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);            \
            std::fprintf(stderr, "\n");                   \
        }                                                 \
    } while (0)

constexpr int kBlock = 128;
constexpr size_t kBlockBytes = sizeof(float) + kBlock;

static uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Weights for `rows` rows of `k` values. The first two blocks hold all 256 codes,
// the rest are random codes; scales include powers of two and arbitrary values.
static std::vector<uint8_t> make_weights(int64_t k, int64_t rows, std::mt19937 & rng) {
    const int64_t blocks = k / kBlock * rows;
    std::vector<uint8_t> data(blocks * kBlockBytes);
    std::uniform_real_distribution<float> scale_distribution(1e-6f, 1e-2f);
    for (int64_t b = 0; b < blocks; ++b) {
        uint8_t * block = data.data() + b * kBlockBytes;
        const float scale = b % 5 == 0 ? 1.0f : b % 5 == 1 ? std::ldexp(1.0f, -12 - int(b % 9)) : scale_distribution(rng);
        std::memcpy(block, &scale, sizeof(float));
        for (int i = 0; i < kBlock; ++i) {
            block[sizeof(float) + i] = b < 2 ? uint8_t(b * kBlock + i) : uint8_t(rng() & 0xFF);
        }
    }
    return data;
}

static std::vector<float> dequantize_cpu(const std::vector<uint8_t> & data, int64_t n) {
    std::vector<float> out(n);
    ggml_get_type_traits(GGML_TYPE_F8_E4M3_B128)->to_float(data.data(), out.data(), n);
    return out;
}

// Runs one op on the CUDA backend. build() creates the inputs and returns the
// output; fill() uploads the inputs once they are allocated.
template <typename Build, typename Fill>
static std::vector<float> run_cuda(ggml_backend_t backend, Build build, Fill fill) {
    ggml_init_params params = {64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * out = build(ctx);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    fill();
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static void test_get_rows_is_bit_exact(ggml_backend_t backend, std::mt19937 & rng) {
    const int64_t k = 512, rows = 12;
    const std::vector<uint8_t> data = make_weights(k, rows, rng);
    const std::vector<float> want = dequantize_cpu(data, k * rows);

    ggml_tensor * weights = nullptr;
    ggml_tensor * index = nullptr;
    const std::vector<float> got = run_cuda(backend,
        [&](ggml_context * ctx) {
            weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3_B128, k, rows);
            index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
            return ggml_get_rows(ctx, weights, index);
        },
        [&]() {
            ggml_backend_tensor_set(weights, data.data(), 0, data.size());
            std::vector<int32_t> ids(rows);
            for (int64_t r = 0; r < rows; ++r) {
                ids[r] = int32_t(r);
            }
            ggml_backend_tensor_set(index, ids.data(), 0, ids.size() * sizeof(int32_t));
        });

    int mismatches = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (float_bits(got[i]) != float_bits(want[i]) && mismatches++ < 8) {
            CHECK(false, "get_rows element %zu (code 0x%02x): %.9g != %.9g", i,
                data[i / kBlock * kBlockBytes + sizeof(float) + i % kBlock], got[i], want[i]);
        }
    }
    CHECK(mismatches == 0, "get_rows: %d mismatching elements", mismatches);
}

// out[r, c] = W[r, hot[c]] for one-hot activation columns.
static int hot_position(int64_t c, int64_t k) {
    return int((c * 389 + 7) % k);
}

static void test_mul_mat_reads_weights_exactly(ggml_backend_t backend, std::mt19937 & rng) {
    const int64_t k = 2048, rows = 40;
    const std::vector<uint8_t> data = make_weights(k, rows, rng);
    const std::vector<float> w = dequantize_cpu(data, k * rows);

    for (int64_t ncols : {1, 2, 3, 5, 8, 9, 16, 40}) {
        ggml_tensor * weights = nullptr;
        ggml_tensor * x = nullptr;
        const std::vector<float> got = run_cuda(backend,
            [&](ggml_context * ctx) {
                weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3_B128, k, rows);
                x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols);
                return ggml_mul_mat(ctx, weights, x);
            },
            [&]() {
                ggml_backend_tensor_set(weights, data.data(), 0, data.size());
                std::vector<float> xs(k * ncols, 0.0f);
                for (int64_t c = 0; c < ncols; ++c) {
                    xs[c * k + hot_position(c, k)] = 1.0f;
                }
                ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));
            });

        int mismatches = 0;
        for (int64_t c = 0; c < ncols; ++c) {
            for (int64_t r = 0; r < rows; ++r) {
                const float want = w[r * k + hot_position(c, k)];
                // A sum of +0 terms turns -0 into +0, so compare values, not bits.
                if (got[c * rows + r] != want && mismatches++ < 4) {
                    CHECK(false, "mul_mat ncols=%lld row %lld col %lld: %.9g != %.9g",
                        (long long) ncols, (long long) r, (long long) c, got[c * rows + r], want);
                }
            }
        }
        CHECK(mismatches == 0, "mul_mat ncols=%lld: %d mismatching outputs", (long long) ncols, mismatches);
    }
}

static void test_mul_mat_id_reads_weights_exactly(ggml_backend_t backend, std::mt19937 & rng) {
    const int64_t k = 512, rows = 64, experts = 16, used = 4;
    const std::vector<uint8_t> data = make_weights(k, rows * experts, rng);
    const std::vector<float> w = dequantize_cpu(data, k * rows * experts);

    // Up to 8 tokens run per token. Larger batches run per expert, 8 rows at a
    // time: about 3 rows per expert for 12 tokens, about 10 (a full and a partial
    // group) for 40 tokens.
    for (int64_t tokens : {1, 3, 8, 12, 40}) {
        std::vector<int32_t> ids(used * tokens);
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t u = 0; u < used; ++u) {
                ids[t * used + u] = int32_t((t * 5 + u * 3) % experts);
            }
        }
        ggml_tensor * weights = nullptr;
        ggml_tensor * x = nullptr;
        ggml_tensor * index = nullptr;
        const std::vector<float> got = run_cuda(backend,
            [&](ggml_context * ctx) {
                weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F8_E4M3_B128, k, rows, experts);
                x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, used, tokens);
                index = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
                return ggml_mul_mat_id(ctx, weights, x, index);
            },
            [&]() {
                ggml_backend_tensor_set(weights, data.data(), 0, data.size());
                std::vector<float> xs(k * used * tokens, 0.0f);
                for (int64_t c = 0; c < used * tokens; ++c) {
                    xs[c * k + hot_position(c, k)] = 1.0f;
                }
                ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));
                ggml_backend_tensor_set(index, ids.data(), 0, ids.size() * sizeof(int32_t));
            });

        // dst is [rows, used, tokens].
        int mismatches = 0;
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t u = 0; u < used; ++u) {
                const int64_t c = t * used + u;
                for (int64_t r = 0; r < rows; ++r) {
                    const float want = w[(ids[c] * rows + r) * k + hot_position(c, k)];
                    if (got[c * rows + r] != want && mismatches++ < 4) {
                        CHECK(false, "mul_mat_id tokens=%lld token %lld slot %lld row %lld: %.9g != %.9g",
                            (long long) tokens, (long long) t, (long long) u, (long long) r, got[c * rows + r], want);
                    }
                }
            }
        }
        CHECK(mismatches == 0, "mul_mat_id tokens=%lld: %d mismatching outputs", (long long) tokens, mismatches);
    }
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    std::mt19937 rng(20261015);
    test_get_rows_is_bit_exact(backend, rng);
    test_mul_mat_reads_weights_exactly(backend, rng);
    test_mul_mat_id_reads_weights_exactly(backend, rng);
    ggml_backend_free(backend);
    if (failures == 0) {
        std::printf("test_f8_e4m3_b128_cuda: all passed\n");
        return 0;
    }
    std::printf("test_f8_e4m3_b128_cuda: %d failure(s)\n", failures);
    return 1;
}
