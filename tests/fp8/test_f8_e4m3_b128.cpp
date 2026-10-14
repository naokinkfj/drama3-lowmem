// Tests for GGML_TYPE_F8_E4M3_B128: blocks of 128 FP8 E4M3 values sharing one
// FP32 scale, used to keep Drama 3's block-scaled FP8 experts as shipped.
//
// What this protects:
//   - every E4M3 byte decodes to its exact value (no rounding anywhere)
//   - dequantization equals fp32(e4m3) * d bit for bit, matching how the
//     checkpoint's reference values are formed
//   - the CPU dot product and mul_mat_id agree with an F32 computation
//   - quantizing values that are already representable gives the same bytes back
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
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

// Independent E4M3 decoder: bias 7, no infinities, 0x7F/0xFF are NaN.
static double e4m3_reference(uint8_t bits) {
    const int sign = bits >> 7;
    const int exponent = (bits >> 3) & 0xF;
    const int mantissa = bits & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return sign ? -magnitude : magnitude;
}

static bool is_nan_code(uint8_t bits) {
    return (bits & 0x7F) == 0x7F;
}

static uint8_t random_code(std::mt19937 & rng) {
    for (;;) {
        const uint8_t bits = static_cast<uint8_t>(rng() & 0xFF);
        if (!is_nan_code(bits)) {
            return bits;
        }
    }
}

static void write_block(uint8_t * block, float scale, const uint8_t * codes) {
    std::memcpy(block, &scale, sizeof(float));
    std::memcpy(block + sizeof(float), codes, kBlock);
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void test_traits() {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3_B128);
    CHECK(std::strcmp(traits->type_name, "f8_e4m3_b128") == 0, "type name is %s", traits->type_name);
    CHECK(traits->blck_size == kBlock, "block size %lld", (long long) traits->blck_size);
    CHECK(traits->type_size == kBlockBytes, "type size %zu", traits->type_size);
    CHECK(ggml_row_size(GGML_TYPE_F8_E4M3_B128, 2048) == 16 * kBlockBytes, "row size");
}

static void test_every_code_decodes_exactly() {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3_B128);
    std::vector<uint8_t> block(kBlockBytes);
    std::vector<float> out(kBlock);
    for (int first = 0; first < 256; first += kBlock) {
        uint8_t codes[kBlock];
        for (int i = 0; i < kBlock; ++i) {
            const uint8_t bits = static_cast<uint8_t>(first + i);
            codes[i] = is_nan_code(bits) ? 0 : bits;
        }
        write_block(block.data(), 1.0f, codes);
        traits->to_float(block.data(), out.data(), kBlock);
        for (int i = 0; i < kBlock; ++i) {
            const double want = e4m3_reference(codes[i]);
            CHECK(static_cast<double>(out[i]) == want, "code 0x%02x decoded to %.9g, want %.9g", codes[i], out[i], want);
        }
    }
}

static void test_dequantize_matches_scaled_reference(std::mt19937 & rng) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3_B128);
    std::uniform_real_distribution<float> scale_distribution(1e-6f, 1e-2f);
    const int blocks = 64;
    std::vector<uint8_t> data(blocks * kBlockBytes);
    std::vector<uint8_t> codes(blocks * kBlock);
    std::vector<float> scales(blocks);
    for (int b = 0; b < blocks; ++b) {
        scales[b] = b % 4 == 0 ? std::ldexp(1.0f, -10 - b % 7) : scale_distribution(rng);
        for (int i = 0; i < kBlock; ++i) {
            codes[b * kBlock + i] = random_code(rng);
        }
        write_block(data.data() + b * kBlockBytes, scales[b], codes.data() + b * kBlock);
    }
    std::vector<float> out(blocks * kBlock);
    traits->to_float(data.data(), out.data(), blocks * kBlock);
    for (int b = 0; b < blocks; ++b) {
        for (int i = 0; i < kBlock; ++i) {
            const float want = static_cast<float>(e4m3_reference(codes[b * kBlock + i])) * scales[b];
            const float got = out[b * kBlock + i];
            CHECK(float_bits(got) == float_bits(want), "block %d index %d: %.9g != %.9g", b, i, got, want);
        }
    }
}

static void test_quantize_round_trips_representable_values(std::mt19937 & rng) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3_B128);
    // A block whose largest magnitude is 448 * d quantizes with the same d, so
    // values that are exactly e4m3 * d must come back as the same codes.
    const float scale = 0.0078125f;
    uint8_t codes[kBlock];
    for (int i = 0; i < kBlock; ++i) {
        codes[i] = random_code(rng);
    }
    codes[0] = 0x7E;  // +448, the largest finite E4M3 value
    std::vector<float> values(kBlock);
    for (int i = 0; i < kBlock; ++i) {
        values[i] = static_cast<float>(e4m3_reference(codes[i])) * scale;
    }
    std::vector<uint8_t> block(kBlockBytes);
    traits->from_float_ref(values.data(), block.data(), kBlock);
    float stored_scale;
    std::memcpy(&stored_scale, block.data(), sizeof(float));
    CHECK(stored_scale == scale, "scale %.9g != %.9g", stored_scale, scale);
    for (int i = 0; i < kBlock; ++i) {
        const uint8_t got = block[sizeof(float) + i];
        // +0 and -0 are both exact encodings of zero.
        const bool same = got == codes[i] || (e4m3_reference(got) == 0.0 && e4m3_reference(codes[i]) == 0.0);
        CHECK(same, "index %d: code 0x%02x became 0x%02x", i, codes[i], got);
    }
}

static void fill_random_weights(ggml_tensor * tensor, std::mt19937 & rng, std::vector<float> & dequantized) {
    std::uniform_real_distribution<float> scale_distribution(1e-4f, 4e-3f);
    const int64_t blocks = ggml_nelements(tensor) / kBlock;
    auto * data = static_cast<uint8_t *>(tensor->data);
    dequantized.resize(ggml_nelements(tensor));
    for (int64_t b = 0; b < blocks; ++b) {
        const float scale = scale_distribution(rng);
        uint8_t codes[kBlock];
        for (int i = 0; i < kBlock; ++i) {
            codes[i] = random_code(rng);
            dequantized[b * kBlock + i] = static_cast<float>(e4m3_reference(codes[i])) * scale;
        }
        write_block(data + b * kBlockBytes, scale, codes);
    }
}

static void test_vec_dot(std::mt19937 & rng) {
    const auto * cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_F8_E4M3_B128);
    CHECK(cpu_traits->vec_dot_type == GGML_TYPE_F32, "vec_dot_type should keep F32 activations");
    std::normal_distribution<float> activation(0.0f, 1.0f);
    for (int k : {128, 512, 2048, 2048 + 128}) {
        const int blocks = k / kBlock;
        std::vector<uint8_t> weights(blocks * kBlockBytes);
        std::vector<double> dequantized(k);
        std::uniform_real_distribution<float> scale_distribution(1e-4f, 4e-3f);
        for (int b = 0; b < blocks; ++b) {
            const float scale = scale_distribution(rng);
            uint8_t codes[kBlock];
            for (int i = 0; i < kBlock; ++i) {
                codes[i] = random_code(rng);
                dequantized[b * kBlock + i] = static_cast<double>(static_cast<float>(e4m3_reference(codes[i])) * scale);
            }
            write_block(weights.data() + b * kBlockBytes, scale, codes);
        }
        std::vector<float> x(k);
        for (auto & value : x) {
            value = activation(rng);
        }
        double want = 0.0;
        double magnitude = 0.0;
        for (int i = 0; i < k; ++i) {
            want += dequantized[i] * x[i];
            magnitude += std::fabs(dequantized[i] * x[i]);
        }
        float got = 0.0f;
        cpu_traits->vec_dot(k, &got, 0, weights.data(), 0, x.data(), 0, 1);
        CHECK(std::fabs(got - want) <= 1e-5 * magnitude + 1e-30, "k=%d: %.9g vs %.9g (|sum| %.3g)", k, got, want, magnitude);
    }
}

static void test_mul_mat_id(std::mt19937 & rng) {
    // Shapes of one Drama 3 expert projection: K=2048 in, 512 out, 8 experts here.
    const int k = 2048, n = 512, experts = 8, used = 3, tokens = 2;
    ggml_init_params params = {512ull * 1024 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F8_E4M3_B128, k, n, experts);
    ggml_tensor * weights_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n, experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, used, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);

    std::vector<float> dequantized;
    fill_random_weights(weights, rng, dequantized);
    std::memcpy(weights_f32->data, dequantized.data(), dequantized.size() * sizeof(float));
    std::normal_distribution<float> activation(0.0f, 1.0f);
    for (int64_t i = 0; i < ggml_nelements(input); ++i) {
        static_cast<float *>(input->data)[i] = activation(rng);
    }
    const int32_t chosen[tokens][used] = {{5, 0, 3}, {7, 5, 1}};
    std::memcpy(ids->data, chosen, sizeof(chosen));

    ggml_tensor * out = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_tensor * reference = ggml_mul_mat_id(ctx, weights_f32, input, ids);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, reference);
    ggml_graph_compute_with_ctx(ctx, graph, 8);

    // Outputs near zero come from cancellation, so compare the largest error
    // with the typical output size rather than element by element.
    double worst = 0.0, squares = 0.0;
    for (int64_t i = 0; i < ggml_nelements(out); ++i) {
        const double got = static_cast<float *>(out->data)[i];
        const double want = static_cast<float *>(reference->data)[i];
        worst = std::fmax(worst, std::fabs(got - want));
        squares += want * want;
    }
    const double rms = std::sqrt(squares / ggml_nelements(out));
    CHECK(worst <= 1e-5 * rms, "mul_mat_id max error %.3g vs output rms %.3g", worst, rms);
    ggml_free(ctx);
}

int main() {
    std::mt19937 rng(20261015);
    test_traits();
    test_every_code_decodes_exactly();
    test_dequantize_matches_scaled_reference(rng);
    test_quantize_round_trips_representable_values(rng);
    test_vec_dot(rng);
    test_mul_mat_id(rng);
    if (failures == 0) {
        std::printf("test_f8_e4m3_b128: all passed\n");
        return 0;
    }
    std::printf("test_f8_e4m3_b128: %d failure(s)\n", failures);
    return 1;
}
