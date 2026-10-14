// The tower samples its nine residual codebooks inside one ggml graph. This
// checks that graph against the CPU sampler's rule for the same uniform draw:
// the code must be identical unless the draw sits on a boundary between two
// codes (within float rounding).
#include "engine/community_models/fish_drama3/sampler_graph.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using engine::models::fish_drama3::build_drama3_sampler;

namespace {

struct Reference {
    int code;
    double margin;  // distance of the draw from the nearest code boundary, in probability
};

// Mirrors sample() in generator.cpp, but inverts the CDF at a given uniform.
Reference reference_draw(const std::vector<float> & logits, float temperature, int top_k, float top_p, float uniform) {
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int l, int r) { return logits[l] > logits[r]; });
    if (temperature <= 0 || top_k == 1) return {order.front(), 1.0};
    const double maximum = logits[order.front()] / temperature;
    std::vector<double> probabilities(order.size());
    double total = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        probabilities[rank] = std::exp(logits[order[rank]] / temperature - maximum);
        total += probabilities[rank];
    }
    double cumulative = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        cumulative += probabilities[rank] / total;
        if (rank > 0 && (cumulative > top_p || rank >= static_cast<size_t>(top_k))) probabilities[rank] = 0;
    }
    const double kept = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    const double threshold = uniform * kept;
    double cdf = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        const double before = cdf;
        cdf += probabilities[rank];
        if (cdf > threshold) {
            const double margin = std::min(threshold - before, cdf - threshold) / kept;
            return {order[rank], margin};
        }
    }
    return {order.front(), 0.0};
}

}  // namespace

int main() {
    const int codebook = 4096, top_k = 30;
    const float temperature = 1.0F, top_p = 0.9F;
    std::mt19937 rng(7);
    std::normal_distribution<float> spread(0.0F, 3.0F);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    int checked = 0, mismatched = 0, skipped = 0;
    for (int trial = 0; trial < 400; ++trial) {
        ggml_init_params params{64ull * 1024 * 1024, nullptr, false};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, codebook, 1);
        ggml_tensor * uniform = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_tensor * first_only = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, top_k);
        std::vector<float> values(codebook);
        // Peaked distributions like the model's: a few large logits on top of noise.
        for (auto & value : values) value = spread(rng);
        for (int i = 0; i < 5; ++i) values[rng() % codebook] += 8.0F + 4.0F * unit(rng);
        std::copy(values.begin(), values.end(), static_cast<float *>(logits->data));
        const float draw = unit(rng);
        *static_cast<float *>(uniform->data) = draw;
        std::fill_n(static_cast<float *>(first_only->data), top_k, 0.0F);
        static_cast<float *>(first_only->data)[0] = 1.0F;

        ggml_tensor * code = build_drama3_sampler(ctx, logits, uniform, first_only, temperature, top_k, top_p);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, code);
        ggml_graph_compute_with_ctx(ctx, graph, 4);
        const int got = static_cast<int>(*static_cast<float *>(code->data));
        const Reference want = reference_draw(values, temperature, top_k, top_p, draw);
        ggml_free(ctx);
        if (want.margin < 1e-5) {
            ++skipped;
            continue;
        }
        ++checked;
        if (got != want.code) {
            ++mismatched;
            std::fprintf(stderr, "trial %d: graph drew %d, reference %d (u=%.6f)\n", trial, got, want.code, draw);
        }
    }
    std::printf("test_sampler_graph: %d checked, %d on boundaries skipped, %d mismatched\n", checked, skipped, mismatched);
    return mismatched == 0 && checked > 300 ? 0 : 1;
}
