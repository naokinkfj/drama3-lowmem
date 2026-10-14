#include "engine/community_models/fish_drama3/sampler_graph.h"

#include <algorithm>

namespace engine::models::fish_drama3 {

ggml_tensor * build_drama3_sampler(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * uniform,
                                   ggml_tensor * first_only, float temperature, int top_k, float top_p) {
    const int64_t codes = logits->ne[0];
    auto * column = ggml_reshape_2d(ctx, logits, codes, 1);
    // The CPU rule accepts any top_k; ranks past the codebook do not exist.
    const int64_t kept_ranks = std::min<int64_t>(top_k, codes);
    if (temperature <= 0.0F || kept_ranks == 1) {
        return ggml_cast(ctx, ggml_argmax(ctx, column), GGML_TYPE_F32);
    }
    auto * probabilities = ggml_soft_max(ctx, ggml_scale(ctx, column, 1.0F / temperature));
    auto * order = ggml_argsort(ctx, probabilities, GGML_SORT_ORDER_DESC);
    // Probabilities in descending order: gather rows of a one-column view.
    auto * sorted = ggml_get_rows(ctx, ggml_reshape_2d(ctx, probabilities, 1, codes), ggml_reshape_1d(ctx, order, codes));
    auto * top = ggml_view_1d(ctx, ggml_reshape_1d(ctx, sorted, codes), kept_ranks, 0);
    // Drop a rank once the cumulative probability exceeds top_p (equality is
    // kept, as on the CPU), but never rank 0.
    auto * beyond = ggml_step(ctx, ggml_scale_bias(ctx, ggml_cumsum(ctx, top), 1.0F, -top_p));
    auto * first = ggml_view_1d(ctx, first_only, kept_ranks, 0);
    auto * keep = ggml_clamp(ctx, ggml_add(ctx, ggml_scale_bias(ctx, beyond, -1.0F, 1.0F), first), 0.0F, 1.0F);
    auto * kept = ggml_mul(ctx, top, keep);
    // The drawn rank is the number of kept ranks whose CDF lies below
    // uniform * total. The total is the CDF's own last entry, so for any
    // uniform < 1 the threshold stays below it and the rank stays in range;
    // the clamp guards against anything else.
    auto * cdf = ggml_cumsum(ctx, kept);
    auto * total = ggml_view_1d(ctx, cdf, 1, static_cast<size_t>(kept_ranks - 1) * ggml_element_size(cdf));
    auto * threshold = ggml_mul(ctx, total, uniform);
    auto * below = ggml_step(ctx, ggml_sub(ctx, ggml_repeat(ctx, threshold, cdf), cdf));
    auto * rank = ggml_clamp(ctx, ggml_sum(ctx, below), 0.0F, static_cast<float>(kept_ranks - 1));
    auto * ranked_codes = ggml_cast(ctx, ggml_view_1d(ctx, order, kept_ranks, 0), GGML_TYPE_F32);
    auto * code = ggml_get_rows(ctx, ggml_reshape_2d(ctx, ranked_codes, 1, kept_ranks), ggml_cast(ctx, rank, GGML_TYPE_I32));
    return ggml_reshape_1d(ctx, code, 1);
}

}  // namespace engine::models::fish_drama3
