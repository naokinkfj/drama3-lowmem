#include "engine/community_models/fish_drama3/sampler_graph.h"

namespace engine::models::fish_drama3 {

ggml_tensor * build_drama3_sampler(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * uniform,
                                   ggml_tensor * first_only, float temperature, int top_k, float top_p) {
    const int64_t codes = logits->ne[0];
    auto * column = ggml_reshape_2d(ctx, logits, codes, 1);
    if (temperature <= 0.0F || top_k == 1) {
        return ggml_cast(ctx, ggml_argmax(ctx, column), GGML_TYPE_F32);
    }
    auto * probabilities = ggml_soft_max(ctx, ggml_scale(ctx, column, 1.0F / temperature));
    auto * order = ggml_argsort(ctx, probabilities, GGML_SORT_ORDER_DESC);
    // Probabilities in descending order: gather rows of a one-column view.
    auto * sorted = ggml_get_rows(ctx, ggml_reshape_2d(ctx, probabilities, 1, codes), ggml_reshape_1d(ctx, order, codes));
    auto * top = ggml_view_1d(ctx, ggml_reshape_1d(ctx, sorted, codes), top_k, 0);
    // Keep a rank while the cumulative probability stays within top_p, and always rank 0.
    auto * within = ggml_step(ctx, ggml_scale_bias(ctx, ggml_cumsum(ctx, top), -1.0F, top_p));
    auto * kept = ggml_mul(ctx, top, ggml_clamp(ctx, ggml_add(ctx, within, first_only), 0.0F, 1.0F));
    // The drawn rank is the number of kept ranks whose CDF lies below uniform * total.
    auto * cdf = ggml_cumsum(ctx, kept);
    auto * threshold = ggml_mul(ctx, ggml_sum(ctx, kept), uniform);
    auto * rank = ggml_sum(ctx, ggml_step(ctx, ggml_sub(ctx, ggml_repeat(ctx, threshold, cdf), cdf)));
    auto * ranked_codes = ggml_cast(ctx, ggml_view_1d(ctx, order, top_k, 0), GGML_TYPE_F32);
    auto * code = ggml_get_rows(ctx, ggml_reshape_2d(ctx, ranked_codes, 1, top_k), ggml_cast(ctx, rank, GGML_TYPE_I32));
    return ggml_reshape_1d(ctx, code, 1);
}

}  // namespace engine::models::fish_drama3
