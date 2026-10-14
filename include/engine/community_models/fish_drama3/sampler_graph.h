#pragma once

#include "ggml.h"

namespace engine::models::fish_drama3 {

// Builds a ggml subgraph that draws one code from `logits` (a column of
// codebook_size values) with the same rule as the CPU sampler: scale by
// 1/temperature, keep the top_k most likely codes, keep the most likely
// prefix whose cumulative probability (over all codes) stays within top_p but
// never drop the first, then invert the CDF of the kept codes at `uniform`
// (one F32 value in [0, 1)). `first_only` is an F32 input of top_k values
// [1, 0, 0, ...]. Returns the drawn code as a one-element F32 tensor.
// temperature <= 0 or top_k == 1 picks the most likely code.
ggml_tensor * build_drama3_sampler(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * uniform,
                                   ggml_tensor * first_only, float temperature, int top_k, float top_p);

}  // namespace engine::models::fish_drama3
