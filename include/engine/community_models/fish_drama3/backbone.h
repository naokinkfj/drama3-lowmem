#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::fish_drama3 {

struct Drama3BackboneDecoderConfig {
    int64_t vocab_size = 252288;
    int64_t hidden_size = 2048;
    int64_t intermediate_size = 512;
    int64_t experts = 256;
    int64_t experts_per_token = 8;
    int64_t layers = 40;
    int64_t heads = 16;
    int64_t kv_heads = 2;
    int64_t head_dim = 256;
    int64_t full_attention_interval = 4;
    int64_t linear_conv_kernel_dim = 4;
    int64_t linear_key_head_dim = 128;
    int64_t linear_num_key_heads = 16;
    int64_t linear_num_value_heads = 32;
    int64_t linear_value_head_dim = 128;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 10000000.0F;
    float partial_rotary_factor = 0.25F;
    std::string weight_prefix = "model.language_model";
    std::string lm_head_prefix = "lm_head";
    bool tie_word_embeddings = false;
    bool round_bf16_activations = false;
    int64_t decode_cache_steps = 2048;
    int64_t cpu_expert_layers = 0;
    // When set, only these LM head rows (eos, then lm_head_count rows from
    // lm_head_first) are kept on the device, as F32.
    int32_t lm_head_eos = -1;
    int32_t lm_head_first = -1;
    int32_t lm_head_count = 0;
    size_t prefill_cache_bytes = 256u * 1024u * 1024u;
    bool profile_prefill_cache = false;
};

struct Drama3BackboneDecoderForwardResult {
    std::vector<float> hidden;
    int64_t steps = 0;
};

class Drama3BackboneDecoderRuntime {
public:
    Drama3BackboneDecoderRuntime(
        std::shared_ptr<const engine::assets::TensorSource> weights,
        Drama3BackboneDecoderConfig config,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~Drama3BackboneDecoderRuntime();

    Drama3BackboneDecoderRuntime(const Drama3BackboneDecoderRuntime &) = delete;
    Drama3BackboneDecoderRuntime & operator=(const Drama3BackboneDecoderRuntime &) = delete;

    std::vector<float> token_embedding(const std::vector<int32_t> & token_ids);
    Drama3BackboneDecoderForwardResult forward_embeddings(const std::vector<float> & embeddings, int64_t steps);
    // LM head logits for [eos, first, first + 1, ..., first + count - 1] only.
    std::vector<float> candidate_logits(const std::vector<float> & hidden, int32_t eos, int32_t first, int32_t count);
    void release_graphs();

    class DecodeSession {
    public:
        virtual ~DecodeSession() = default;
        virtual void reset() = 0;
        virtual Drama3BackboneDecoderForwardResult prefill_embeddings(
            const std::vector<float> & embeddings,
            int64_t steps) = 0;
        virtual std::vector<float> run_embedding_step(const std::vector<float> & embedding) = 0;
        virtual int64_t valid_steps() const noexcept = 0;
    };

    std::unique_ptr<DecodeSession> create_decode_session(int64_t cache_steps);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_drama3
