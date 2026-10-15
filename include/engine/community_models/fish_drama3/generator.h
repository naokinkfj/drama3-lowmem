// Modified for drama3-lowmem (2026); see NOTICE.
#pragma once

#include "engine/community_models/fish_drama3/prompt.h"

#include <functional>
#include <optional>

namespace engine::models::fish_drama3 {

struct Drama3Sampling {
    float temperature = 1.0F;
    float top_p = 0.9F;
    int top_k = 30;
    int64_t max_frames = 1024;
    uint32_t seed = 0;
    // drama3-lowmem: stop a take whose main token repeats this many frames in
    // a row (a stuck silence); 0 disables the check.
    int64_t stall_frames = 0;
};

struct Drama3Generation {
    codecs::FishDacCodes codes;
    runtime::AudioBuffer audio;
    bool reached_limit = false;
    bool stalled = false;
};

class Drama3Generator {
public:
    Drama3Generator(std::shared_ptr<const Drama3Assets> assets, core::ExecutionContext & execution,
                    const runtime::SessionOptions & options);
    ~Drama3Generator();
    codecs::FishDacCodes encode_reference(const runtime::AudioBuffer & audio);
    void warm_streaming_codec(const codecs::FishDacCodes & reference);
    Drama3Generation generate(const std::string & text, const std::vector<Drama3Take> & references,
        const std::vector<Drama3Take> & history, const Drama3Sampling & sampling, bool streaming,
        const std::function<void(const runtime::AudioBuffer &)> & on_audio = {});

private:
    std::shared_ptr<const Drama3Assets> assets_;
    fish_audio::FishAudioTextTokenizer tokenizer_;
    int64_t cache_steps_;
    bool profile_first_audio_ = false;
    bool offline_codec_stream_ = true;
    bool profile_steps_ = false;
    // drama3-lowmem numerical validation: when set, generate() writes every
    // step's backbone hidden state and raw logits to this directory, and
    // teacher_frames_path_ (optional) forces the generated frames.
    std::string teacher_dump_dir_;
    std::string teacher_frames_path_;
    std::unique_ptr<Drama3BackboneDecoderRuntime> backbone_;
    std::unique_ptr<Drama3Tower> tower_;
    std::unique_ptr<core::ExecutionContext> codec_execution_;
    std::unique_ptr<codecs::FishDacCodecRuntime> codec_;
    // Reference encoding runs once per reference; on a device backend it runs
    // on the CPU, because the encode graph needs ~2 GiB of VRAM for 25 s.
    std::unique_ptr<core::ExecutionContext> encoder_execution_;
    std::unique_ptr<codecs::FishDacCodecRuntime> encoder_;
    std::optional<runtime::AudioBuffer> cached_reference_audio_;
    codecs::FishDacCodes cached_reference_codes_;
};

}  // namespace engine::models::fish_drama3
