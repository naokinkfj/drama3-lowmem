#include "engine/community_models/fish_drama3/generator.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace engine::models::fish_drama3 {
namespace {

int32_t sample(const std::vector<float> & logits, float temperature, int top_k,
               float top_p, std::mt19937 & random) {
    if (logits.empty()) throw std::runtime_error("Drama 3 sampler has no logits");
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t left, int32_t right) { return logits[left] > logits[right]; });
    if (temperature == 0 || top_k == 1) return order.front();
    const double maximum = logits[order.front()] / temperature;
    std::vector<double> probabilities(order.size());
    double total = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        probabilities[rank] = std::exp(logits[order[rank]] / temperature - maximum);
        total += probabilities[rank];
    }
    if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("Drama 3 sampler encountered nonfinite logits");
    double cumulative = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        cumulative += probabilities[rank] / total;
        if (rank > 0 && (cumulative > top_p || rank >= static_cast<size_t>(top_k))) probabilities[rank] = 0;
    }
    std::discrete_distribution<size_t> choice(probabilities.begin(), probabilities.end());
    return order[choice(random)];
}

void write_f32(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("failed to write " + path.string());
}

std::string frames_json(const std::vector<Drama3Frame> & frames) {
    std::ostringstream out;
    out << '[';
    for (size_t row = 0; row < frames.size(); ++row) {
        out << (row ? ",[" : "[");
        for (size_t i = 0; i < frames[row].size(); ++i) out << (i ? "," : "") << frames[row][i];
        out << ']';
    }
    out << ']';
    return out.str();
}

std::vector<Drama3Frame> read_frames(const std::filesystem::path & path) {
    const auto root = io::json::parse_file(path);
    const auto * frames = root.is_array() ? &root : root.find("frames");
    if (frames == nullptr) throw std::runtime_error("teacher frames json needs a frames array: " + path.string());
    std::vector<Drama3Frame> out;
    for (const auto & row : frames->as_array()) {
        const auto & values = row.as_array();
        if (values.size() != 11) throw std::runtime_error("teacher frames need 11 values per row");
        Drama3Frame frame{};
        for (size_t i = 0; i < values.size(); ++i) frame[i] = static_cast<int32_t>(values[i].as_i64());
        out.push_back(frame);
    }
    return out;
}

std::vector<float> last_hidden(const std::vector<float> & values, int64_t width) {
    if (values.size() < static_cast<size_t>(width)) throw std::runtime_error("Drama 3 backbone returned too few hidden values");
    return {values.end() - width, values.end()};
}

}  // namespace

Drama3Generator::Drama3Generator(std::shared_ptr<const Drama3Assets> assets,
    core::ExecutionContext & execution, const runtime::SessionOptions & options)
    : assets_(std::move(assets)), tokenizer_(assets_->tokenizer_assets),
      cache_steps_(runtime::parse_positive_i64_option(options.options, {"fish_drama3.context_length"}, 2048)) {
    if (cache_steps_ > 262144) throw std::runtime_error("Drama 3 context_length exceeds the checkpoint's maximum position count");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.profile_first_audio"})) {
        profile_first_audio_ = runtime::parse_bool_option(*value, "fish_drama3.profile_first_audio");
    }
    auto backbone_config = assets_->backbone;
    backbone_config.decode_cache_steps = cache_steps_;
    backbone_config.cpu_expert_layers = runtime::parse_i64_option(options.options, {"fish_drama3.cpu_expert_layers"}).value_or(4);
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.profile_steps"})) {
        profile_steps_ = runtime::parse_bool_option(*value, "fish_drama3.profile_steps");
    }
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.teacher_dump"})) teacher_dump_dir_ = *value;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.teacher_frames"})) teacher_frames_path_ = *value;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.offline_codec"})) {
        if (*value == "full") offline_codec_stream_ = false;
        else if (*value != "stream") throw std::runtime_error("fish_drama3.offline_codec requires stream or full");
    }
    const auto backbone_prefill_cache_mb = runtime::parse_i64_option(options.options, {"fish_drama3.backbone_prefill_cache_mb"}).value_or(256);
    if (backbone_prefill_cache_mb < 0 || backbone_prefill_cache_mb > 4096) throw std::runtime_error("Drama 3 backbone_prefill_cache_mb must be between 0 and 4096");
    backbone_config.prefill_cache_bytes = static_cast<size_t>(backbone_prefill_cache_mb) * 1024u * 1024u;
    backbone_config.profile_prefill_cache = profile_first_audio_;
    backbone_ = std::make_unique<Drama3BackboneDecoderRuntime>(assets_->model_weights, backbone_config, execution,
        64 * 1024 * 1024, 8 * 1024 * 1024, assets::TensorStorageType::Native);
    const auto tower_prefill_cache_mb = runtime::parse_i64_option(options.options, {"fish_drama3.tower_prefill_cache_mb"}).value_or(256);
    if (tower_prefill_cache_mb < 0 || tower_prefill_cache_mb > 4096) throw std::runtime_error("Drama 3 tower_prefill_cache_mb must be between 0 and 4096");
    tower_ = std::make_unique<Drama3Tower>(assets_->model_weights, assets_->tower, execution, cache_steps_,
        static_cast<size_t>(tower_prefill_cache_mb) * 1024u * 1024u);
    auto codec_backend = options.backend;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_backend"})) {
        if (*value == "cpu") codec_backend.type = core::BackendType::Cpu;
        else if (*value != "same") throw std::runtime_error("fish_drama3.codec_backend requires cpu or same");
    }
    codec_execution_ = std::make_unique<core::ExecutionContext>(codec_backend);
    codecs::FishDacCodecRuntimeOptions codec_options;
    codec_options.weight_context_bytes = 16 * 1024 * 1024;
    const auto component = codecs::FishDacCodecComponent::load_from_tensor_source(assets_->codec_weights,
        assets_->codec, {}, codec_execution_->backend(), codec_backend.type, codec_options);
    codec_ = std::make_unique<codecs::FishDacCodecRuntime>(component, codec_backend,
        codec_backend.threads, codec_options.graph_arena_bytes);
    assets_->model_weights->release_storage();
    assets_->codec_weights->release_storage();
}

Drama3Generator::~Drama3Generator() = default;

codecs::FishDacCodes Drama3Generator::encode_reference(const runtime::AudioBuffer & audio) {
    if (cached_reference_audio_ && cached_reference_audio_->sample_rate == audio.sample_rate &&
        cached_reference_audio_->channels == audio.channels && cached_reference_audio_->samples == audio.samples) {
        return cached_reference_codes_;
    }
    auto result = codec_->encode_codes(audio);
    codec_->release_encode_graph();
    validate_drama3_codes(result, assets_->tower.codebook_size);
    // Keep one short reference in host memory. Exact waveform equality avoids
    // stale path-based hits when an uploaded file is replaced or re-decoded.
    constexpr size_t max_cached_samples = 44100 * 30;
    if (audio.samples.size() <= max_cached_samples) {
        cached_reference_audio_ = audio;
        cached_reference_codes_ = result;
    } else {
        cached_reference_audio_.reset();
        cached_reference_codes_ = {};
    }
    return result;
}

void Drama3Generator::warm_streaming_codec(const codecs::FishDacCodes & reference) {
    validate_drama3_codes(reference, assets_->tower.codebook_size);
    codecs::FishDacCodes frame;
    frame.codebooks = reference.codebooks;
    frame.frames = 1;
    for (int64_t book = 0; book < reference.codebooks; ++book) {
        frame.codes.push_back(reference.codes[static_cast<size_t>(book * reference.frames)]);
    }
    // Initialize and capture the first-frame codec graph during preparation.
    // Discard decoded reference audio and reset all causal stream state.
    for (int pass = 0; pass < 4; ++pass) {
        codec_->reset_decode_stream();
        (void)codec_->decode_stream(frame);
    }
    codec_->reset_decode_stream();
}

Drama3Generation Drama3Generator::generate(const std::string & text, const std::vector<Drama3Take> & references,
    const std::vector<Drama3Take> & history, const Drama3Sampling & sampling, bool streaming,
    const std::function<void(const runtime::AudioBuffer &)> & on_audio) {
    if (!std::isfinite(sampling.temperature) || sampling.temperature < 0 ||
        !std::isfinite(sampling.top_p) || sampling.top_p <= 0 || sampling.top_p > 1 ||
        sampling.top_k <= 0 || sampling.max_frames <= 0) throw std::runtime_error("invalid Drama 3 sampling parameters");
    using ProfileClock = std::chrono::steady_clock;
    const auto profile_start = profile_first_audio_ ? ProfileClock::now() : ProfileClock::time_point{};
    auto profile_marked = profile_start;
    std::vector<std::pair<const char *, double>> first_audio_phases;
    const auto mark_phase = [&](const char * name) {
        if (!profile_first_audio_) return;
        const auto now = ProfileClock::now();
        first_audio_phases.emplace_back(name, std::chrono::duration<double, std::milli>(now - profile_marked).count());
        profile_marked = now;
    };
    const auto prompt = build_drama3_prompt(*assets_, tokenizer_, text, references, history);
    if (sampling.max_frames > cache_steps_ ||
        prompt.size() > static_cast<size_t>(cache_steps_ - sampling.max_frames)) {
        throw std::runtime_error("Drama 3 prompt plus max_tokens exceeds fish_drama3.context_length; increase it or shorten history");
    }
    auto decoder = backbone_->create_decode_session(cache_steps_);
    decoder->reset();
    tower_->reset();
    mark_phase("setup_ms");
    std::vector<int32_t> ids;
    for (const auto & frame : prompt) ids.push_back(frame[0]);
    auto embeddings = backbone_->token_embedding(ids);
    tower_->mix_backbone_embeddings(embeddings, prompt);
    mark_phase("embedding_ms");
    const auto prefill = decoder->prefill_embeddings(embeddings, static_cast<int64_t>(prompt.size()));
    mark_phase("backbone_prefill_ms");
    tower_->prefill(prefill.hidden, prompt);
    mark_phase("tower_prefill_ms");
    auto hidden = last_hidden(prefill.hidden, assets_->backbone.hidden_size);
    Drama3Frame current = prompt.back();
    std::mt19937 random(sampling.seed);
    const bool dumping = !teacher_dump_dir_.empty();
    const auto teacher = teacher_frames_path_.empty() ? std::vector<Drama3Frame>{} : read_frames(teacher_frames_path_);
    std::vector<float> dump_hidden, dump_semantic, dump_residual;
    std::deque<int32_t> recent;
    std::vector<Drama3Frame> output_frames;
    Drama3Generation result;
    if (streaming) codec_->reset_decode_stream();
    // Per-frame time split for tuning (fish_drama3.profile_steps).
    double lm_head_ms = 0, sample_ms = 0, tower_ms = 0, codec_ms = 0, backbone_ms = 0;
    auto lap = ProfileClock::now();
    const auto split = [&](double & bucket) {
        if (!profile_steps_) return;
        const auto now = ProfileClock::now();
        bucket += std::chrono::duration<double, std::milli>(now - lap).count();
        lap = now;
    };
    for (int64_t step = 0; step < sampling.max_frames; ++step) {
        if (!teacher.empty() && step >= static_cast<int64_t>(teacher.size())) break;
        split(backbone_ms);
        const auto candidates = backbone_->candidate_logits(
            hidden, assets_->eos, assets_->tower.semantic_begin, static_cast<int32_t>(assets_->tower.codebook_size));
        split(lm_head_ms);
        if (dumping) {
            dump_hidden.insert(dump_hidden.end(), hidden.begin(), hidden.end());
            dump_semantic.insert(dump_semantic.end(), candidates.begin(), candidates.end());
        }
        const int32_t first = sample(candidates, sampling.temperature, sampling.top_k, sampling.top_p, random);
        const int32_t second = sample(candidates, 1.0F, 30, 0.95F, random);
        const int32_t first_token = first == 0 ? assets_->eos : assets_->tower.semantic_begin + first - 1;
        const bool repeats = first != 0 && std::find(recent.begin(), recent.end(), first_token) != recent.end();
        int32_t selected = repeats ? second : first;
        if (!teacher.empty()) selected = teacher[static_cast<size_t>(step)][0] - assets_->tower.semantic_begin + 1;
        split(sample_ms);
        if (selected == 0) break;
        if (step == 0) mark_phase("semantic_ms");
        Drama3Frame next{};
        next[0] = assets_->tower.semantic_begin + selected - 1;
        next[1] = selected - 1;
        const auto shared_input = tower_->shared_input({current});
        auto tower_hidden = hidden;
        const int64_t position = decoder->valid_steps() - 1;
        const auto & config = assets_->tower;
        for (int64_t layer = 0; layer < config.shared_layers + config.residual_layers; ++layer) {
            const int64_t book = layer < config.shared_layers ? 0 : layer - config.shared_layers;
            const auto tower_output = tower_->run_layer(layer, tower_hidden, {next[1 + book]}, {1.0F},
                                                        layer == 0 ? shared_input : std::vector<float>{}, position);
            tower_hidden = tower_output.hidden;
            if (layer >= config.shared_layers) {
                next[2 + book] = sample(tower_output.logits, sampling.temperature, sampling.top_k, sampling.top_p, random);
                if (!teacher.empty()) next[2 + book] = teacher[static_cast<size_t>(step)][2 + book];
                if (dumping) dump_residual.insert(dump_residual.end(), tower_output.logits.begin(), tower_output.logits.end());
            }
        }
        output_frames.push_back(next);
        split(tower_ms);
        if (step == 0) mark_phase("tower_first_frame_ms");
        recent.push_back(next[0]);
        if (recent.size() > 16) recent.pop_front();
        if (streaming) {
            codecs::FishDacCodes codes;
            codes.codebooks = 10;
            codes.frames = 1;
            codes.codes.assign(next.begin() + 1, next.end());
            auto chunk = codec_->decode_stream(codes);
            if (step == 0) mark_phase("codec_first_frame_ms");
            runtime::append_audio_buffer(result.audio, chunk);
            if (on_audio) on_audio(chunk);
            if (step == 0 && profile_first_audio_) {
                std::cerr << "[drama3.first_audio] prompt_rows=" << prompt.size();
                for (const auto & phase : first_audio_phases) std::cerr << " " << phase.first << "=" << phase.second;
                std::cerr << " total_ms=" << std::chrono::duration<double, std::milli>(ProfileClock::now() - profile_start).count() << '\n';
            }
        }
        split(codec_ms);
        if (step + 1 == sampling.max_frames) { result.reached_limit = true; break; }
        current = next;
        auto embedding = backbone_->token_embedding({current[0]});
        tower_->mix_backbone_embeddings(embedding, {current});
        hidden = decoder->run_embedding_step(embedding);
    }
    if (output_frames.empty()) throw std::runtime_error("Drama 3 ended before producing an audio frame");
    if (profile_steps_) {
        const double frames = static_cast<double>(output_frames.size());
        std::cerr << "[drama3.profile_steps] frames=" << output_frames.size()
                  << " backbone_ms=" << backbone_ms / frames << " lm_head_ms=" << lm_head_ms / frames
                  << " sample_ms=" << sample_ms / frames << " tower_ms=" << tower_ms / frames
                  << " codec_ms=" << codec_ms / frames << " (per frame)\n";
    }
    if (dumping) {
        // Teacher-forced numerical dump for tools/reference: per step hidden,
        // semantic candidates [eos, codes...] and residual codebook logits, plus
        // one prefill over prompt + all frames to check prefill against decode.
        const std::filesystem::path dir(teacher_dump_dir_);
        std::filesystem::create_directories(dir);
        write_f32(dir / "hidden_decode.f32", dump_hidden);
        write_f32(dir / "semantic.f32", dump_semantic);
        write_f32(dir / "residual.f32", dump_residual);
        std::vector<Drama3Frame> all_rows = prompt;
        all_rows.insert(all_rows.end(), output_frames.begin(), output_frames.end());
        std::vector<int32_t> all_ids;
        for (const auto & frame : all_rows) all_ids.push_back(frame[0]);
        auto all_embeddings = backbone_->token_embedding(all_ids);
        tower_->mix_backbone_embeddings(all_embeddings, all_rows);
        auto check = backbone_->create_decode_session(cache_steps_);
        check->reset();
        write_f32(dir / "hidden_prefill.f32", check->prefill_embeddings(all_embeddings, static_cast<int64_t>(all_rows.size())).hidden);
        std::ofstream meta(dir / "meta.json");
        meta << "{\"hidden_size\": " << assets_->backbone.hidden_size
             << ", \"codebook_size\": " << assets_->tower.codebook_size
             << ", \"num_codebooks\": 10, \"eos\": " << assets_->eos
             << ", \"semantic_begin\": " << assets_->tower.semantic_begin
             << ", \"shared_layers\": " << assets_->tower.shared_layers
             << ", \"residual_layers\": " << assets_->tower.residual_layers
             << ", \"prompt_rows\": " << frames_json(prompt)
             << ", \"frames\": " << frames_json(output_frames) << "}\n";
    }
    result.codes.codebooks = 10;
    result.codes.frames = static_cast<int64_t>(output_frames.size());
    result.codes.codes.resize(output_frames.size() * 10);
    for (int64_t book = 0; book < 10; ++book) {
        for (size_t row = 0; row < output_frames.size(); ++row) result.codes.codes[book * output_frames.size() + row] = output_frames[row][book + 1];
    }
    const auto codec_started = ProfileClock::now();
    if (!streaming && offline_codec_stream_) {
        // drama3-lowmem: decode through the causal streaming graph one frame at
        // a time. Its memory does not grow with the take, whereas the
        // whole-take graph needed about 8 GiB of VRAM for a 30 s take.
        codec_->reset_decode_stream();
        for (const auto & frame : output_frames) {
            codecs::FishDacCodes codes;
            codes.codebooks = 10;
            codes.frames = 1;
            codes.codes.assign(frame.begin() + 1, frame.end());
            runtime::append_audio_buffer(result.audio, codec_->decode_stream(codes));
        }
        codec_->reset_decode_stream();
    } else if (!streaming) {
        result.audio = codec_->decode_codes(result.codes);
        codec_->release_runtime_graphs();
    }
    if (profile_steps_ && !streaming) {
        std::cerr << "[drama3.profile_steps] offline_codec_ms_per_frame="
                  << std::chrono::duration<double, std::milli>(ProfileClock::now() - codec_started).count() /
                         static_cast<double>(output_frames.size()) << '\n';
    }
    if (result.reached_limit) std::cerr << "[fish_drama3] max_tokens reached; audio may be incomplete\n";
    return result;
}

}  // namespace engine::models::fish_drama3
