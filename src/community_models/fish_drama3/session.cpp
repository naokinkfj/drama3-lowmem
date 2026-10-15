// Modified for drama3-lowmem (2026); see NOTICE.
#include "engine/community_models/fish_drama3/session.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/community_models/fish_drama3/text_chunks.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace engine::models::fish_drama3 {
namespace {
namespace json = engine::io::json;

codecs::FishDacCodes parse_codes(const json::Value & value, int64_t codebook_size) {
    codecs::FishDacCodes codes;
    const auto & books = value.as_array();
    codes.codebooks = static_cast<int64_t>(books.size());
    if (books.size() != 10) throw std::runtime_error("Drama 3 vq_codes needs 10 codebooks");
    codes.frames = static_cast<int64_t>(books.front().as_array().size());
    for (const auto & book : books) {
        if (static_cast<int64_t>(book.as_array().size()) != codes.frames) throw std::runtime_error("Drama 3 vq_codes rows must have equal length");
        for (const auto & code : book.as_array()) {
            const int64_t integer = code.as_i64();
            if (integer < 0 || integer >= codebook_size || code.as_number() != static_cast<double>(integer)) {
                throw std::runtime_error("Drama 3 vq_codes requires integers in the codebook range");
            }
            codes.codes.push_back(static_cast<int32_t>(integer));
        }
    }
    validate_drama3_codes(codes, codebook_size);
    return codes;
}

// Codepoints that are spoken: tags such as <|speaker:0|> or [laughs], white
// space and ASCII punctuation are left out.
int64_t spoken_codepoints(const std::string & text) {
    int64_t count = 0;
    for (size_t i = 0; i < text.size();) {
        if (text.compare(i, 2, "<|") == 0) {
            const size_t end = text.find("|>", i + 2);
            if (end != std::string::npos) { i = end + 2; continue; }
        }
        if (text[i] == '[') {
            const size_t end = text.find(']', i + 1);
            if (end != std::string::npos) { i = end + 1; continue; }
        }
        const auto byte = static_cast<unsigned char>(text[i]);
        const size_t width = byte < 0x80 ? 1 : byte < 0xE0 ? 2 : byte < 0xF0 ? 3 : 4;
        if (byte >= 0x80 || std::isalnum(byte)) ++count;
        i += width;
    }
    return count;
}

runtime::AudioBuffer read_reference(const std::string & path) {
    auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
    return {wav.sample_rate, wav.channels, std::move(wav.samples)};
}

json::Value take_json(const Drama3Take & take) {
    json::Value::Array books;
    for (int64_t book = 0; book < 10; ++book) {
        json::Value::Array frames;
        for (int64_t frame = 0; frame < take.codes.frames; ++frame) {
            frames.push_back(json::Value::make_number(take.codes.codes[book * take.codes.frames + frame]));
        }
        books.push_back(json::Value::make_array(std::move(frames)));
    }
    return json::Value::make_object({{"text", json::Value::make_string(take.text)},
                                    {"vq_codes", json::Value::make_array(std::move(books))}});
}

codecs::FishDacCodes join_codes(const std::vector<Drama3Take> & takes) {
    codecs::FishDacCodes merged;
    merged.codebooks = 10;
    for (const auto & take : takes) merged.frames += take.codes.frames;
    merged.codes.reserve(static_cast<size_t>(merged.frames * 10));
    for (int64_t book = 0; book < 10; ++book) {
        for (const auto & take : takes) {
            const auto begin = take.codes.codes.begin() + book * take.codes.frames;
            merged.codes.insert(merged.codes.end(), begin, begin + take.codes.frames);
        }
    }
    return merged;
}

}  // namespace

Drama3Session::Drama3Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
                             std::shared_ptr<const Drama3Assets> assets)
    : runtime::RuntimeSessionBase(options), task_(task), assets_(std::move(assets)) {
    if (!assets_ || (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning)) {
        throw std::runtime_error("Fish Drama 3 supports TTS and voice cloning");
    }
    generator_ = std::make_unique<Drama3Generator>(assets_, execution_context(), options);
}
Drama3Session::~Drama3Session() = default;
std::string Drama3Session::family() const { return "fish_drama3"; }
runtime::VoiceTaskKind Drama3Session::task_kind() const { return task_.task; }
runtime::RunMode Drama3Session::run_mode() const { return task_.mode; }

void Drama3Session::prepare(const runtime::SessionPreparationRequest & request) {
    defaults_ = {};
    defaults_.text_input = request.text;
    defaults_.voice = request.voice;
    defaults_.options = request.options;
    mark_prepared();
}

Drama3Take Drama3Session::conditioning_take(const json::Value & value) {
    Drama3Take take;
    take.text = json::require_string(value, "text");
    validate_drama3_text(take.text, "conditioning text");
    if (const auto * codes = value.find("vq_codes")) take.codes = parse_codes(*codes, assets_->tower.codebook_size);
    else if (const auto * audio = value.find("audio_path")) take.codes = generator_->encode_reference(read_reference(audio->as_string()));
    else if (const auto * audio = value.find("audio")) take.codes = generator_->encode_reference(read_reference(audio->as_string()));
    else throw std::runtime_error("Drama 3 conditioning requires vq_codes, audio_path, or audio");
    return take;
}

runtime::TaskResult Drama3Session::run(const runtime::TaskRequest & request) { return synthesize(request, false); }
runtime::TaskResult Drama3Session::generate_stream(const runtime::TaskRequest & request) { return synthesize(request, true); }

runtime::TaskResult Drama3Session::synthesize(const runtime::TaskRequest & request, bool streaming) {
    require_prepared("Fish Drama 3 generation");
    auto effective = defaults_;
    if (request.text_input) effective.text_input = request.text_input;
    if (request.voice) effective.voice = request.voice;
    for (const auto & option : request.options) effective.options[option.first] = option.second;
    effective.input_artifacts = request.input_artifacts;
    if (!effective.text_input) throw std::runtime_error("Drama 3 requires input text");
    Drama3Sampling sampling;
    sampling.temperature = runtime::parse_finite_float_option(effective.options, {"temperature"}).value_or(1.0F);
    sampling.top_p = runtime::parse_finite_float_option(effective.options, {"top_p"}).value_or(0.9F);
    sampling.top_k = runtime::parse_int_option(effective.options, {"top_k"}).value_or(30);
    sampling.max_frames = runtime::parse_positive_i64_option(effective.options, {"max_tokens", "max_new_tokens"}, 1024);
    sampling.seed = runtime::parse_u32_option(effective.options, {"seed"}).value_or(runtime::random_u32_seed());
    // drama3-lowmem quality guard: a take that gets stuck repeating one frame
    // (usually silence) is stopped and, offline, regenerated with another seed.
    sampling.stall_frames = runtime::parse_i64_option(effective.options, {"stall_frames"}).value_or(64);
    const int64_t retries = runtime::parse_i64_option(effective.options, {"stall_retries"}).value_or(2);
    // A take far shorter than its text (default under 0.02 s per spoken
    // character, a fifth of fast English speech) ended early; it is retried too.
    const float min_seconds_per_char = runtime::parse_finite_float_option(effective.options, {"min_seconds_per_char"}).value_or(0.02F);
    const bool split_turns = runtime::parse_bool_option(
        runtime::find_option(effective.options, {"split_speaker_turns"}).value_or("true"), "split_speaker_turns");
    std::vector<Drama3Take> references;
    if (effective.voice && effective.voice->speaker && effective.voice->speaker->cached_voice_id &&
        !effective.voice->speaker->audio) {
        throw std::runtime_error("Drama 3 requires reference audio or a reference descriptor with VQ codes");
    }
    if (effective.voice && effective.voice->speaker && effective.voice->speaker->audio) {
        const auto transcript = runtime::find_option(effective.options, {"reference_text"});
        if (!transcript) throw std::runtime_error("Drama 3 reference audio requires reference_text");
        references.push_back({*transcript, generator_->encode_reference(*effective.voice->speaker->audio)});
    }
    if (const auto ordered = runtime::find_option(effective.options, {"multi_reference_cond", "references"})) {
        const auto descriptors = json::parse(*ordered);
        for (const auto & value : descriptors.as_array()) references.push_back(conditioning_take(value));
    }
    if (task_.task == runtime::VoiceTaskKind::VoiceCloning && references.empty()) {
        throw std::runtime_error("Drama 3 cloning requires at least one reference");
    }
    if (const auto prepare = runtime::find_option(effective.options, {"encode_reference_only"});
        prepare && runtime::parse_bool_option(*prepare, "encode_reference_only")) {
        if (references.empty()) throw std::runtime_error("Reference preparation requires reference audio or codes");
        if (task_.mode == runtime::RunMode::Streaming) generator_->warm_streaming_codec(references.front().codes);
        runtime::TaskResult result;
        for (size_t index = 0; index < references.size(); ++index) {
            result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::AcousticTokens,
                "drama3_reference_" + std::to_string(index), json::stringify(take_json(references[index])),
                {{"family", "fish_drama3"}, {"schema", "drama3_reference_v1"}}));
        }
        return result;
    }
    std::vector<Drama3Take> history;
    for (const auto & artifact : effective.input_artifacts) {
        if (artifact.kind != runtime::ArtifactKind::AcousticTokens) continue;
        const auto artifact_family = artifact.meta.find("family");
        if (artifact_family != artifact.meta.end() && artifact_family->second != "fish_drama3") {
            throw std::runtime_error("Drama 3 continuation cannot consume acoustic tokens from another model family");
        }
        const std::string payload(reinterpret_cast<const char *>(artifact.payload.data()), artifact.payload.size());
        history.push_back(conditioning_take(json::parse(payload)));
    }
    const auto takes = runtime::find_option(effective.options, {"previous_takes"});
    const auto takes_file = runtime::find_option(effective.options, {"previous_takes_file"});
    if (takes && takes_file) throw std::runtime_error("choose previous_takes or previous_takes_file");
    if (takes || takes_file) {
        const auto descriptors = takes_file ? json::parse_file(*takes_file) : json::parse(*takes);
        const auto append_take = [&](const json::Value & value) {
            if (value.is_string()) {
                auto generated = generator_->generate(value.as_string(), references, history, sampling, false);
                history.push_back({value.as_string(), std::move(generated.codes)});
            } else history.push_back(conditioning_take(value));
        };
        if (descriptors.is_array()) {
            for (const auto & value : descriptors.as_array()) append_take(value);
        } else {
            append_take(descriptors);
        }
    }
    // drama3-lowmem: about one sentence per chunk (60 codepoints) instead of
    // 200. Over 8 seeds of a 182-character Japanese paragraph, 200 stalled 5
    // times and lost text twice even with retries, 100 skipped a sentence once,
    // and 60 completed 8/8 without a retry.
    const int64_t chunk_size = runtime::parse_positive_i64_option(effective.options, {"text_chunk_size"}, 60);
    const auto mode = engine::text::parse_text_chunk_mode_override(effective.options).value_or(engine::text::TextChunkMode::TagAware);
    std::vector<runtime::TaskRequest> chunks;
    for (auto & text : drama3_text_chunks(effective.text_input->text, chunk_size, mode, split_turns)) {
        auto chunk = effective;
        chunk.text_input->text = std::move(text);
        chunks.push_back(std::move(chunk));
    }
    runtime::TaskResult result;
    runtime::AudioBuffer merged;
    std::vector<Drama3Take> generated_takes;
    const auto started = std::chrono::steady_clock::now();
    bool limited = false;
    int64_t failed_chunks = 0;
    for (size_t index = 0; index < chunks.size(); ++index) {
        auto chunk_sampling = sampling;
        chunk_sampling.seed += static_cast<uint32_t>(index);
        auto context = history;
        if (!generated_takes.empty()) context.push_back(generated_takes.back());
        std::function<void(const runtime::AudioBuffer &)> callback;
        if (streaming && stream_event_sink()) {
            callback = [&](const runtime::AudioBuffer & audio) {
                runtime::StreamEvent event;
                event.audio_output = audio;
                stream_event_sink()(event);
            };
        }
        const auto & text = chunks[index].text_input->text;
        auto generated = generator_->generate(text, references, context, chunk_sampling, streaming, callback);
        const double min_seconds = min_seconds_per_char * static_cast<double>(spoken_codepoints(text));
        const auto failed = [&](const Drama3Generation & take) {
            const double seconds = take.audio.sample_rate > 0
                ? static_cast<double>(take.audio.samples.size()) / take.audio.sample_rate : 0.0;
            return take.stalled || (!take.reached_limit && seconds < min_seconds);
        };
        // Streamed audio has already left, so only offline takes are retried.
        int64_t attempt = 1;
        for (; failed(generated) && !streaming && attempt <= retries; ++attempt) {
            std::cerr << "[fish_drama3] take " << (generated.stalled ? "stalled" : "ended early")
                      << "; regenerating chunk " << index << " (attempt " << attempt << ")\n";
            chunk_sampling.seed += 7919U;
            // A retry that ends before its first frame throws; keep the take we have.
            try {
                generated = generator_->generate(text, references, context, chunk_sampling, streaming, callback);
            } catch (const std::exception & error) {
                std::cerr << "[fish_drama3] retry failed: " << error.what() << '\n';
            }
        }
        // The final state of every chunk, so a take that still stalls after
        // its retries is visible in the log and in the result.
        const char * outcome = "ok";
        if (failed(generated)) {
            ++failed_chunks;
            outcome = generated.stalled ? "stalled" : "ended early";
        }
        std::cerr << "[fish_drama3] chunk " << index << " final: " << outcome << " after " << attempt - 1 << " retries\n";
        runtime::append_audio_buffer(merged, generated.audio);
        limited = limited || generated.reached_limit;
        generated_takes.push_back({text, std::move(generated.codes)});
    }
    Drama3Take take{effective.text_input->text, join_codes(generated_takes)};
    result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::AcousticTokens,
        "drama3_take", json::stringify(take_json(take)), {{"family", "fish_drama3"}, {"schema", "drama3_take_v1"}}));
    result.audio_output = std::move(merged);
    result.custom_schema_output = runtime::CustomSchemaOutput{"drama3_generation_v1", json::Value::make_object({
        {"frames", json::Value::make_number(take.codes.frames)},
        {"reached_limit", json::Value::make_bool(limited)},
        {"failed_chunks", json::Value::make_number(failed_chunks)},
        {"wall_ms", json::Value::make_number(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count())}})};
    return result;
}

}  // namespace engine::models::fish_drama3
