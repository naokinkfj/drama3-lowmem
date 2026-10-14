#include "engine/community_models/fish_drama3/text_chunks.h"

#include <regex>

namespace engine::models::fish_drama3 {
namespace {

const std::regex leading_speaker(R"(^\s*(<\|speaker:[0-9]+\|>))");
const std::regex any_speaker(R"(<\|speaker:[0-9]+\|>)");

std::vector<std::string> speaker_turns(const std::string & text) {
    std::vector<std::string> turns;
    size_t begin = 0;
    while (begin < text.size()) {
        size_t next = text.find("<|speaker:", begin + 1);
        if (next == std::string::npos) next = text.size();
        turns.push_back(text.substr(begin, next - begin));
        begin = next;
    }
    return turns;
}

bool blank(const std::string & text) {
    return text.find_first_not_of(" \t\r\n") == std::string::npos;
}

}  // namespace

std::vector<std::string> drama3_text_chunks(const std::string & text, int64_t chunk_size,
                                            engine::text::TextChunkMode mode, bool split_turns) {
    std::vector<std::string> raw;
    for (const auto & turn : split_turns ? speaker_turns(text) : std::vector<std::string>{text}) {
        if (blank(turn)) continue;
        for (auto & chunk : engine::text::split_text_chunks(turn, chunk_size, mode)) raw.push_back(std::move(chunk));
    }
    // Carry the speaker of the turn a chunk belongs to: the last tag seen so far.
    std::vector<std::string> chunks;
    std::string speaker;
    for (const auto & chunk : raw) {
        std::smatch lead;
        const bool tagged = std::regex_search(chunk, lead, leading_speaker);
        const std::string in_effect = speaker;  // the speaker at this chunk's start
        for (std::sregex_iterator it(chunk.begin(), chunk.end(), any_speaker), end; it != end; ++it) speaker = it->str();
        if (blank(std::regex_replace(chunk, any_speaker, ""))) continue;  // a lone tag carries over to the next chunk
        chunks.push_back(tagged || in_effect.empty() ? chunk : in_effect + chunk);
    }
    if (chunks.empty() && !blank(text)) chunks.push_back(text);
    return chunks;
}

}  // namespace engine::models::fish_drama3
