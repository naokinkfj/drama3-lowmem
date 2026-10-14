// Chunking for Drama 3: long speaker turns are split into sentence-sized
// chunks, and every chunk must keep its turn's speaker tag, because the prompt
// gives untagged text to speaker 0.
#include "engine/community_models/fish_drama3/text_chunks.h"

#include <cstdio>
#include <string>
#include <vector>

using engine::models::fish_drama3::drama3_text_chunks;
using engine::text::TextChunkMode;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL line %d: ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

static bool starts_with(const std::string & text, const std::string & prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

static std::string body(const std::string & chunk) {
    std::string out = chunk;
    while (starts_with(out, "<|speaker:")) out.erase(0, out.find("|>") + 2);
    const auto first = out.find_first_not_of(" ");
    return first == std::string::npos ? "" : out.substr(first);
}

int main() {
    const std::string long_turn =
        "今日は、家庭で動かす人工知能の話をします。最近の音声合成モデルはとても大きく、そのままでは一般的なパソコンのメモリに収まりません。"
        "そこで、よく使う部分だけをグラフィックボードに置き、残りをメインメモリに置いて計算する工夫が生まれました。";

    // A long turn by speaker 1 keeps the tag on every chunk.
    auto chunks = drama3_text_chunks("<|speaker:1|>" + long_turn, 60, TextChunkMode::TagAware, true);
    CHECK(chunks.size() >= 3, "expected the long turn to split, got %zu chunk(s)", chunks.size());
    for (const auto & chunk : chunks) {
        CHECK(starts_with(chunk, "<|speaker:1|>"), "chunk lost its speaker tag: %s", chunk.c_str());
        CHECK(!body(chunk).empty(), "tag-only chunk: %s", chunk.c_str());
    }

    // A short turn followed by a long one: the long one's chunks keep speaker 1.
    chunks = drama3_text_chunks("<|speaker:0|>ねえ、聞いてる？<|speaker:1|>" + long_turn, 60, TextChunkMode::TagAware, true);
    CHECK(!chunks.empty() && starts_with(chunks.front(), "<|speaker:0|>"), "first chunk should be speaker 0");
    for (size_t i = 1; i < chunks.size(); ++i) {
        CHECK(starts_with(chunks[i], "<|speaker:1|>"), "chunk %zu should be speaker 1: %s", i, chunks[i].c_str());
    }

    // Without turn splitting, chunks inside the second turn still carry its tag.
    chunks = drama3_text_chunks("<|speaker:0|>ねえ、聞いてる？<|speaker:1|>" + long_turn, 60, TextChunkMode::TagAware, false);
    bool seen_one = false;
    for (const auto & chunk : chunks) {
        CHECK(starts_with(chunk, "<|speaker:"), "untagged chunk in tagged text: %s", chunk.c_str());
        CHECK(!body(chunk).empty(), "tag-only chunk: %s", chunk.c_str());
        if (starts_with(chunk, "<|speaker:1|>")) seen_one = true;
        else CHECK(!seen_one, "speaker went back to 0 inside speaker 1's turn: %s", chunk.c_str());
    }

    // Untagged text stays untagged; the prompt supplies speaker 0.
    chunks = drama3_text_chunks(long_turn, 60, TextChunkMode::TagAware, true);
    CHECK(chunks.size() >= 3, "expected the paragraph to split");
    for (const auto & chunk : chunks) CHECK(!starts_with(chunk, "<|"), "untagged text gained a tag: %s", chunk.c_str());

    std::printf("test_text_chunks: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
