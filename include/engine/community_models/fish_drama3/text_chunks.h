#pragma once

#include "engine/framework/text/chunking.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::fish_drama3 {

// Splits request text into the chunks Drama 3 generates one after another.
// With split_turns every <|speaker:N|> turn starts a new chunk. Long turns are
// split further at sentence and clause breaks (chunk_size codepoints), and
// every chunk carries the speaker tag of the turn it came from, so a speaker
// never falls back to the prompt's default speaker 0 mid-turn.
std::vector<std::string> drama3_text_chunks(const std::string & text, int64_t chunk_size,
                                            engine::text::TextChunkMode mode, bool split_turns);

}  // namespace engine::models::fish_drama3
