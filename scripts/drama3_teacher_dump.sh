#!/usr/bin/env bash
# Dump per-step hidden states and raw logits for each case, one process per
# case, for comparison against tools/reference (see docs/PLAN.md Phase 4).
#
#   scripts/drama3_teacher_dump.sh <model.gguf> <cpu_expert_layers> <output-dir> <cases.json> [frames-from-dir]
#
# Without frames-from-dir the model samples its own frames (seed 123 or the
# case's options.seed) and records them in <output-dir>/<case>/meta.json.
# With frames-from-dir every case is forced onto <frames-from-dir>/<case>/meta.json's
# frames, so two packages can be compared position by position.
set -euo pipefail

model=$1
cpu_expert_layers=$2
output=$(realpath -m "$3")
cases=$(realpath "$4")
frames_from=${5:+$(realpath "$5")}
source_dir=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$output"

for name in $(python3 -c 'import json,sys; print(" ".join(c["name"] for c in json.load(open(sys.argv[1]))))' "$cases"); do
    case_dir="$output/$name"
    mkdir -p "$case_dir"
    # One generate() call per case: chunking by speaker turn or length would
    # overwrite the dump once per chunk, so both are turned off here.
    python3 -c 'import json,sys; json.dump([dict(c, options=dict(c.get("options", {}), split_speaker_turns="false", text_chunk_size="100000")) for c in json.load(open(sys.argv[1])) if c["name"] == sys.argv[2]], open(sys.argv[3], "w"), ensure_ascii=False)' \
        "$cases" "$name" "$case_dir/case.json"
    extra=(--session-option "fish_drama3.teacher_dump=$case_dir")
    if [[ -n "$frames_from" ]]; then
        extra+=(--session-option "fish_drama3.teacher_frames=$frames_from/$name/meta.json")
    fi
    "$source_dir/scripts/drama3_run_cases.sh" "$model" "$cpu_expert_layers" "$case_dir/run" "$case_dir/case.json" "${extra[@]}" > /dev/null
    echo "$name: $(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(len(m["prompt_rows"]), "prompt rows,", len(m["frames"]), "frames")' "$case_dir/meta.json")"
done
