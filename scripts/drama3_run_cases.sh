#!/usr/bin/env bash
# Generate a fixed set of Drama 3 cases with one model package and one expert
# placement, recording GPU/host memory once per second.
#
#   scripts/drama3_run_cases.sh <model.gguf> <cpu_expert_layers> <output-dir> <cases.json>
#
# cases.json: [{"name": "...", "text": "...", "language": "en"|"ja",
#               "voice_ref": optional wav, "reference_text": optional}]
# Runs audiocpp_cli inside the drama3-lowmem-dev image (see docker/dev.Dockerfile).
set -euo pipefail

model=$(realpath "$1")
cpu_expert_layers=$2
output=$(realpath -m "$3")
cases=$(realpath "$4")
source_dir=$(cd "$(dirname "$0")/.." && pwd)
threads=${THREADS:-16}
context_length=${CONTEXT_LENGTH:-4096}
max_tokens=${MAX_TOKENS:-1500}
mkdir -p "$output"

nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 1 > "$output/gpu-memory.csv" &
gpu_sampler=$!
(while true; do echo "$(date +%T),$(free -m | awk '/^Mem:/{print $3","$7} /^Swap:/{print $3}' | paste -sd,)"; sleep 1; done) > "$output/host-memory.csv" &
host_sampler=$!
trap 'kill $gpu_sampler $host_sampler 2>/dev/null || true' EXIT

python3 - "$cases" > "$output/requests.tsv" <<'EOF'
import json, sys
for case in json.load(open(sys.argv[1])):
    print("\t".join([case["name"], case["text"], case.get("voice_ref", ""), case.get("reference_text", "")]))
EOF

while IFS=$'\t' read -r name text voice_ref reference_text; do
    extra=()
    if [[ -n "$voice_ref" ]]; then
        extra+=(--voice-ref "$(realpath "$voice_ref")" --request-option "reference_text=$reference_text")
    fi
    echo "== $name" | tee -a "$output/run.log"
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$source_dir":"$source_dir" -v "$(dirname "$model")":"$(dirname "$model")" -v "$output":"$output" \
        drama3-lowmem-dev \
        "$source_dir/build/linux-cuda-release/bin/audiocpp_cli" \
        --task tts --family fish_drama3 --backend cuda --mode offline \
        --model "$model" --text "$text" --max-tokens "$max_tokens" --seed 123 --threads "$threads" \
        --session-option "fish_drama3.cpu_expert_layers=$cpu_expert_layers" \
        --session-option "fish_drama3.context_length=$context_length" \
        --metrics ${extra[@]+"${extra[@]}"} --out "$output/$name.wav" 2>&1 | tail -n 20 | tee -a "$output/run.log"
done < "$output/requests.tsv"

echo "peak GPU MiB: $(cut -d, -f2 "$output/gpu-memory.csv" | sort -n | tail -1)" | tee -a "$output/run.log"
echo "min host available MiB: $(cut -d, -f3 "$output/host-memory.csv" | sort -n | head -1)" | tee -a "$output/run.log"
echo "max swap used MiB: $(cut -d, -f4 "$output/host-memory.csv" | sort -n | tail -1)" | tee -a "$output/run.log"
