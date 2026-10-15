#!/usr/bin/env bash
# Generate a fixed set of Drama 3 cases in one warm session with one model
# package and one expert placement, recording GPU/host memory once per second.
#
#   scripts/drama3_run_cases.sh <model.gguf> <cpu_expert_layers|auto> <output-dir> <cases.json> [extra audiocpp_cli args...]
#
# cases.json: [{"name": "...", "text": "...", "language": "en"|"ja",
#               "voice_ref": optional wav (relative paths are relative to the repository root),
#               "reference_text": optional,
#               "options": optional {request option: string value}}]
# MEMORY_LIMIT (e.g. 26g) caps the container's RAM with swap disabled, to emulate a
# machine with less memory. SAMPLE_MS sets the GPU memory sampling interval (default 1000).
# A short warm-up request runs first so first-use costs (graph capture, page
# faults) do not land on the measured cases. Runs audiocpp_cli inside the
# drama3-lowmem-dev image (see docker/dev.Dockerfile).
set -euo pipefail

model=$(realpath "$1")
cpu_expert_layers=$2
output=$(realpath -m "$3")
cases=$(realpath "$4")
shift 4
source_dir=$(cd "$(dirname "$0")/.." && pwd)
threads=${THREADS:-16}
context_length=${CONTEXT_LENGTH:-4096}
max_tokens=${MAX_TOKENS:-1500}
mkdir -p "$output"

python3 - "$cases" "$max_tokens" "$source_dir" "$output/mounts.txt" "$(dirname "$model")" "$output" > "$output/requests.json" <<'EOF'
import json, os, sys
cases_path, max_tokens, source_dir, mounts_path, model_dir, output_dir = sys.argv[1:]
cases = json.load(open(cases_path))
mounted = {source_dir, model_dir, output_dir}  # mounted below; docker rejects a duplicate
mounts = set()
requests = [{"id": "warmup", "text": "This is a warm up.", "options": {"seed": "1", "max_tokens": "64"}}]
for case in cases:
    request = {"id": case["name"], "text": case["text"], "options": {"seed": "123", "max_tokens": max_tokens}}
    request["options"].update(case.get("options", {}))
    if case.get("voice_ref"):
        request["voice_ref"] = os.path.realpath(os.path.join(source_dir, case["voice_ref"]))
        mounts.add(os.path.dirname(request["voice_ref"]))
        request["options"]["reference_text"] = case["reference_text"]
    requests.append(request)
print(json.dumps(requests, ensure_ascii=False, indent=1))
open(mounts_path, "w").write("".join(f"{m}\n" for m in sorted(mounts - mounted)))
EOF
mount_args=()
while read -r dir; do mount_args+=(-v "$dir:$dir:ro"); done < "$output/mounts.txt"
memory_args=()
if [ -n "${MEMORY_LIMIT:-}" ]; then memory_args=(--memory "$MEMORY_LIMIT" --memory-swap "$MEMORY_LIMIT"); fi

nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms "${SAMPLE_MS:-1000}" > "$output/gpu-memory.csv" &
gpu_sampler=$!
(while true; do echo "$(date +%T),$(free -m | awk '/^Mem:/{print $3","$7} /^Swap:/{print $3}' | paste -sd,)"; sleep 1; done) > "$output/host-memory.csv" &
host_sampler=$!
trap 'kill $gpu_sampler $host_sampler 2>/dev/null || true' EXIT

# Pass ggml CUDA tuning variables (e.g. GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F) through to the container.
env_args=()
for name in $(compgen -e | grep '^GGML_' || true); do env_args+=(-e "$name"); done
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp ${env_args[@]+"${env_args[@]}"} \
    ${memory_args[@]+"${memory_args[@]}"} ${mount_args[@]+"${mount_args[@]}"} \
    -v "$source_dir":"$source_dir" -v "$(dirname "$model")":"$(dirname "$model")" -v "$output":"$output" \
    drama3-lowmem-dev \
    "$source_dir/build/linux-cuda-release/bin/audiocpp_cli" \
    --task tts --family fish_drama3 --backend cuda --mode offline \
    --model "$model" --model-spec-override "$source_dir/model_specs/fish_drama3.json" \
    --threads "$threads" \
    --session-option "fish_drama3.cpu_expert_layers=$cpu_expert_layers" \
    --session-option "fish_drama3.context_length=$context_length" \
    --request-sequence "$output/requests.json" --out-dir "$output" --metrics "$@" > "$output/run.log" 2>&1 || {
        echo "audiocpp_cli failed; see $output/run.log" >&2
        tail -n 5 "$output/run.log" >&2
        failed=1
    }

{
    grep -E "metrics\[[^]]*\]\.(wall_ms|audio_duration_ms|rtf)=|frames|reached_limit|rror|failed" "$output/run.log" || true
    echo "peak GPU MiB: $(cut -d, -f2 "$output/gpu-memory.csv" | sort -n | tail -1)"
    echo "min host available MiB: $(cut -d, -f3 "$output/host-memory.csv" | sort -n | head -1)"
    echo "max swap used MiB: $(cut -d, -f4 "$output/host-memory.csv" | sort -n | tail -1)"
} | tee "$output/summary.txt"
exit "${failed:-0}"
