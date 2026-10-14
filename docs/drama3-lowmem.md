# drama3-lowmem: Fish Audio Drama 3 on one consumer GPU, without extra quantization

Fish Audio Drama 3 is a 35B-parameter mixture-of-experts text-to-speech model. Its
official checkpoint is 41.5 GB and the official runtime (SGLang-Omni) is tested on
80 GB+ data-center GPUs. drama3-lowmem runs the same weights, at the precision Fish
Audio shipped them, on an RTX 5090 (32 GB) with system RAM holding the rest.

Built with Fish Audio. The Drama 3 weights are licensed under the Fish Audio
Community License v1.0; this repository's code stays under Apache-2.0.

## What "without extra quantization" means here

The official checkpoint stores the routed experts as FP8 E4M3 with one FP32 scale
per 128x128 block, and everything else as BF16. drama3-lowmem keeps exactly that:

- **Experts** use a new ggml type, `f8_e4m3_b128`: 128 FP8 bytes plus one FP32
  scale per block (132 bytes). Each 128-column chunk of a row lies inside one
  checkpoint scale block, so the bytes are copied from the checkpoint unchanged.
  Dequantization is bit-exact (tested for every E4M3 code).
- **BF16 tensors** stay BF16; 1-D tensors (norms, biases) are stored as F32.
- **Codec** weights are stored as FP16, matching the official runtime's default
  decode dtype.

The resulting package is 42.0 GB, the size of the original checkpoint. No
re-quantization, no calibration, no pruning.

## How it fits

Each of the 40 MoE layers holds 256 experts (about 0.83 GB per layer in FP8), but a
token only uses 8 of them. The first `cpu_expert_layers` layers keep their experts
in system RAM and compute them on the CPU; everything else (attention, shared
experts, routers, the remaining expert layers, the audio tower, the codec) runs on
the GPU. Per CPU layer only the layer's activations (about 8 KB) cross PCIe;
weights never move.

On the CPU the FP8 dot product is an AVX-512 kernel that streams the expert bytes
at about 67 GB/s (close to dual-channel DDR5 bandwidth). On the GPU a dedicated
FP8 mat-vec kernel handles `mul_mat_id` at batch 1 (about 2x faster than F16
experts).

## Results on the reference machine

RTX 5090 32 GB, Ryzen 9 9950X3D, 59 GiB RAM, CUDA 12.9, Linux. RTF is generation
time divided by audio length (lower is faster; 1.0 is real time).

| Package | CPU expert layers | RTF (EN short / JA short / JA long / dialogue) | Peak VRAM |
|---|---|---|---|
| FP8, unquantized (this project) | 9 | 0.36 / 0.30 / 0.31 / 0.34 | 31.9 GB |
| FP8, unquantized (this project) | 12 | 0.36 / 0.33 / - / - | 29.1 GB |
| F16-expanded experts (first attempt) | 26 | 1.25-2.2 | 30.1 GB |
| Community Q4_K_S (for comparison, quantized) | 0 | 0.27 | 23.7 GB |

Per frame at 9 CPU layers: backbone about 10 ms, audio tower 2.8 ms, sampling
0.4 ms; codec decoding overlaps with generation on a background thread. The
process uses about 10 GB of system RAM at 9 CPU layers.

Voice cloning: the first request with a new reference also encodes it on the
CPU (about 1 s for a 4 s clip, 7 s for a 25 s clip), so a 3.4 s clone took RTF
0.69; later requests with the same voice reuse the codes (RTF 0.37). All
outputs above transcribe back to their input text with Whisper (character error
rate 0-3 %, the 3 % being a kana/kanji spelling difference).

## Accuracy

An independent FP32 PyTorch reference (`tools/reference/`, written from the
official SGLang-Omni code, not from this port) was run teacher-forced on five
cases (the official-numerics emulation on three of them). Results per decoding
step:

| | hidden-state relative error | top-1 next-token agreement |
|---|---|---|
| this port (FP8 experts) vs FP32 reference | 2.0-3.9 % | 94.4-98.9 % |
| official runtime numerics (BF16 + FP8 activations, emulated) vs FP32 reference | 3.6-4.6 % | 87.6-97.0 % |

The port is as close to exact math as the official runtime, or closer. Keeping
the experts in FP8 instead of expanding them to F16 makes no measurable
difference.

## Quality guards (not in the official runtime)

Two failure modes show up with the official sampling rule (top-p 0.9, top-k 30,
repetition-aware resampling), independent of this port's numerics:

1. **Stuck silence.** The model repeats one silence frame indefinitely: end-of-
   speech is the second most likely token but top-p keeps cutting it.
2. **Early stop in dialogue.** A multi-speaker text ends after the first speaker.

drama3-lowmem therefore:

- generates long text about one sentence at a time (`text_chunk_size` 60 instead
  of 200), each chunk continuing from the previous one;
- generates each `<|speaker:N|>` turn as its own chunk (`split_speaker_turns`);
- stops a take that repeats one frame `stall_frames` (64, about 3 s) times, or
  that is far shorter than its text (`min_seconds_per_char`, 0.02), and
  regenerates it with another seed (`stall_retries`, 2).

Over 8 seeds each (Whisper character error rate < 0.15 counts as complete):

| Text | Official chunking, no guards | With drama3-lowmem defaults |
|---|---|---|
| Japanese paragraph, 182 characters | 5/8 | 8/8 (no retry needed) |
| Japanese four-line, two-speaker dialogue | 4/8 | 8/8 |
| English paragraph, 4 sentences | - | 8/8 |

All of these can be turned off per request.

## Quick start (Linux + NVIDIA)

```bash
# 1. Build (no local CUDA toolkit needed)
docker build -f docker/dev.Dockerfile -t drama3-lowmem-dev docker/
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/src -w /src drama3-lowmem-dev \
  scripts/build_linux.sh --backend cuda --cuda-arch 120 --build-type Release \
    --model-set custom --models fish_drama3

# 2. Convert the official checkpoint (CPU only, about 5 minutes, < 45 GB RAM)
hf download fishaudio/drama-3 --local-dir models/drama-3
python3 -m venv venv && venv/bin/pip install torch numpy safetensors "gguf>=0.19" \
  --extra-index-url https://download.pytorch.org/whl/cpu
venv/bin/python tools/community_models/convert_fish_drama3.py \
  --checkpoint models/drama-3 --output models/drama3-fp8-intermediate.gguf \
  --quantized-output models/drama3-fp8.gguf --gguf-tool <path to audiocpp_gguf> \
  --expert-type orig --backbone-type orig --embedding-type orig --fp8-expert-layers 40

# 3. Generate
build/linux-cuda-release/bin/audiocpp_cli --task tts --family fish_drama3 --backend cuda \
  --model models/drama3-fp8.gguf --threads 16 \
  --session-option fish_drama3.cpu_expert_layers=9 \
  --text '<|speaker:0|>Hello there.<|speaker:1|>Hi! Nice to meet you.' --out out.wav
```

Add `--voice-ref ref.wav --request-option 'reference_text=...'` to clone a voice.

## Choosing `cpu_expert_layers`

The GPU needs about 6 GB for everything except the GPU-resident expert layers, plus
about 0.83 GB per expert layer it keeps. System RAM needs about 0.83 GB per
CPU-resident layer plus about 3 GB.

| GPU VRAM | `cpu_expert_layers` | System RAM needed | Expected RTF |
|---|---|---|---|
| 32 GB (headless) | 9 | ~10 GB | 0.3 (measured) |
| 32 GB (with a desktop) | 10-12 | ~12 GB | 0.31-0.36 (measured) |
| 24 GB | ~19 | ~19 GB | ~0.4-0.5 (estimate) |
| 16 GB | ~29 | ~27 GB | ~0.6-0.8 (estimate) |
| 12 GB | ~34 | ~31 GB | ~0.8-1.0 (estimate) |

Only the 32 GB rows were measured. Estimates scale the measured per-layer cost
(about 0.5 ms per CPU layer per frame on DDR5) and assume a GPU at least as
capable as the expert kernels need; slower GPUs and RAM will be slower. Use 16 CPU
threads on a 16-core CPU; using all SMT threads was 6x slower.

## Options added by drama3-lowmem

| Option | Kind | Default | Effect |
|---|---|---|---|
| `fish_drama3.cpu_expert_layers` | session | 4 | Expert layers kept and computed on the CPU |
| `fish_drama3.offline_codec` | session | `stream` | Decode offline takes frame by frame (constant VRAM) instead of one graph per take |
| `fish_drama3.reference_encoder` | session | `cpu` | Encode reference audio on the CPU |
| `fish_drama3.profile_steps` | session | false | Print the per-frame time split |
| `fish_drama3.teacher_dump` / `teacher_frames` | session | - | Numerical validation dumps (see `tools/reference`) |
| `stall_frames`, `min_seconds_per_char`, `stall_retries` | request | 64, 0.02, 2 | Regenerate stalled or early-ended chunks |
| `text_chunk_size` | request | 60 (was 200) | Characters per chunk; about one sentence |
| `split_speaker_turns` | request | true | One chunk per speaker turn |

## Tests

- `tests/fp8/run.sh`: CPU FP8 type (exact decode of all codes, dot product, `mul_mat_id`)
- `tests/fp8/run_cuda.sh`: CUDA FP8 type, bit-exact against the CPU (and
  `test-backend-ops` covers it: MUL_MAT 1144/1144, MUL_MAT_ID 844/844)
- `tests/drama3/run.sh`: the on-device sampler against the CPU sampling rule,
  including boundary cases
- `tools/reference/`: FP32 and official-numerics reference, compared with
  `tools/community_models/drama3_compare_dumps.py`

## Limitations

- Tested on one machine (RTX 5090, Linux). Other GPUs, Windows and smaller VRAM
  configurations are untested.
- The quality guards are heuristics on top of the official sampling; they reduce
  but do not eliminate sampling failures, and dialogue turns are generated
  sequentially.
- The vendored ggml is a fork; the FP8 block type has no upstream counterpart
  (upstream's FP8 proposals keep scales per tensor or per channel, which cannot
  express 128x128 block scales).
- No hosted service may be built on the weights without a separate license from
  Fish Audio.
