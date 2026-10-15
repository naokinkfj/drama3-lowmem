# drama3-lowmem

English | [日本語](README.ja.md)

Run [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3), a 35B-parameter
mixture-of-experts text-to-speech model, **without extra quantization** on one
consumer GPU plus system RAM.

**Built with Fish Audio.** drama3-lowmem is an independent project, not affiliated
with or endorsed by Fish Audio.

The official checkpoint is 41.5 GB and the official runtime (SGLang-Omni) targets
80 GB+ data-center GPUs. drama3-lowmem keeps the weights at the precision Fish Audio
shipped them (FP8 experts, BF16 elsewhere) and keeps the expert layers that do not
fit in VRAM in system RAM, computing them on the CPU, in the spirit of llama.cpp's
`--n-cpu-moe`.

## Results

On an RTX 5090 (32 GB), a Ryzen 9 9950X3D and 59 GiB RAM, with 9 of the 40 expert
layers in RAM:

| | |
|---|---|
| Speed (RTF, generation time / audio length) | 0.29-0.36 (30 s of speech in about 9-11 s) |
| Peak VRAM / system RAM | 31.9 GB / about 10 GB |
| Package size | 42.0 GB (the official checkpoint is 41.5 GB) |
| Accuracy vs. an FP32 reference | hidden-state error 2.0-3.9 %, top-1 agreement 94.4-98.9 % (the official runtime's numerics: 3.6-4.6 %, 87.6-97.0 %) |
| Robustness, 8 seeds each | Japanese paragraph 8/8, two-speaker dialogue 8/8, English paragraph 8/8 |

## How it works

- **A new ggml type, `f8_e4m3_b128`,** stores the official FP8 expert bytes
  unchanged with one FP32 scale per 128 values, with AVX-512 (CPU) and CUDA
  kernels. Decoding is bit-exact.
- **Expert placement:** the first `cpu_expert_layers` layers keep their experts in
  RAM; only about 8 KB of activations per layer cross PCIe.
- **Per-frame work is trimmed:** the LM head computes only the audio-code rows,
  the 13-layer audio tower and its sampling run as one GPU graph, and codec
  decoding overlaps generation on a background thread.
- **Quality guards (not in the official runtime):** sentence and speaker-turn
  chunking, and regeneration of takes that stall in silence or end early.

Details, measurements and the memory guide for smaller GPUs are in
[docs/drama3-lowmem.md](docs/drama3-lowmem.md).

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
Only clone voices you have permission to use.

## Documentation

- [docs/drama3-lowmem.md](docs/drama3-lowmem.md): how it works, results, accuracy,
  options, memory guide and licenses
- [docs/report/index.html](docs/report/index.html): a work report for non-specialist
  readers, with audio samples (Japanese)
- [docs/PLAN.md](docs/PLAN.md): plan and progress log (Japanese)
- [tools/reference/README.md](tools/reference/README.md): the FP32 reference used for
  accuracy checks
- [UPSTREAM_README.md](UPSTREAM_README.md): the README of audio.cpp, which this
  repository is built on

## License

- **Code:** Apache License 2.0 ([LICENSE](LICENSE)). drama3-lowmem is derived from
  [audio.cpp](https://github.com/0xShug0/audio.cpp) (Copyright 2026 ShugoAI LLC)
  through [JarodMica/audio.cpp](https://github.com/JarodMica/audio.cpp), branch
  `drama-3-native`, commit `4dbb4d1`. [NOTICE](NOTICE) lists the changes and the
  vendored third-party components (ggml, llama.cpp tokenizer, cJSON, cpp-httplib,
  libyaml, SentencePiece) with their licenses.
- **Model weights:** not included. The Drama 3 checkpoint, its codec and any
  package converted from them are under the
  [Fish Audio Community License v1.0](licenses/FISH-AUDIO-COMMUNITY-LICENSE-1.0.md):
  free for research and non-commercial use; free for commercial use while your
  and your affiliates' total gross revenue is below USD 500,000 a year; and a
  separate license from Fish Audio for a Hosted Service, where others generate
  speech or clone voices from their own input (free, unmonetized research demos
  excepted). See
  [docs/drama3-lowmem.md](docs/drama3-lowmem.md#licenses) before redistributing a
  converted package.

## Acknowledgments

Fish Audio for Drama 3; ShugoAI and the audio.cpp contributors; JarodMica for the
native Drama 3 port; the SGLang-Omni team for the reference implementation;
the ggml and llama.cpp authors.
