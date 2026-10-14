# Drama 3 FP32 reference

An independent PyTorch implementation of Fish Audio Drama 3's text backbone and codebook
tower, used to check a port numerically. It computes "ideal math": every weight is
upcast to float32 (or float64 with `--dtype float64`) and every operation runs in that
dtype. The only loss it keeps is the checkpoint's own: BF16 weights and FP8 E4M3 routed
experts with 128x128 block scales, which are dequantized exactly. It does not emulate the
official runtime's BF16 activations or its FP8 activation quantization, so a port should
match it to within the error of those approximations, not bit for bit.

It was written from the official implementation, not from the port:

| Part | Mirrors |
|---|---|
| Frame embedding (`Backbone.embed`) | sglang-omni `audio_decoder.py` `AudioDecoder.embed_backbone` |
| Backbone (`Backbone.forward`) | sglang `models/qwen3_5.py` `Qwen3_5MoeForCausalLM`; HF `qwen3_5_moe` as a cross-check |
| Gated DeltaNet (`linear_attention`, `gated_delta_rule_*`) | `Qwen3_5GatedDeltaNet`; FLA `chunk_gated_delta_rule` / `fused_recurrent`; `fla/layernorm_gated.py` |
| Gated full attention (`full_attention`) | `Qwen3_5AttentionDecoderLayer` (per-head `[q, gate]` rows, NeoX partial rotary) |
| MoE (`moe`) | sglang `qwen2_moe.py` `Qwen2MoeSparseMoeBlock`; HF `Qwen3_5MoeSparseMoeBlock` |
| Main-token logits | `sglang_model.py` `backbone_logits` and `AudioDecoder.main_token_candidate_ids` |
| Tower (`Tower`) | `decoder_layers.py` `DecoderLayer` / `DecoderSelfAttention` / `InterleavedRotaryEmbedding`, `audio_decoder.py` `shared_input` / `prefill` / `decode_step`, `model_inputs.py` `next_frame_labels`, `weights.py` key layout |
| Prompt rows (`make_case.py`) | `prompt.py` `build_prompt_rows`, imported from an official checkout |

Sources read: sglang-omni `e0baa428`, sglang `b6778515`, transformers `637b76e4`.

## Files

- `drama3_reference.py`: the model and three commands.
  - `run <case_dir>`: reads `meta.json`, writes `ref_hidden.f32`, `ref_semantic.f32`,
    `ref_residual.f32` and `ref_meta.json` into the same directory.
  - `check-steps <case_dir> [--rows N]`: one causal forward over the first N rows
    against a token-by-token decode (recurrent delta rule, conv window, KV caches), and
    the tower over all labelled rows at once against one row a step.
  - `check-gdn`: the chunked delta rule against the recurrent one on random inputs.
- `check_hf_backbone.py <case_dir> [--layers 4] [--rows 80]`: loads the first layers of
  the backbone into transformers' `Qwen3_5MoeTextModel` (experts dequantized to dense
  float32) and compares final-norm hidden states with `Backbone.forward`. An
  independent implementation of the backbone math; needs `transformers>=5`.
- `compare_dump.py <case_dir>...`: compares a port's teacher-forced dump
  (`hidden_prefill.f32`, `hidden_decode.f32`, `semantic.f32`, `residual.f32`) with the
  `ref_*` files in the same directory (relative L2, per-row percentiles, argmax
  agreement) and shows the closest router ties from `ref_meta.json`.
- `make_case.py`: writes a synthetic `meta.json` from the official `build_prompt_rows`
  plus random valid frames, and prints the rows and the detokenized prompt.

## Interface

Input `meta.json`: `prompt_rows` and `frames`, each a list of
`[main_token, c0, ..., c9]`. Text rows carry zero codes, as `build_prompt_rows` writes
them. Reference audio and generated frames are `[semantic_begin + c0, c0, ..., c9]`. Any
of `hidden_size`, `codebook_size`, `num_codebooks`, `eos`, `semantic_begin`,
`shared_layers` or `residual_layers` that is present is checked against `config.json`.

Outputs are little-endian float32 in C order, with P prompt rows and F frames:

- `ref_hidden.f32` `[P+F, 2048]`: the backbone's final-norm hidden (the lm_head input)
  at every row, from one causal forward.
- `ref_semantic.f32` `[F, 4097]`: raw lm_head logits at row `P-1+t` for
  `[eos, semantic_begin + 0 .. semantic_begin + 4095]`.
- `ref_residual.f32` `[F, 9, 4096]`: raw logits of codebook `r+1` at decode step `t`,
  teacher-forced with codebooks `0..r` of `frames[t]`.

## How the tower is driven

Tower position p sees the backbone hidden at p, the frame row at p (codebooks 2..9 of a
semantic row enter the first shared layer) and the label: the row at p+1. Shared layers
embed the label's codebook 0, residual layer r embeds its codebook r, and head r
predicts codebook r+1. A label that is not semantic (text, eos) contributes a zero
embedding. The run mirrors the runtime: a prefill over prompt rows `0..P-2`, whose
labels are the next prompt row, then one step per frame at row `P-1+t` with label
`frames[t]`. The official prefill also writes row P-1 with a placeholder label, but the
decode step of the same forward overwrites that slot before anything attends to it, so
the two are the same.

## Running

```
PY=python3
$PY make_case.py --official-prompt <sglang-omni>/sglang_omni/models/fishaudio_drama3/prompt.py \
    --out cases/smoke_en --frames 8
$PY drama3_reference.py --threads 8 run cases/smoke_en
$PY drama3_reference.py --threads 8 check-steps cases/smoke_en --rows 24
$PY drama3_reference.py check-gdn
```

`--model-dir` defaults to `models/drama-3` (HF layout with
`model.safetensors.index.json`). The shards are memory-mapped and weights are upcast
when used, so the resident set stays small (activations plus one upcast matrix at a
time). The shards are read through the page cache, so low free memory makes a run
slower but does not stop it. `peak_rss_gib` in `ref_meta.json` counts mapped shard pages
too (about 30-37 GiB on the long cases); most of that is clean, reclaimable page cache.
The tower keeps its float32 weights (about 4 GiB) because it runs once per frame.

## Choices where the sources leave room

- Rotary angles are formed in float64 (`position * base^(-2i/d)`) and then cast. The
  runtime's float32 table differs by about 1e-4 rad at position 2000.
- The full forward uses the chunked delta rule (chunk 64) and `check-steps` uses the
  recurrent one. Both are exact forms of the same recurrence; they agree to about 1e-6
  in float32 and 1e-15 in float64.
- Top-8 routing uses `torch.topk` on the softmax probabilities, renormalized
  (`norm_topk_prob`). Real inputs do produce near-ties: in `en_short` (178 rows)
  `log(p8 / p9)` is below 5e-7, a few float32 ulps, at three (row, layer) pairs where the
  8th expert carries 7-12% of the routing weight; float64 confirms the ties. Two implementations can break such a tie differently, and
  the difference then carries to every later row through attention and the delta-rule
  state. `ref_meta.json` lists the smallest logit gaps among
  tokens whose 8th expert carries at least 1e-3 of the weight (`router_near_ties`), so
  that a spike in a comparison can be traced to one. The reference's float32 and float64 runs pick the
  same experts everywhere in `en_short`.

## Validation record (Ryzen 9 9950X3D, torch 2.14 CPU)

| Check | Result |
|---|---|
| `check-gdn` (200 tokens, 4 chunks) | chunked vs recurrent rel-L2 6e-7 (float32), 1e-15 (float64) |
| `check_hf_backbone.py`, 4 layers, 120 rows of `en_short` | vs transformers 5.19 rel-L2 2.3e-7, max abs 3.8e-6 |
| `check-steps`, 24 rows of `en_short` | hidden rel-L2 4.0e-7; tower logits rel-L2 5.5e-7 |
| `check-steps`, 64 rows of `smoke_ref` (with a voice reference) | hidden rel-L2 6.1e-7; tower logits rel-L2 1.1e-6 |
| `check-steps`, 150 rows of `en_short` | identical to 7e-7 up to row 108; at row 109 layer 20 `log(p8/p9)` is 4e-7 with the 8th expert at 12% weight, the stepwise path picks the other expert, and rows after that differ by 0.4-9% |
| `run --dtype float64` vs float32, `en_short` | hidden rel-L2 5.3e-7 (worst row 2e-6); semantic 3.2e-7, residual 5.2e-7; argmax identical |
| `run` after the cleanup pass | outputs bit-identical to the run before it |
| `make_case.py` | detokenized rows reproduce the `reference-bracket-v1` template; text rows carry zero codes; reference rows are `[semantic_begin + c0, c0..c9]` |
