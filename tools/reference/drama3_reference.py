#!/usr/bin/env python3
"""Independent FP32 PyTorch reference for Fish Audio Drama 3: text backbone + codebook tower.

Purpose: numerically validate a port of Drama 3 against "ideal math". Every weight is
upcast to the compute dtype (float32 by default, float64 on request) and every
operation runs in that dtype. Nothing emulates the official runtime's BF16
activations or its FP8 activation quantization; the only lossy step is the checkpoint
itself (BF16 weights, FP8 E4M3 routed experts with 128x128 block scales), which is
dequantized exactly.

Ground truth is the official implementation, not the port:

  sglang-omni  sglang_omni/models/fishaudio_drama3/
      sglang_model.py   Drama3ForCausalLM.backbone_logits: frame embeddings -> backbone
                        -> final-norm hidden -> lm_head
      audio_decoder.py  AudioDecoder.embed_backbone / shared_input / prefill / decode_step
      decoder_layers.py DecoderLayer, DecoderSelfAttention, InterleavedRotaryEmbedding
      model_inputs.py   next_frame_labels (tower prefill labels = the next row)
      weights.py        checkpoint key layout of the tower (wqkv = [q | gate | k | v])
      prompt.py         build_prompt_rows (text rows carry zero codes)
  sglang       python/sglang/srt/models/qwen3_5.py  Qwen3_5MoeForCausalLM (the backbone):
                        Qwen3_5GatedDeltaNet, Qwen3_5AttentionDecoderLayer,
                        Qwen2MoeSparseMoeBlock (qwen2_moe.py), GemmaRMSNorm (layernorm.py),
                        fla/layernorm_gated.py (RMSNormGated)
  transformers models/qwen3_5_moe/modeling_qwen3_5_moe.py was used as a cross-check of
               the backbone math (it agrees with sglang on every point used here).

Memory: the safetensors shards are memory-mapped, never loaded whole. Weights are
upcast on use, and a routed expert is dequantized only when a token selects it, so the
resident set is the page cache of the shards plus activations.

Commands (see README.md):
  run          <case_dir>   write ref_hidden.f32 / ref_semantic.f32 / ref_residual.f32
  check-steps  <case_dir>   full causal forward vs token-by-token recurrent decode
  check-gdn                 chunked vs recurrent gated delta rule on random inputs
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import resource
import struct
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

DEFAULT_MODEL_DIR = "models/drama-3"
GDN_CHUNK_SIZE = 64  # chunk length of the chunked gated delta rule (FLA uses 64)
FP8_BLOCK = 128  # routed experts: one float32 scale per 128x128 weight block

SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
}


# --------------------------------------------------------------------------------------
# Checkpoint access
# --------------------------------------------------------------------------------------


class Checkpoint:
    """Zero-copy, memory-mapped view of a sharded safetensors checkpoint."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        self.entries: dict[str, tuple[mmap.mmap, torch.dtype, list[int], int]] = {}
        for file_name in sorted(set(weight_map.values())):
            path = os.path.join(model_dir, file_name)
            with open(path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))
                # ACCESS_COPY gives a writable (copy-on-write) buffer, which
                # torch.frombuffer wants; nothing ever writes to it.
                mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                begin, _ = info["data_offsets"]
                self.entries[name] = (
                    mapping,
                    SAFETENSORS_DTYPES[info["dtype"]],
                    info["shape"],
                    8 + header_size + begin,
                )
        missing = set(weight_map) - set(self.entries)
        if missing:
            raise ValueError(f"index names {len(missing)} tensors no shard holds")

    def raw(self, name: str) -> torch.Tensor:
        """The stored tensor (stored dtype), backed by the mapping."""
        mapping, dtype, shape, offset = self.entries[name]
        count = math.prod(shape)
        return torch.frombuffer(mapping, dtype=dtype, count=count, offset=offset).view(
            shape
        )

    def get(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        return self.raw(name).to(dtype)

    def rows(self, name: str, row_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Selected rows of a 2-D table (embeddings, lm_head) without upcasting all of it."""
        return self.raw(name)[row_ids].to(dtype)


def dequantize_fp8_block(
    weight: torch.Tensor, scale_inv: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """FP8 E4M3 weight [out, in] with per-128x128-block scale_inv -> dense weight.

    Same convention as sglang's Fp8Config block quantization (DeepSeek-V3 style):
    dense = fp8_value * scale_inv[out // 128, in // 128].
    """
    out_features, in_features = weight.shape
    blocks = weight.to(dtype).view(
        out_features // FP8_BLOCK, FP8_BLOCK, in_features // FP8_BLOCK, FP8_BLOCK
    )
    return (blocks * scale_inv.to(dtype)[:, None, :, None]).view(
        out_features, in_features
    )


# --------------------------------------------------------------------------------------
# Shared math
# --------------------------------------------------------------------------------------


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool = True
) -> torch.Tensor:
    """RMSNorm over the last dim. zero_centered: scale by (1 + w) (GemmaRMSNorm)."""
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed * ((1.0 + weight) if zero_centered else weight)


def rotary_cos_sin(
    positions: torch.Tensor, rotary_dim: int, base: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin [rows, rotary_dim // 2] of position * base^(-2i / rotary_dim).

    Angles are formed in float64 (exact for any realistic position), then cast.
    """
    inverse_frequencies = 1.0 / base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim
    )
    angles = positions.to(torch.float64).unsqueeze(-1) * inverse_frequencies
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX-style rotary on the first 2 * cos.shape[-1] dims of x [rows, heads, dim].

    Pairs are (i, i + rotary_dim / 2), the rest of the head passes through.
    """
    half = cos.shape[-1]
    first, second, rest = x[..., :half], x[..., half : 2 * half], x[..., 2 * half :]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin, rest), dim=-1
    )


def rotate_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """GPT-J (interleaved) rotary on the first 2 * cos.shape[-1] dims of x.

    Pairs are (2i, 2i + 1); mirrors decoder_layers.InterleavedRotaryEmbedding.
    """
    rotary_dim = 2 * cos.shape[-1]
    rotary, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return torch.cat((rotated.flatten(-2), rest), dim=-1)


@dataclass
class KVCache:
    """Keys / values [rows, kv_heads, head_dim] of every position seen so far."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def append(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = torch.cat((self.keys, keys))
            self.values = torch.cat((self.values, values))
        return self.keys, self.values


def causal_gqa_attention(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, scale: float
) -> torch.Tensor:
    """Causal grouped-query attention of the newest query rows over all cached rows.

    query [q_rows, heads, d]; keys / values [kv_rows, kv_heads, d] with the q_rows
    newest positions last. Query head h reads kv head h // (heads / kv_heads).
    Returns [q_rows, heads, d].
    """
    q_rows, head_count, _ = query.shape
    kv_rows, kv_head_count, _ = keys.shape
    group = head_count // kv_head_count
    keys = keys.repeat_interleave(group, dim=1)
    values = values.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query, keys) * scale
    query_positions = torch.arange(kv_rows - q_rows, kv_rows).unsqueeze(1)
    key_positions = torch.arange(kv_rows).unsqueeze(0)
    scores = scores.masked_fill(key_positions > query_positions, float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), values)


def silu_mlp(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


# --------------------------------------------------------------------------------------
# Gated delta rule (the linear-attention core of Qwen3.5's GatedDeltaNet)
# --------------------------------------------------------------------------------------
#
# Per value head, with key-dim x value-dim state S, decay g_t <= 0 and rate beta_t:
#   S_t = exp(g_t) S_{t-1} + k_t u_t^T,   u_t = beta_t (v_t - exp(g_t) S_{t-1}^T k_t)
#   o_t = S_t^T q_t
# (q, k L2-normalized, q scaled by 1/sqrt(key_dim)). This is FLA's / sglang's
# chunk_gated_delta_rule and HF's torch_recurrent_gated_delta_rule.


def gated_delta_rule_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token gated delta rule.

    query / key [T, H, Dk], value [T, H, Dv], g / beta [T, H], state [H, Dk, Dv].
    Returns output [T, H, Dv] and the new state.
    """
    outputs = []
    for t in range(query.shape[0]):
        state = state * g[t].exp()[:, None, None]
        predicted = torch.einsum("hk,hkv->hv", key[t], state)
        update = (value[t] - predicted) * beta[t][:, None]
        state = state + key[t][:, :, None] * update[:, None, :]
        outputs.append(torch.einsum("hk,hkv->hv", query[t], state))
    return torch.stack(outputs), state


def gated_delta_rule_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    chunk_size: int = GDN_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked (WY / UT-transform) form of gated_delta_rule_recurrent, same I/O.

    Inside a chunk with entry state S0 and cumulative decay G_t = sum_{s<=t} g_s,
    the written values u solve (I + A) U = diag(beta) V - diag(beta e^G) K S0 with
    A[t, s] = beta_t e^(G_t - G_s) k_t.k_s for s < t (unit lower triangular system).
    Then o_t = e^(G_t) S0^T q_t + sum_{s<=t} e^(G_t - G_s) (q_t.k_s) u_s and
    S_end = e^(G_C) S0 + sum_s e^(G_C - G_s) k_s u_s^T.
    """
    length = query.shape[0]
    outputs = []
    for begin in range(0, length, chunk_size):
        end = min(begin + chunk_size, length)
        q, k, v = (x[begin:end].transpose(0, 1) for x in (query, key, value))
        chunk_g, chunk_beta = g[begin:end].T, beta[begin:end].T  # [H, C]
        cumulative = chunk_g.cumsum(dim=-1)  # G_t
        rows = end - begin
        causal = torch.ones(rows, rows, dtype=torch.bool).tril()
        # e^(G_t - G_s) for s <= t, zero above the diagonal.
        pair_decay = (
            (cumulative[:, :, None] - cumulative[:, None, :])
            .masked_fill(~causal, float("-inf"))
            .exp()
        )
        strictly_lower = (
            chunk_beta[:, :, None] * (k @ k.transpose(-1, -2)) * pair_decay
        ).tril(-1)
        system = strictly_lower + torch.eye(rows, dtype=q.dtype)
        written_from_values = torch.linalg.solve_triangular(
            system, chunk_beta[:, :, None] * v, upper=False, unitriangular=True
        )
        written_from_state = torch.linalg.solve_triangular(
            system,
            (chunk_beta * cumulative.exp())[:, :, None] * k,
            upper=False,
            unitriangular=True,
        )
        written = written_from_values - written_from_state @ state  # U [H, C, Dv]
        attention = (q @ k.transpose(-1, -2)) * pair_decay  # includes s == t
        chunk_out = (q * cumulative.exp()[:, :, None]) @ state + attention @ written
        outputs.append(chunk_out.transpose(0, 1))
        tail_decay = (cumulative[:, -1:] - cumulative).exp()  # e^(G_C - G_s)
        state = cumulative[:, -1].exp()[:, None, None] * state + (
            k * tail_decay[:, :, None]
        ).transpose(-1, -2) @ written
    return torch.cat(outputs), state


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x / sqrt(sum x^2 + eps), FLA's use_qk_l2norm_in_kernel."""
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)


# --------------------------------------------------------------------------------------
# Text backbone: Qwen3.5 MoE (hybrid GatedDeltaNet / gated full attention)
# --------------------------------------------------------------------------------------


@dataclass
class LinearAttentionCache:
    conv_window: torch.Tensor | None = None  # last (kernel - 1) conv inputs [k-1, C]
    state: torch.Tensor | None = None  # [value_heads, key_dim, value_dim]


class Backbone:
    """Qwen3_5MoeForCausalLM math (sglang qwen3_5.py), plus Drama 3's frame embedding."""

    PREFIX = "model.language_model."

    def __init__(self, checkpoint: Checkpoint, config: dict, dtype: torch.dtype):
        text = config["text_config"]
        decoder = config["audio_decoder_config"]
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.layer_types = text["layer_types"]
        self.eps = text["rms_norm_eps"]
        self.head_count = text["num_attention_heads"]
        self.kv_head_count = text["num_key_value_heads"]
        self.head_dim = text["head_dim"]
        rope = text["rope_parameters"]
        if rope.get("rope_type", "default") != "default" or "mrope_section" in rope:
            raise NotImplementedError(f"unsupported rope_parameters {rope}")
        self.rope_theta = rope["rope_theta"]
        self.rotary_dim = int(self.head_dim * rope["partial_rotary_factor"])
        self.linear_key_heads = text["linear_num_key_heads"]
        self.linear_value_heads = text["linear_num_value_heads"]
        self.linear_key_dim = text["linear_key_head_dim"]
        self.linear_value_dim = text["linear_value_head_dim"]
        self.experts_per_token = text["num_experts_per_tok"]
        self.normalize_top_k = text.get("norm_topk_prob", True)
        self.semantic_begin = config["semantic_start_token_id"]
        self.semantic_end = config["semantic_end_token_id"]
        self.codebook_size = decoder["vocab_size"]
        self.backbone_codebooks = decoder["vq_backbone_codebooks"]
        if text.get("rotary_interleaved", False):
            raise NotImplementedError("the backbone uses NeoX rotary")
        if decoder.get("vq_feature_placement") != "backbone_partial":
            raise NotImplementedError("only vq_feature_placement=backbone_partial")
        # (layer, p[k-1] - p[k] per row) of every MoE call: how close each token came
        # to swapping its k-th expert for the next one. See near_ties().
        self.router_margins: list[tuple[int, torch.Tensor]] = []

    def weight(self, name: str) -> torch.Tensor:
        return self.checkpoint.get(self.PREFIX + name, self.dtype)

    def near_ties(self, count: int = 10) -> list[dict]:
        """The smallest top-k router margins logged since router_margins was cleared.

        A margin below the working precision means the k-th and (k+1)-th experts are
        tied: any implementation may pick either, and the difference then propagates
        to every later row. Assumes one forward over rows 0.. (the full forward).
        """
        entries = [
            (float(margin), row, layer)
            for layer, margins in self.router_margins
            for row, margin in enumerate(margins.tolist())
        ]
        return [
            {"row": row, "layer": layer, "margin": margin}
            for margin, row, layer in sorted(entries)[:count]
        ]

    # -- inputs -------------------------------------------------------------------------

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin) & (token_ids <= self.semantic_end)

    def embed(self, rows: torch.Tensor) -> torch.Tensor:
        """Backbone input of frame rows [T, 1 + codebooks] (AudioDecoder.embed_backbone).

        Semantic rows: (token embedding + sum of codebooks 1..backbone_codebooks
        embeddings) / sqrt(1 + backbone_codebooks). Other rows: token embedding.
        """
        token = self.checkpoint.rows(
            self.PREFIX + "embed_tokens.weight", rows[:, 0], self.dtype
        )
        codebooks = range(1, 1 + self.backbone_codebooks)
        table_rows = torch.stack(
            [
                rows[:, 1 + c].clamp(0, self.codebook_size - 1) + c * self.codebook_size
                for c in codebooks
            ],
            dim=1,
        )
        codes = self.checkpoint.rows(
            "audio_decoder.codebook_embeddings.weight", table_rows, self.dtype
        ).sum(dim=1)
        mixed = (token + codes) / math.sqrt(1.0 + self.backbone_codebooks)
        return torch.where(self.is_semantic(rows[:, 0]).unsqueeze(-1), mixed, token)

    # -- layers -------------------------------------------------------------------------

    def new_cache(self) -> list:
        return [
            LinearAttentionCache() if kind == "linear_attention" else KVCache()
            for kind in self.layer_types
        ]

    def full_attention(
        self, i: int, x: torch.Tensor, positions: torch.Tensor, cache: KVCache
    ) -> torch.Tensor:
        """Gated full attention (Qwen3_5AttentionDecoderLayer.self_attention).

        q_proj rows are per head [query (head_dim), gate (head_dim)]; per-head
        zero-centered q/k RMSNorm, NeoX rotary on the first rotary_dim dims, causal
        GQA, then output * sigmoid(gate) -> o_proj.
        """
        p = f"layers.{i}.self_attn."
        rows = x.shape[0]
        query_and_gate = (x @ self.weight(p + "q_proj.weight").T).view(
            rows, self.head_count, 2 * self.head_dim
        )
        query, gate = query_and_gate.chunk(2, dim=-1)
        key = (x @ self.weight(p + "k_proj.weight").T).view(
            rows, self.kv_head_count, self.head_dim
        )
        value = (x @ self.weight(p + "v_proj.weight").T).view(
            rows, self.kv_head_count, self.head_dim
        )
        query = rms_norm(query, self.weight(p + "q_norm.weight"), self.eps)
        key = rms_norm(key, self.weight(p + "k_norm.weight"), self.eps)
        cos, sin = rotary_cos_sin(positions, self.rotary_dim, self.rope_theta, self.dtype)
        query, key = rotate_neox(query, cos, sin), rotate_neox(key, cos, sin)
        keys, values = cache.append(key, value)
        attended = causal_gqa_attention(query, keys, values, self.head_dim**-0.5)
        gated = attended * torch.sigmoid(gate)
        return gated.reshape(rows, -1) @ self.weight(p + "o_proj.weight").T

    def linear_attention(
        self, i: int, x: torch.Tensor, cache: LinearAttentionCache, recurrent: bool
    ) -> torch.Tensor:
        """Qwen3_5GatedDeltaNet.forward.

        in_proj_qkv -> causal depthwise conv1d (kernel 4) + SiLU over the [q | k | v]
        channels; q, k L2-normalized, q scaled by key_dim^-0.5; key heads shared by
        value_heads / key_heads consecutive value heads; beta = sigmoid(b),
        g = -exp(A_log) * softplus(a + dt_bias); gated delta rule; per-head RMSNorm
        (plain weight) * silu(z); out_proj.
        """
        p = f"layers.{i}.linear_attn."
        rows = x.shape[0]
        key_width = self.linear_key_heads * self.linear_key_dim
        mixed = x @ self.weight(p + "in_proj_qkv.weight").T  # [T, 2*key_width + value_width]
        z = (x @ self.weight(p + "in_proj_z.weight").T).view(
            rows, self.linear_value_heads, self.linear_value_dim
        )
        b = x @ self.weight(p + "in_proj_b.weight").T
        a = x @ self.weight(p + "in_proj_a.weight").T

        conv_weight = self.weight(p + "conv1d.weight").squeeze(1)  # [C, kernel]
        kernel = conv_weight.shape[1]
        if cache.conv_window is None:
            cache.conv_window = mixed.new_zeros(kernel - 1, mixed.shape[1])
        window = torch.cat((cache.conv_window, mixed))  # [kernel - 1 + T, C]
        cache.conv_window = window[-(kernel - 1) :]
        # out[t] = sum_j w[:, j] * window[t + j]; w[:, -1] multiplies the current input.
        convolved = sum(
            conv_weight[:, j] * window[j : j + rows] for j in range(kernel)
        )
        convolved = F.silu(convolved)

        query, key, value = convolved.split(
            [key_width, key_width, self.linear_value_heads * self.linear_value_dim],
            dim=-1,
        )
        group = self.linear_value_heads // self.linear_key_heads
        query = l2_normalize(query.view(rows, self.linear_key_heads, -1))
        key = l2_normalize(key.view(rows, self.linear_key_heads, -1))
        query = query * self.linear_key_dim**-0.5
        query = query.repeat_interleave(group, dim=1)
        key = key.repeat_interleave(group, dim=1)
        value = value.view(rows, self.linear_value_heads, self.linear_value_dim)
        beta = torch.sigmoid(b)
        g = -self.weight(p + "A_log").exp() * F.softplus(a + self.weight(p + "dt_bias"))

        if cache.state is None:
            cache.state = x.new_zeros(
                self.linear_value_heads, self.linear_key_dim, self.linear_value_dim
            )
        rule = gated_delta_rule_recurrent if recurrent else gated_delta_rule_chunked
        core, cache.state = rule(query, key, value, g, beta, cache.state)
        core = rms_norm(core, self.weight(p + "norm.weight"), self.eps, False) * F.silu(z)
        return core.reshape(rows, -1) @ self.weight(p + "out_proj.weight").T

    def moe(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """Qwen2MoeSparseMoeBlock: softmax router, top-k (renormalized) FP8 experts,
        plus a sigmoid-gated shared expert."""
        p = f"layers.{i}.mlp."
        router_probs = torch.softmax(x @ self.weight(p + "gate.weight").T, dim=-1)
        top_probs, top_experts = router_probs.topk(self.experts_per_token + 1, dim=-1)
        self.router_margins.append((i, top_probs[:, -2] - top_probs[:, -1]))
        top_probs, top_experts = top_probs[:, :-1], top_experts[:, :-1]
        if self.normalize_top_k:
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        routed = torch.zeros_like(x)
        for expert in top_experts.unique().tolist():
            token_ids, slots = (top_experts == expert).nonzero(as_tuple=True)
            weights = [self.expert_weight(i, expert, n) for n in ("gate", "up", "down")]
            out = silu_mlp(x[token_ids], *weights)
            routed.index_add_(0, token_ids, out * top_probs[token_ids, slots, None])
        shared = silu_mlp(
            x,
            self.weight(p + "shared_expert.gate_proj.weight"),
            self.weight(p + "shared_expert.up_proj.weight"),
            self.weight(p + "shared_expert.down_proj.weight"),
        )
        shared_gate = torch.sigmoid(x @ self.weight(p + "shared_expert_gate.weight").T)
        return routed + shared_gate * shared

    def expert_weight(self, i: int, expert: int, projection: str) -> torch.Tensor:
        name = f"{self.PREFIX}layers.{i}.mlp.experts.{expert}.{projection}_proj.weight"
        return dequantize_fp8_block(
            self.checkpoint.raw(name),
            self.checkpoint.raw(name + "_scale_inv"),
            self.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: list,
        recurrent: bool = False,
    ) -> torch.Tensor:
        """Final-norm hidden states (the lm_head input) of the new rows x [T, hidden].

        cache holds every earlier row; recurrent picks the token-by-token gated delta
        rule instead of the chunked one.
        """
        for i, kind in enumerate(self.layer_types):
            p = f"layers.{i}."
            normed = rms_norm(x, self.weight(p + "input_layernorm.weight"), self.eps)
            if kind == "linear_attention":
                x = x + self.linear_attention(i, normed, cache[i], recurrent)
            else:
                x = x + self.full_attention(i, normed, positions, cache[i])
            normed = rms_norm(
                x, self.weight(p + "post_attention_layernorm.weight"), self.eps
            )
            x = x + self.moe(i, normed)
        return rms_norm(x, self.weight("norm.weight"), self.eps)

    def lm_head_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.checkpoint.rows("lm_head.weight", token_ids, self.dtype)


# --------------------------------------------------------------------------------------
# Codebook tower (AudioDecoder)
# --------------------------------------------------------------------------------------


class Tower:
    """The Drama 3 codebook tower: shared enrichment layers, then one residual layer
    and head per codebook 1..num_codebooks-1 (audio_decoder.py, decoder_layers.py)."""

    def __init__(self, checkpoint: Checkpoint, config: dict, dtype: torch.dtype):
        decoder = config["audio_decoder_config"]
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.eps = decoder["norm_eps"]
        self.head_count = decoder["n_head"]
        self.kv_head_count = decoder["n_local_heads"]
        self.head_dim = decoder["head_dim"]
        self.rotary_dim = int(self.head_dim * decoder["rotary_percent"])
        self.rope_base = decoder["rope_base"]
        self.codebook_size = decoder["vocab_size"]
        self.num_codebooks = decoder["num_codebooks"]
        self.backbone_codebooks = decoder["vq_backbone_codebooks"]
        self.semantic_begin = config["semantic_start_token_id"]
        self.semantic_end = config["semantic_end_token_id"]
        shared = decoder["shared_enrichment_layers"]
        # (checkpoint prefix, token embedding name) of every layer, shared ones first.
        self.layers = [
            (f"audio_decoder_mtp_shared.layers.{i}.", "semantic_embedding")
            for i in range(shared)
        ] + [
            (f"audio_decoder_mtp.layers.{r}.", "codebook_embedding")
            for r in range(self.num_codebooks - 1)
        ]
        self.shared_layer_count = shared
        self.upcast: dict[str, torch.Tensor] = {}

    def weight(self, name: str) -> torch.Tensor:
        """Upcast once and keep (~4 GiB in float32): the tower runs once per frame."""
        if name not in self.upcast:
            self.upcast[name] = self.checkpoint.get(name, self.dtype)
        return self.upcast[name]

    def new_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin) & (token_ids <= self.semantic_end)

    def shared_input(self, input_rows: torch.Tensor) -> torch.Tensor:
        """Codebooks backbone_codebooks+1 .. num_codebooks-1 of semantic input rows,
        summed (AudioDecoder.shared_input); zero for other rows."""
        codebooks = range(1 + self.backbone_codebooks, self.num_codebooks)
        table_rows = torch.stack(
            [
                input_rows[:, 1 + c].clamp(0, self.codebook_size - 1)
                + c * self.codebook_size
                for c in codebooks
            ],
            dim=1,
        )
        summed = self.checkpoint.rows(
            "audio_decoder.codebook_embeddings.weight", table_rows, self.dtype
        ).sum(dim=1)
        return torch.where(self.is_semantic(input_rows[:, 0]).unsqueeze(-1), summed, 0.0)

    def attention(
        self, p: str, x: torch.Tensor, positions: torch.Tensor, cache: KVCache
    ) -> torch.Tensor:
        """DecoderSelfAttention. Checkpoint wqkv rows are [query | gate | key | value]
        (weights.interleave_query_and_gate only regroups them for the fused kernel)."""
        rows = x.shape[0]
        query_width = self.head_count * self.head_dim
        kv_width = self.kv_head_count * self.head_dim
        query, gate, key, value = (
            x @ self.weight(p + "transformer_layer.attention.wqkv.weight").T
        ).split([query_width, query_width, kv_width, kv_width], dim=-1)
        query = rms_norm(
            query.view(rows, self.head_count, self.head_dim),
            self.weight(p + "transformer_layer.attention.q_norm.weight"),
            self.eps,
        )
        key = rms_norm(
            key.view(rows, self.kv_head_count, self.head_dim),
            self.weight(p + "transformer_layer.attention.k_norm.weight"),
            self.eps,
        )
        value = value.view(rows, self.kv_head_count, self.head_dim)
        cos, sin = rotary_cos_sin(positions, self.rotary_dim, self.rope_base, self.dtype)
        query, key = rotate_interleaved(query, cos, sin), rotate_interleaved(key, cos, sin)
        keys, values = cache.append(key, value)
        attended = causal_gqa_attention(query, keys, values, self.head_dim**-0.5)
        gated = attended.reshape(rows, -1) * torch.sigmoid(gate)
        return gated @ self.weight(p + "transformer_layer.attention.wo.weight").T

    def layer(
        self,
        index: int,
        hidden: torch.Tensor,
        code_ids: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor | None,
        positions: torch.Tensor,
        cache: KVCache,
    ) -> torch.Tensor:
        """DecoderLayer.forward: mix a code embedding into the hidden, then a pre-norm
        attention + SiLU MLP block; the layer returns its final-norm output."""
        p, embedding_name = self.layers[index]
        embedding = self.checkpoint.rows(
            p + embedding_name + ".weight",
            code_ids.clamp(0, self.codebook_size - 1),
            self.dtype,
        )
        embedding = torch.where(label_is_semantic.unsqueeze(-1), embedding, 0.0)
        if shared_input is not None:
            embedding = embedding + shared_input
        mixed = torch.cat(
            (
                rms_norm(embedding, self.weight(p + "enorm.weight"), self.eps),
                rms_norm(hidden, self.weight(p + "hnorm.weight"), self.eps),
            ),
            dim=-1,
        ) @ self.weight(p + "eh_proj.weight").T
        attended = self.attention(
            p,
            rms_norm(
                mixed, self.weight(p + "transformer_layer.attention_norm.weight"), self.eps
            ),
            positions,
            cache,
        )
        residual = attended + mixed
        feed_forward = silu_mlp(
            rms_norm(
                residual, self.weight(p + "transformer_layer.ffn_norm.weight"), self.eps
            ),
            self.weight(p + "transformer_layer.feed_forward.w1.weight"),
            self.weight(p + "transformer_layer.feed_forward.w3.weight"),
            self.weight(p + "transformer_layer.feed_forward.w2.weight"),
        )
        return rms_norm(
            feed_forward + residual, self.weight(p + "final_layernorm.weight"), self.eps
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        input_rows: torch.Tensor,
        label_rows: torch.Tensor,
        cache: list[KVCache],
        with_logits: bool,
    ) -> torch.Tensor | None:
        """Run every tower layer on new rows, extending each layer's KV cache.

        hidden: backbone final-norm output at these rows; input_rows: the frame rows
        at these positions; label_rows: the frame that follows each row (the one
        being predicted). Shared layers read codebook 0 of the label, residual layer
        r reads codebook r and its head predicts codebook r + 1. Returns raw logits
        [rows, num_codebooks - 1, codebook_size] when with_logits.
        """
        label_is_semantic = self.is_semantic(label_rows[:, 0])
        shared_input = self.shared_input(input_rows)
        for index in range(self.shared_layer_count):
            hidden = self.layer(
                index,
                hidden,
                label_rows[:, 1],
                label_is_semantic,
                shared_input if index == 0 else None,
                positions,
                cache[index],
            )
        logits = []
        for r in range(self.num_codebooks - 1):
            index = self.shared_layer_count + r
            hidden = self.layer(
                index,
                hidden,
                label_rows[:, 1 + r],
                label_is_semantic,
                None,
                positions,
                cache[index],
            )
            if with_logits:
                head = self.weight(f"audio_decoder_mtp.heads.{r}.weight")
                logits.append(hidden @ head.T)
        return torch.stack(logits, dim=1) if with_logits else None


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------


@dataclass
class Case:
    directory: str
    meta: dict
    prompt_rows: torch.Tensor  # [P, 1 + codebooks]
    frames: torch.Tensor  # [F, 1 + codebooks]
    rows: torch.Tensor = field(init=False)  # prompt rows then frames

    def __post_init__(self) -> None:
        self.rows = torch.cat((self.prompt_rows, self.frames))


def load_case(directory: str, config: dict) -> Case:
    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)
    width = 1 + config["audio_decoder_config"]["num_codebooks"]
    prompt_rows = torch.tensor(meta["prompt_rows"], dtype=torch.long).view(-1, width)
    frames = torch.tensor(meta.get("frames", []), dtype=torch.long).view(-1, width)
    expected = {
        "hidden_size": config["text_config"]["hidden_size"],
        "codebook_size": config["audio_decoder_config"]["vocab_size"],
        "num_codebooks": config["audio_decoder_config"]["num_codebooks"],
        "eos": config["eos_token_id"],
        "semantic_begin": config["semantic_start_token_id"],
        "shared_layers": config["audio_decoder_config"]["shared_enrichment_layers"],
        "residual_layers": config["audio_decoder_config"]["num_codebooks"] - 1,
    }
    for key, value in expected.items():
        if key in meta and meta[key] != value:
            raise ValueError(f"meta.json {key}={meta[key]} but the checkpoint has {value}")
    if prompt_rows.shape[0] == 0:
        raise ValueError("meta.json has no prompt rows")
    return Case(directory=directory, meta=meta, prompt_rows=prompt_rows, frames=frames)


def candidate_token_ids(config: dict) -> torch.Tensor:
    """Main-token candidates in output order: eos, then the semantic band
    (AudioDecoder.main_token_candidate_ids; eos sorts below the band)."""
    begin, end = config["semantic_start_token_id"], config["semantic_end_token_id"]
    ids = sorted({config["eos_token_id"], *range(begin, end + 1)})
    if ids[0] != config["eos_token_id"]:
        raise ValueError("eos is not below the semantic band; output order would differ")
    return torch.tensor(ids)


def load_model(model_dir: str, dtype: torch.dtype) -> tuple[dict, Backbone, Tower]:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    checkpoint = Checkpoint(model_dir)
    return config, Backbone(checkpoint, config, dtype), Tower(checkpoint, config, dtype)


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


@torch.no_grad()
def command_run(args: argparse.Namespace) -> None:
    dtype = getattr(torch, args.dtype)
    config, backbone, tower = load_model(args.model_dir, dtype)
    case = load_case(args.case_dir, config)
    prompt_count, frame_count = case.prompt_rows.shape[0], case.frames.shape[0]
    total = prompt_count + frame_count
    timings: dict[str, float] = {}
    print(f"case {args.case_dir}: {prompt_count} prompt rows + {frame_count} frames")

    # 1. One causal backbone forward over prompt rows + teacher-forced frames.
    started = time.time()
    positions = torch.arange(total)
    hidden = backbone.forward(
        backbone.embed(case.rows), positions, backbone.new_cache(), recurrent=False
    )
    timings["backbone_s"] = time.time() - started
    print(f"backbone forward {timings['backbone_s']:.1f}s")

    # 2. Main-token logits over [eos, semantic band] at the row before each frame.
    candidates = candidate_token_ids(config)
    decode_rows = torch.arange(prompt_count - 1, prompt_count - 1 + frame_count)
    semantic = hidden[decode_rows] @ backbone.lm_head_rows(candidates).T

    # 3. Tower: prefill over the prompt rows whose next row is still prompt (labels =
    #    next row, model_inputs.next_frame_labels), then one decode step per frame at
    #    the row before it, teacher-forced with that frame's codes.
    started = time.time()
    tower_cache = tower.new_cache()
    if prompt_count > 1:
        prefill = torch.arange(prompt_count - 1)
        tower.forward(
            hidden[prefill],
            prefill,
            case.rows[prefill],
            case.rows[prefill + 1],
            tower_cache,
            with_logits=False,
        )
    residual = []
    for t in range(frame_count):
        row = prompt_count - 1 + t
        step = torch.tensor([row])
        residual.append(
            tower.forward(
                hidden[step], step, case.rows[step], case.frames[t : t + 1],
                tower_cache, with_logits=True,
            )[0]
        )
    timings["tower_s"] = time.time() - started
    print(f"tower {timings['tower_s']:.1f}s")

    def write(name: str, tensor: torch.Tensor) -> None:
        array = tensor.to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
        array.tofile(os.path.join(args.case_dir, name))

    hidden_size = config["text_config"]["hidden_size"]
    codebook_size = config["audio_decoder_config"]["vocab_size"]
    residual_count = config["audio_decoder_config"]["num_codebooks"] - 1
    write("ref_hidden.f32", hidden)
    write("ref_semantic.f32", semantic.reshape(frame_count, len(candidates)))
    write(
        "ref_residual.f32",
        torch.stack(residual)
        if residual
        else torch.zeros(0, residual_count, codebook_size),
    )
    meta = {
        "reference": "tools/reference/drama3_reference.py",
        "model_dir": args.model_dir,
        "compute_dtype": args.dtype,
        "outputs": {
            "ref_hidden.f32": [total, hidden_size],
            "ref_semantic.f32": [frame_count, len(candidates)],
            "ref_residual.f32": [frame_count, residual_count, codebook_size],
        },
        "semantic_candidates": "eos, then semantic_begin + 0 .. semantic_begin + 4095",
        "prompt_rows": prompt_count,
        "frames": frame_count,
        "timings": timings,
        "peak_rss_gib": round(peak_rss_gib(), 2),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "router_near_ties": backbone.near_ties(),
        "notes": [
            "All weights upcast to the compute dtype; every op runs in it. BF16 "
            "weights and FP8 E4M3 experts (x 128x128 scale_inv blocks) are "
            "dequantized exactly; no activation quantization is emulated.",
            "Backbone: one causal forward over prompt rows + frames, chunked gated "
            "delta rule (chunk 64).",
            "Rotary angles are formed in float64 and cast to the compute dtype.",
            "router_near_ties: smallest gaps between the 8th and 9th router "
            "probabilities. A gap near the working precision is a tie any "
            "implementation may break either way; differences from that row on "
            "are then expected.",
            "Tower: prefill over prompt rows 0..P-2 (label = next row), then one step "
            "per frame at row P-1+t with label frames[t]; equivalent to the official "
            "prefill whose last row gets a placeholder label that the same "
            "forward's decode step overwrites.",
        ],
    }
    with open(os.path.join(args.case_dir, "ref_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in ("timings", "peak_rss_gib")}))


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual - expected).abs()
    return {
        "max_abs": float(diff.max()),
        "max_abs_over_max_ref": float(diff.max() / expected.abs().max()),
        "rel_l2": float(diff.norm() / expected.norm()),
    }


@torch.no_grad()
def command_check_steps(args: argparse.Namespace) -> None:
    """Full causal forward (chunked GDN, batched attention, batched tower) against
    token-by-token decoding (recurrent GDN, conv / KV caches, one tower row a step)."""
    dtype = getattr(torch, args.dtype)
    config, backbone, tower = load_model(args.model_dir, dtype)
    case = load_case(args.case_dir, config)
    rows = case.rows[: args.rows]
    count = rows.shape[0]
    positions = torch.arange(count)
    embedded = backbone.embed(rows)

    started = time.time()
    full = backbone.forward(embedded, positions, backbone.new_cache(), recurrent=False)
    full_s = time.time() - started
    near_ties = backbone.near_ties(5)
    started = time.time()
    cache = backbone.new_cache()
    stepped = torch.cat(
        [
            backbone.forward(embedded[i : i + 1], positions[i : i + 1], cache, True)
            for i in range(count)
        ]
    )
    step_s = time.time() - started

    # Tower: all labelled rows at once vs one row at a time.
    labelled = torch.arange(count - 1)
    batched = tower.forward(
        full[labelled], labelled, rows[labelled], rows[labelled + 1],
        tower.new_cache(), with_logits=True,
    )
    tower_cache = tower.new_cache()
    single = torch.cat(
        [
            tower.forward(
                full[i : i + 1], positions[i : i + 1], rows[i : i + 1],
                rows[i + 1 : i + 2], tower_cache, with_logits=True,
            )
            for i in range(count - 1)
        ]
    )
    row_error = (stepped - full).norm(dim=-1) / full.norm(dim=-1)
    diverged = (row_error > 1e-4).nonzero()
    report = {
        "rows": count,
        "dtype": args.dtype,
        "backbone_full_s": round(full_s, 1),
        "backbone_stepwise_s": round(step_s, 1),
        "hidden_full_vs_stepwise": relative_error(stepped, full),
        "hidden_worst_row": int(row_error.argmax()),
        "first_row_rel_error_above_1e-4": int(diverged[0]) if len(diverged) else None,
        "router_near_ties_full_forward": near_ties,
        "tower_logits_batched_vs_stepwise": relative_error(single, batched),
        "peak_rss_gib": round(peak_rss_gib(), 2),
    }
    print(json.dumps(report, indent=2))


@torch.no_grad()
def command_check_gdn(args: argparse.Namespace) -> None:
    """Chunked vs recurrent gated delta rule on random inputs (no checkpoint needed)."""
    generator = torch.Generator().manual_seed(0)
    report = {}
    for dtype in (torch.float32, torch.float64):
        length, heads, key_dim, value_dim = 200, 4, 128, 128

        def rand(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, dtype=torch.float64).to(dtype)

        query = l2_normalize(rand(length, heads, key_dim)) * key_dim**-0.5
        key = l2_normalize(rand(length, heads, key_dim))
        value = rand(length, heads, value_dim)
        g = -F.softplus(rand(length, heads))  # decays in (0, 1)
        beta = torch.sigmoid(rand(length, heads))
        state = rand(heads, key_dim, value_dim) * 0.1
        expected, expected_state = gated_delta_rule_recurrent(
            query, key, value, g, beta, state
        )
        actual, actual_state = gated_delta_rule_chunked(
            query, key, value, g, beta, state
        )
        report[str(dtype)] = {
            "output": relative_error(actual, expected),
            "final_state": relative_error(actual_state, expected_state),
        }
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0: default)")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="write ref_*.f32 into a dump directory")
    run.add_argument("case_dir")
    run.set_defaults(handler=command_run)
    steps = commands.add_parser("check-steps", help="full forward vs stepwise decode")
    steps.add_argument("case_dir")
    steps.add_argument("--rows", type=int, default=24, help="leading rows to compare")
    steps.set_defaults(handler=command_check_steps)
    gdn = commands.add_parser("check-gdn", help="chunked vs recurrent delta rule")
    gdn.set_defaults(handler=command_check_gdn)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
