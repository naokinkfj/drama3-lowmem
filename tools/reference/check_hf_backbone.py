#!/usr/bin/env python3
"""Cross-check the reference backbone against Hugging Face transformers' Qwen3.5-MoE.

Builds transformers' Qwen3_5MoeTextModel with the first --layers layers of the Drama 3
backbone (the default 4 covers three Gated DeltaNet layers and the first gated
full-attention layer, each with its MoE), loads the same checkpoint weights (routed
experts dequantized to float32), feeds both models the same frame embeddings and
compares the final-norm hidden states. Needs `pip install "transformers>=5"` and about
8 GiB of RAM per layer for the dense float32 experts (4 layers peaked at 31 GiB).

  check_hf_backbone.py <case_dir> [--layers 4] [--rows 80]
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

from drama3_reference import (
    DEFAULT_MODEL_DIR,
    load_case,
    load_model,
    relative_error,
)

TEXT_CONFIG_KEYS = (
    "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
    "hidden_act", "rms_norm_eps", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_value_head_dim", "linear_num_key_heads", "linear_num_value_heads",
    "num_experts", "num_experts_per_tok", "moe_intermediate_size",
    "shared_expert_intermediate_size", "rope_parameters", "max_position_embeddings",
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case_dir")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rows", type=int, default=80)
    args = parser.parse_args()

    config, backbone, _ = load_model(args.model_dir, torch.float32)
    text = config["text_config"]
    hf_config = Qwen3_5MoeTextConfig(
        **{key: text[key] for key in TEXT_CONFIG_KEYS},
        num_hidden_layers=args.layers,
        layer_types=text["layer_types"][: args.layers],
        vocab_size=16,  # inputs_embeds are passed in; keep the unused table tiny
    )
    hf_config._attn_implementation = "eager"
    model = Qwen3_5MoeTextModel(hf_config).float().eval()

    checkpoint = backbone.checkpoint
    prefix = backbone.PREFIX
    state = {}
    for name in model.state_dict():
        if name == "embed_tokens.weight":
            continue
        layer = int(name.split(".")[1]) if name.startswith("layers.") else None
        if name.endswith("mlp.experts.gate_up_proj"):
            state[name] = torch.stack(
                [
                    torch.cat(
                        [backbone.expert_weight(layer, e, "gate"),
                         backbone.expert_weight(layer, e, "up")]
                    )
                    for e in range(text["num_experts"])
                ]
            )
        elif name.endswith("mlp.experts.down_proj"):
            state[name] = torch.stack(
                [backbone.expert_weight(layer, e, "down")
                 for e in range(text["num_experts"])]
            )
        else:
            state[name] = checkpoint.get(prefix + name, torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"embed_tokens.weight"} or unexpected:
        raise ValueError(f"weight mismatch: missing {missing}, unexpected {unexpected}")

    case = load_case(args.case_dir, config)
    rows = case.rows[: args.rows]
    embedded = backbone.embed(rows)
    started = time.time()
    expected = model(inputs_embeds=embedded[None]).last_hidden_state[0]
    hf_s = time.time() - started

    backbone.layer_types = backbone.layer_types[: args.layers]
    started = time.time()
    actual = backbone.forward(
        embedded, torch.arange(rows.shape[0]), backbone.new_cache(), recurrent=False
    )
    print(json.dumps({
        "layers": args.layers,
        "layer_types": backbone.layer_types,
        "rows": rows.shape[0],
        "hf_transformers_s": round(hf_s, 1),
        "reference_s": round(time.time() - started, 1),
        "reference_vs_transformers": relative_error(actual, expected),
    }, indent=2))


if __name__ == "__main__":
    main()
