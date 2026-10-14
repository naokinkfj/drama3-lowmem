#!/usr/bin/env python3
"""Build a synthetic Drama 3 case directory (meta.json) with the official prompt builder.

The prompt rows come from sglang-omni's own build_prompt_rows
(sglang_omni/models/fishaudio_drama3/prompt.py, plain torch + tokenizers), imported
from a path so this tree does not vendor it. Frames are random but valid semantic
frames [semantic_begin + c0, c0, ..., c9]; they only exercise the teacher-forced code
path, they are not speech.

  make_case.py --official-prompt sglang-omni/.../prompt.py --out runs/ref/smoke \
      --text "Hello from the reference." --frames 8

Also prints the rows and the detokenized text rows, as a check that the prompt
layout is the one expected (reference-bracket-v1, text rows carry zero codes).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch
from tokenizers import Tokenizer


def load_official_prompt_module(path: str):
    spec = importlib.util.spec_from_file_location("drama3_official_prompt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", default="models/drama-3")
    parser.add_argument("--official-prompt", required=True, help="path to official prompt.py")
    parser.add_argument("--out", required=True, help="case directory to create")
    parser.add_argument("--text", default="Hello! This is a short reference test.")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--reference-frames",
        type=int,
        default=0,
        help="add one voice reference with this many random code frames",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    decoder = config["audio_decoder_config"]
    num_codebooks, codebook_size = decoder["num_codebooks"], decoder["vocab_size"]
    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    prompt = load_official_prompt_module(args.official_prompt)
    generator = torch.Generator().manual_seed(args.seed)

    references = []
    if args.reference_frames:
        references.append(
            prompt.ReferencePart(
                transcript="A short reference clip.",
                audio_codes=torch.randint(
                    0, codebook_size, (num_codebooks, args.reference_frames),
                    generator=generator,
                ),
            )
        )
    rows = prompt.build_prompt_rows(
        args.text, references, tokenizer,
        num_codebooks=num_codebooks, codebook_size=codebook_size,
    )
    semantic_begin = config["semantic_start_token_id"]
    if tokenizer.token_to_id(prompt.SEMANTIC_BEGIN_TOKEN) != semantic_begin:
        raise ValueError("tokenizer <|semantic:0|> disagrees with config.json")
    codes = torch.randint(
        0, codebook_size, (args.frames, num_codebooks), generator=generator
    )
    frames = torch.cat((codes[:, :1] + semantic_begin, codes), dim=1)

    os.makedirs(args.out, exist_ok=True)
    meta = {
        "hidden_size": config["text_config"]["hidden_size"],
        "codebook_size": codebook_size,
        "num_codebooks": num_codebooks,
        "eos": config["eos_token_id"],
        "semantic_begin": semantic_begin,
        "shared_layers": decoder["shared_enrichment_layers"],
        "residual_layers": num_codebooks - 1,
        "prompt_rows": rows.tolist(),
        "frames": frames.tolist(),
        "source": "tools/reference/make_case.py (official build_prompt_rows, "
        f"random frames, seed {args.seed})",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)

    is_semantic = (rows[:, 0] >= semantic_begin) & (
        rows[:, 0] <= config["semantic_end_token_id"]
    )
    text_rows = rows[~is_semantic]
    print(f"prompt rows {tuple(rows.shape)}, frames {tuple(frames.shape)} -> {args.out}")
    print(f"text rows {int((~is_semantic).sum())}, all text-row codes zero: "
          f"{bool((text_rows[:, 1:] == 0).all())}")
    if is_semantic.any():
        audio = rows[is_semantic]
        print(f"reference audio rows {audio.shape[0]}, main == semantic_begin + c0: "
              f"{bool((audio[:, 0] == audio[:, 1] + semantic_begin).all())}")
    print("detokenized prompt (semantic rows shown as <|semantic:c0|>):")
    print(repr(tokenizer.decode(rows[:, 0].tolist(), skip_special_tokens=False)))
    print("first rows:", rows[:4].tolist())
    print("last rows:", rows[-3:].tolist())


if __name__ == "__main__":
    main()
