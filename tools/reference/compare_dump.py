#!/usr/bin/env python3
"""Compare a port's teacher-forced dump with the reference outputs in the same directory.

Port files read when present (little-endian float32), as the fish_drama3 teacher dump
writes them:
  hidden_prefill.f32 [P+F, 2048]   final-norm hidden of every row
  hidden_decode.f32  [>=F, 2048]   hidden at row P-1+t (the row each decode step reads)
  semantic.f32       [>=F, 4097]   main-token logits, [eos, semantic band]
  residual.f32       [F, 9, 4096]  codebook 1..9 logits
Extra trailing decode rows of the port (a step after the last frame) are ignored.

  compare_dump.py <case_dir> [<case_dir> ...]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


def load(directory: str, name: str, *shape: int) -> np.ndarray | None:
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype="<f4").reshape(-1, *shape)


def summary(port: np.ndarray, ref: np.ndarray) -> dict:
    diff = port - ref
    row_error = np.linalg.norm(diff.reshape(len(ref), -1), axis=1) / np.linalg.norm(
        ref.reshape(len(ref), -1), axis=1
    )
    result = {
        "rows": len(ref),
        "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(ref)),
        "max_abs": float(np.abs(diff).max()),
        "row_rel_l2_p50_p90_max": [
            round(float(x), 5) for x in np.percentile(row_error, [50, 90, 100])
        ],
        "worst_row": int(row_error.argmax()),
    }
    if ref.shape[-1] in (4096, 4097):
        result["argmax_agreement"] = float((port.argmax(-1) == ref.argmax(-1)).mean())
    return result


def compare(directory: str) -> dict:
    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)
    prompt_count, frame_count = len(meta["prompt_rows"]), len(meta["frames"])
    ref_hidden = load(directory, "ref_hidden.f32", 2048)
    if ref_hidden is None:
        raise FileNotFoundError(f"{directory}: no ref_hidden.f32; run drama3_reference.py run first")
    report: dict = {"case": directory, "prompt_rows": prompt_count, "frames": frame_count}
    pairs = [
        ("hidden_prefill.f32", ref_hidden, (2048,)),
        ("hidden_decode.f32", ref_hidden[prompt_count - 1 :], (2048,)),
        ("semantic.f32", load(directory, "ref_semantic.f32", 4097), (4097,)),
        ("residual.f32", load(directory, "ref_residual.f32", 9, 4096), (9, 4096)),
    ]
    for name, ref, shape in pairs:
        port = load(directory, name, *shape)
        if port is None or ref is None:
            continue
        rows = min(len(port), len(ref))
        report[name] = summary(port[:rows], ref[:rows])
    meta_path = os.path.join(directory, "ref_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            report["router_near_ties"] = json.load(f).get("router_near_ties", [])[:3]
    return report


if __name__ == "__main__":
    for case_dir in sys.argv[1:]:
        print(json.dumps(compare(case_dir), indent=2))
