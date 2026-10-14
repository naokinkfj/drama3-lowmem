# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""Compare teacher-forced Drama 3 dumps position by position.

Each case directory holds meta.json plus float32 files written by the C++
runtime (hidden_decode, semantic, residual, hidden_prefill) or by
tools/reference (ref_hidden, ref_semantic, ref_residual). Both sides must be
forced onto the same frames.

  python drama3_compare_dumps.py <case-dir-A> <case-dir-B> [--a-kind port|ref] [--b-kind port|ref]
  python drama3_compare_dumps.py <case-dir> --self   # decode vs prefill inside one port dump
"""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("a", type=Path)
parser.add_argument("b", type=Path, nargs="?")
parser.add_argument("--a-kind", choices=("port", "ref"), default="port")
parser.add_argument("--b-kind", choices=("port", "ref"), default="port")
parser.add_argument("--self", action="store_true", help="compare a port dump's decode hidden with its own prefill")
parser.add_argument("--json", type=Path, help="also write the summary here")
arguments = parser.parse_args()


def load(case, kind):
    meta = json.loads((case / "meta.json").read_text())
    width, books, size = meta["hidden_size"], meta["residual_layers"], meta["codebook_size"]
    prompt, frames = len(meta["prompt_rows"]), len(meta["frames"])
    if kind == "port":
        hidden = np.fromfile(case / "hidden_decode.f32", dtype=np.float32).reshape(-1, width)
        semantic = np.fromfile(case / "semantic.f32", dtype=np.float32).reshape(-1, size + 1)
        residual = np.fromfile(case / "residual.f32", dtype=np.float32).reshape(-1, books, size)
    else:
        full = np.fromfile(case / "ref_hidden.f32", dtype=np.float32).reshape(-1, width)
        hidden = full[prompt - 1: prompt - 1 + frames]
        semantic = np.fromfile(case / "ref_semantic.f32", dtype=np.float32).reshape(-1, size + 1)
        residual = np.fromfile(case / "ref_residual.f32", dtype=np.float32).reshape(-1, books, size)
    return meta, hidden, semantic, residual


def cosine(x, y):
    return np.sum(x * y, axis=-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-30)


def relative_l2(x, y):
    return np.linalg.norm(x - y, axis=-1) / (np.linalg.norm(y, axis=-1) + 1e-30)


def topk_overlap(x, y, k):
    top_x = np.argsort(-x, axis=-1)[..., :k]
    top_y = np.argsort(-y, axis=-1)[..., :k]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(top_x.reshape(-1, k), top_y.reshape(-1, k))])


def stats(values):
    return {"min": float(np.min(values)), "mean": float(np.mean(values)), "max": float(np.max(values))}


summary = {}
if arguments.self:
    meta, hidden, _, _ = load(arguments.a, "port")
    prompt = len(meta["prompt_rows"])
    prefill = np.fromfile(arguments.a / "hidden_prefill.f32", dtype=np.float32).reshape(-1, meta["hidden_size"])
    aligned = prefill[prompt - 1: prompt - 1 + hidden.shape[0]]
    summary["decode_vs_prefill_hidden_cosine"] = stats(cosine(hidden, aligned))
    summary["decode_vs_prefill_hidden_rel_l2"] = stats(relative_l2(hidden, aligned))
else:
    meta_a, hidden_a, semantic_a, residual_a = load(arguments.a, arguments.a_kind)
    meta_b, hidden_b, semantic_b, residual_b = load(arguments.b, arguments.b_kind)
    if meta_a["frames"] != meta_b["frames"] or meta_a["prompt_rows"] != meta_b["prompt_rows"]:
        raise SystemExit("dumps were not forced onto the same prompt and frames")
    steps = min(len(hidden_a), len(hidden_b))
    hidden_a, hidden_b = hidden_a[:steps], hidden_b[:steps]
    semantic_a, semantic_b = semantic_a[:steps], semantic_b[:steps]
    residual_a, residual_b = residual_a[:steps], residual_b[:steps]
    summary["steps"] = steps
    summary["hidden_cosine"] = stats(cosine(hidden_a, hidden_b))
    summary["hidden_rel_l2"] = stats(relative_l2(hidden_a, hidden_b))
    summary["semantic_top1_agreement"] = float(np.mean(np.argmax(semantic_a, -1) == np.argmax(semantic_b, -1)))
    summary["semantic_top10_overlap"] = float(topk_overlap(semantic_a, semantic_b, 10))
    summary["semantic_rel_l2"] = stats(relative_l2(semantic_a, semantic_b))
    agreement = np.mean(np.argmax(residual_a, -1) == np.argmax(residual_b, -1), axis=0)
    summary["residual_top1_agreement_per_book"] = [round(float(x), 4) for x in agreement]
    summary["residual_top1_agreement_min"] = float(np.min(agreement))
    summary["residual_rel_l2"] = stats(relative_l2(residual_a, residual_b))
print(json.dumps(summary, indent=1))
if arguments.json:
    arguments.json.write_text(json.dumps(summary, indent=1))
