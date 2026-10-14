#!/usr/bin/env python3
"""Compare two sets of teacher-forced Drama 3 outputs, per decode step.

A source is DIR:KIND. KIND is
  port  the port's teacher dump: hidden_decode.f32 [>=F, 2048] (hidden at row P-1+t),
        semantic.f32 [>=F, 4097], residual.f32 [F, 9, 4096]; extra trailing decode
        rows (a step after the last frame) are ignored
  ref   drama3_reference.py run:                      ref_hidden / ref_semantic / ...
  off   drama3_reference.py run --emulate official:   off_hidden / off_semantic / ...
The second source is the baseline. meta.json (prompt rows, frames) is read from the
first directory that has one.

  compare_dump.py runs/tf/fp8/en_short:port runs/tf/f16/en_short:off

Prints JSON: per-step hidden cosine and relative L2 (min / mean / max), semantic
top-1 agreement, top-10 overlap and relative L2, residual top-1 agreement per book
and relative L2. When both sources have a router table (ref / off), also the rows
and layers whose top-8 expert sets differ, and the baseline's closest router ties.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


def load(path: str, *shape: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(-1, *shape)


def stats(values: np.ndarray) -> dict:
    return {"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())}


def rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


class Source:
    def __init__(self, spec: str, prompt_count: int, frame_count: int) -> None:
        self.directory, self.kind = spec.rsplit(":", 1)
        rows = slice(0, frame_count)
        if self.kind == "port":
            self.hidden = load(self.path("hidden_decode.f32"), 2048)[rows]
            self.semantic = load(self.path("semantic.f32"), 4097)[rows]
            self.residual = load(self.path("residual.f32"), 9, 4096)[rows]
            self.router = None
        elif self.kind in ("ref", "off"):
            prefix = self.kind + "_"
            hidden = load(self.path(prefix + "hidden.f32"), 2048)
            self.hidden = hidden[prompt_count - 1 : prompt_count - 1 + frame_count]
            self.semantic = load(self.path(prefix + "semantic.f32"), 4097)
            self.residual = load(self.path(prefix + "residual.f32"), 9, 4096)
            router_path = self.path(prefix + "router.i16")
            self.router = (
                np.fromfile(router_path, dtype="<i2").reshape(-1, len(hidden), 8)
                if os.path.exists(router_path)
                else None
            )
            meta_path = self.path(prefix + "meta.json")
            with open(meta_path) as f:
                self.near_ties = json.load(f).get("router_near_ties", [])
        else:
            raise ValueError(f"unknown kind {self.kind!r} in {spec!r}")

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)


def case_meta(specs: list[str]) -> dict:
    for spec in specs:
        path = os.path.join(spec.rsplit(":", 1)[0], "meta.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("no meta.json in either directory")


def compare(spec: str, baseline_spec: str) -> dict:
    meta = case_meta([spec, baseline_spec])
    prompt_count, frame_count = len(meta["prompt_rows"]), len(meta["frames"])
    a = Source(spec, prompt_count, frame_count)
    b = Source(baseline_spec, prompt_count, frame_count)

    hidden_cosine = (a.hidden * b.hidden).sum(-1) / (
        np.linalg.norm(a.hidden, axis=-1) * np.linalg.norm(b.hidden, axis=-1)
    )
    top10_a = np.argsort(-a.semantic, axis=-1)[:, :10]
    top10_b = np.argsort(-b.semantic, axis=-1)[:, :10]
    top10_overlap = np.mean(
        [len(set(x) & set(y)) / 10 for x, y in zip(top10_a, top10_b)]
    )
    per_book = (a.residual.argmax(-1) == b.residual.argmax(-1)).mean(axis=0)
    report = {
        "compare": spec,
        "baseline": baseline_spec,
        "steps": frame_count,
        "hidden_cosine": stats(hidden_cosine),
        "hidden_rel_l2": stats(rel_l2(a.hidden, b.hidden)),
        "semantic_top1_agreement": float(
            (a.semantic.argmax(-1) == b.semantic.argmax(-1)).mean()
        ),
        "semantic_top10_overlap": float(top10_overlap),
        "semantic_rel_l2": stats(rel_l2(a.semantic, b.semantic)),
        "residual_top1_agreement_per_book": [round(float(x), 4) for x in per_book],
        "residual_top1_agreement_min": float(per_book.min()),
        "residual_rel_l2": stats(rel_l2(a.residual, b.residual)),
    }
    if a.router is not None and b.router is not None:
        differs = (a.router != b.router).any(axis=-1)  # [layers, rows]
        layers, rows = differs.nonzero()
        flipped = set(zip(rows.tolist(), layers.tolist()))
        report["router"] = {
            "rows_with_a_different_expert_set": int(differs.any(axis=0).sum()),
            "row_layer_pairs_different": int(differs.sum()),
            "first_different": (
                {"row": int(rows.min()), "layers": sorted(layers[rows == rows.min()].tolist())}
                if len(rows)
                else None
            ),
            "baseline_near_ties": [
                {**tie, "flipped": (tie["row"], tie["layer"]) in flipped}
                for tie in b.near_ties[:5]
            ],
        }
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=2))
