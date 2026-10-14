# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "safetensors>=0.5", "torch>=2.6"]
# ///
"""Measure what storing Drama 3's scaled FP8 experts as F16 or BF16 loses.

The reference is the exact product FP8 value x block scale computed in FP32.
For every routed expert matrix this reports, per storage type, the largest
relative error, the relative Frobenius error, and how many non-zero values
underflow to zero, land in the subnormal range, or overflow.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True, help="JSON summary")
parser.add_argument("--threads", type=int, default=16)
arguments = parser.parse_args()
torch.set_num_threads(arguments.threads)

weight_map = json.loads((arguments.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
shards = {name: safe_open(arguments.checkpoint / name, framework="pt", device="cpu") for name in dict.fromkeys(weight_map.values())}
expert_names = [name for name in weight_map if ".experts." in name and not name.endswith("_scale_inv")]

SMALLEST_NORMAL = {torch.float16: torch.finfo(torch.float16).tiny, torch.bfloat16: torch.finfo(torch.bfloat16).tiny}
totals = {
    dtype_name: {"elements": 0, "nonzero": 0, "underflow_to_zero": 0, "subnormal": 0, "overflow": 0,
                 "max_relative_error": 0.0, "max_relative_error_tensor": None,
                 "max_frobenius_relative_error": 0.0, "max_frobenius_tensor": None,
                 "sum_squared_error": 0.0, "sum_squared_reference": 0.0}
    for dtype_name in ("f16", "bf16")
}
non_fp8 = []
for index, name in enumerate(expert_names):
    tensor = shards[weight_map[name]].get_tensor(name)
    if tensor.dtype != torch.float8_e4m3fn:
        non_fp8.append((name, str(tensor.dtype)))
        continue
    scale = shards[weight_map[name + "_scale_inv"]].get_tensor(name + "_scale_inv").float()
    reference = tensor.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:tensor.shape[0], :tensor.shape[1]]
    nonzero = reference != 0
    for dtype_name, dtype in (("f16", torch.float16), ("bf16", torch.bfloat16)):
        stored = reference.to(dtype)
        restored = stored.float()
        total = totals[dtype_name]
        total["elements"] += reference.numel()
        total["nonzero"] += int(nonzero.sum())
        total["underflow_to_zero"] += int((nonzero & (restored == 0)).sum())
        total["subnormal"] += int(((restored != 0) & (restored.abs() < SMALLEST_NORMAL[dtype])).sum())
        total["overflow"] += int(torch.isinf(restored).sum())
        error = restored - reference
        relative = (error.abs() / reference.abs())[nonzero]
        largest = float(relative.max()) if relative.numel() else 0.0
        if largest > total["max_relative_error"]:
            total["max_relative_error"], total["max_relative_error_tensor"] = largest, name
        squared_error = float(error.double().pow(2).sum())
        squared_reference = float(reference.double().pow(2).sum())
        frobenius = (squared_error / squared_reference) ** 0.5 if squared_reference else 0.0
        if frobenius > total["max_frobenius_relative_error"]:
            total["max_frobenius_relative_error"], total["max_frobenius_tensor"] = frobenius, name
        total["sum_squared_error"] += squared_error
        total["sum_squared_reference"] += squared_reference
    if index % 1000 == 0:
        print(f"{index + 1}/{len(expert_names)} {name}", flush=True)

for total in totals.values():
    total["overall_frobenius_relative_error"] = (total.pop("sum_squared_error") / total.pop("sum_squared_reference")) ** 0.5
summary = {"checkpoint": str(arguments.checkpoint), "expert_matrices": len(expert_names), "non_fp8_experts": non_fp8, "storage": totals}
arguments.output.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
