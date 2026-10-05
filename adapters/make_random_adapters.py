#!/usr/bin/env python3
"""Create N random PEFT-shaped adapter directories for pool scaling tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

NUM_LAYERS = 32
HIDDEN = 4096
KV = 1024
MLP = 14336

MODULE_SHAPES = {
    "q_proj": (HIDDEN, HIDDEN),
    "k_proj": (HIDDEN, KV),
    "v_proj": (HIDDEN, KV),
    "o_proj": (HIDDEN, HIDDEN),
    "gate_proj": (HIDDEN, MLP),
    "up_proj": (HIDDEN, MLP),
    "down_proj": (MLP, HIDDEN),
}


def adapter_keys(layer: int, module: str, r: int) -> tuple[str, str]:
    base = f"base_model.model.model.layers.{layer}."
    if module in ("q_proj", "k_proj", "v_proj", "o_proj"):
        prefix = base + f"self_attn.{module}"
    else:
        prefix = base + f"mlp.{module}"
    return f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"


def write_adapter(out: Path, r: int, target_modules: list[str], seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "r": r,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": target_modules,
        "bias": "none",
        "inference_mode": True,
        "use_rslora": False,
    }
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in range(NUM_LAYERS):
        for mod in target_modules:
            k_in, k_out = MODULE_SHAPES[mod]
            a_key, b_key = adapter_keys(layer, mod, r)
            tensors[a_key] = torch.randn(r, k_in, generator=g, dtype=torch.float16) * 0.01
            tensors[b_key] = torch.randn(k_out, r, generator=g, dtype=torch.float16) * 0.01
    save_file(tensors, out / "adapter_model.safetensors")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("adapters/random"))
    p.add_argument("--num-adapters", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=list(MODULE_SHAPES.keys()),
    )
    args = p.parse_args()
    for i in range(args.num_adapters):
        write_adapter(args.out / f"adapter_{i}", args.rank, args.target_modules, seed=1000 + i)
    print(f"Wrote {args.num_adapters} adapters under {args.out}")


if __name__ == "__main__":
    main()
