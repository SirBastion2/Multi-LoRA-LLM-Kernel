"""Replace targeted Linear layers in a Hugging Face model with MultiLoRALinear."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from persona.adapter_pool import AdapterPool, LayerPools
from persona.lora_layer import LoRAContext, MultiLoRALinear, make_slot_hook


def _attn_mlp_paths() -> List[Tuple[str, str, str]]:
    """(module_path, a_key, b_key) within one decoder layer."""
    paths = []
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        paths.append((f"self_attn.{name}", f"A_qkv", f"B_{name.split('_')[0]}"))
    # o_proj uses dedicated pools
    paths[3] = ("self_attn.o_proj", "A_o", "B_o")
    for name in ("gate_proj", "up_proj", "down_proj"):
        if name == "gate_proj":
            paths.append((f"mlp.{name}", "A_gu", "B_gate"))
        elif name == "up_proj":
            paths.append((f"mlp.{name}", "A_gu", "B_up"))
        else:
            paths.append((f"mlp.{name}", "A_down", "B_down"))
    return paths


def _module_name_from_path(mod_path: str) -> str:
    return mod_path.rsplit(".", 1)[-1]


def _resolve_a_pool(layer_pools: LayerPools, a_key: str, b_key: str, r: int) -> torch.Tensor:
    """Return a contiguous [S, R, K] A pool for one module.

    Packed pools (A_qkv, A_gu) are sliced along the rank axis. Contiguous copies
    are required by the CUDA BGMV kernels. Callers must load adapters into the
    pool *before* patching so the copies see final weights.
    """
    raw = getattr(layer_pools, a_key)
    if a_key == "A_qkv":
        if b_key == "B_q":
            return raw[:, :r, :].contiguous()
        if b_key == "B_k":
            return raw[:, r : 2 * r, :].contiguous()
        if b_key == "B_v":
            return raw[:, 2 * r : 3 * r, :].contiguous()
        raise ValueError(f"Unexpected b_key for A_qkv: {b_key}")
    if a_key == "A_gu":
        if b_key == "B_gate":
            return raw[:, :r, :].contiguous()
        if b_key == "B_up":
            return raw[:, r : 2 * r, :].contiguous()
        raise ValueError(f"Unexpected b_key for A_gu: {b_key}")
    # A_o / A_down are already [S, R, K]
    return raw if raw.is_contiguous() else raw.contiguous()


def patch_model_with_pools(
    model: nn.Module,
    pool: AdapterPool,
    ctx: LoRAContext,
    target_modules: Optional[Sequence[str]] = None,
) -> nn.Module:
    """
    Walk `model.model.layers` (Mistral-style) and wrap targeted linears.

    Does not load weights; assumes `model` is already on device.
    Load adapters into `pool` before calling this (A_qkv/A_gu slices are copied).
    """
    allow = set(target_modules) if target_modules is not None else None
    layers = model.model.layers
    for layer_idx, block in enumerate(layers):
        lp = pool.layers[layer_idx]
        for mod_path, a_key, b_key in _attn_mlp_paths():
            mod_name = _module_name_from_path(mod_path)
            if allow is not None and mod_name not in allow:
                continue
            parent, attr = _rsplit_once(block, mod_path)
            base_linear = getattr(parent, attr)
            if not isinstance(base_linear, nn.Module):
                continue
            if isinstance(base_linear, MultiLoRALinear):
                continue  # already patched
            a_pool = _resolve_a_pool(lp, a_key, b_key, pool.r)
            b_pool = getattr(lp, b_key)
            wrapped = MultiLoRALinear(base_linear, ctx, a_pool, b_pool)
            _set_module(block, mod_path, wrapped)
    # Avoid stacking duplicate hooks on re-entry
    if not getattr(model, "_persona_slot_hook", False):
        model.register_forward_pre_hook(make_slot_hook(ctx), with_kwargs=True)
        model._persona_slot_hook = True
    return model


def _rsplit_once(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    cur = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    return cur, parts[-1]


def _set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent, attr = _rsplit_once(root, path)
    setattr(parent, attr, module)


def load_bitsandbytes_model_skeleton() -> Dict[str, Any]:
    """
    Document the intended load path (owner runs on GPU with real weights).

    Returns kwargs for `AutoModelForCausalLM.from_pretrained` — not executed here.
    """
    return {
        "quantization_config_note": "BitsAndBytesConfig load_in_4bit=True, nf4, compute_dtype=float16",
        "device_map": "auto",
        "torch_dtype": "float16",
    }
