"""PEFT adapter parsing, validation, scale folding, and GPU pool packing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open

# Mistral-7B module geometry (see ARCHITECTURE.md §2.2)
HIDDEN = 4096
KV_OUT = 1024
MLP_INTER = 14336
NUM_LAYERS = 32

ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


@dataclass
class AdapterConfig:
    r: int
    lora_alpha: float
    target_modules: List[str]
    use_rslora: bool = False

    @property
    def scale(self) -> float:
        if self.use_rslora:
            return self.lora_alpha / (self.r ** 0.5)
        return self.lora_alpha / self.r


def _read_adapter_config(path: Path) -> AdapterConfig:
    with open(path / "adapter_config.json") as f:
        raw = json.load(f)
    return AdapterConfig(
        r=int(raw["r"]),
        lora_alpha=float(raw["lora_alpha"]),
        target_modules=list(raw["target_modules"]),
        use_rslora=bool(raw.get("use_rslora", False)),
    )


def _layer_index_from_key(key: str) -> int:
    # base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight
    marker = ".layers."
    i = key.index(marker) + len(marker)
    j = key.index(".", i)
    return int(key[i:j])


def _module_suffix(key: str) -> str:
    if ".self_attn." in key:
        return key.split(".self_attn.")[-1].split(".lora_")[0]
    if ".mlp." in key:
        return key.split(".mlp.")[-1].split(".lora_")[0]
    raise ValueError(f"Unrecognized adapter key: {key}")


def load_peft_adapter_tensors(
    adapter_dir: Path,
    dtype: torch.dtype = torch.float16,
) -> Tuple[AdapterConfig, Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
    """Load one adapter into {layer: {module: (A, B_scaled)}}."""
    cfg = _read_adapter_config(adapter_dir)
    scale = cfg.scale
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        weights_path = adapter_dir / "adapter_model.bin"
        raise FileNotFoundError(
            f"Expected adapter_model.safetensors in {adapter_dir}"
        )

    per_layer: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.endswith(".weight"):
                continue
            if ".lora_A." not in key and ".lora_B." not in key:
                continue
            layer = _layer_index_from_key(key)
            suffix = _module_suffix(key)
            if suffix not in cfg.target_modules:
                continue
            per_layer.setdefault(layer, {})
            ab = per_layer[layer].get(suffix)
            tensor = f.get_tensor(key).to(dtype=dtype)
            if ".lora_A." in key:
                if ab is None:
                    per_layer[layer][suffix] = (tensor, None)
                else:
                    per_layer[layer][suffix] = (tensor, ab[1])
            else:
                b_scaled = tensor * scale
                if ab is None:
                    per_layer[layer][suffix] = (None, b_scaled)
                else:
                    per_layer[layer][suffix] = (ab[0], b_scaled)

    for layer, mods in per_layer.items():
        for suffix, (a, b) in mods.items():
            if a is None or b is None:
                raise ValueError(f"Missing A or B for layer {layer} {suffix}")

    return cfg, per_layer


@dataclass
class LayerPools:
    """Per-decoder-layer GPU pool tensors (one slot dimension S)."""

    A_qkv: torch.Tensor
    B_q: torch.Tensor
    B_k: torch.Tensor
    B_v: torch.Tensor
    A_o: torch.Tensor
    B_o: torch.Tensor
    A_gu: torch.Tensor
    B_gate: torch.Tensor
    B_up: torch.Tensor
    A_down: torch.Tensor
    B_down: torch.Tensor


def _empty_layer_pools(
    slots: int,
    r: int,
    device: torch.device,
    dtype: torch.dtype,
) -> LayerPools:
    return LayerPools(
        A_qkv=torch.zeros(slots, 3 * r, HIDDEN, dtype=dtype, device=device),
        B_q=torch.zeros(slots, HIDDEN, r, dtype=dtype, device=device),
        B_k=torch.zeros(slots, KV_OUT, r, dtype=dtype, device=device),
        B_v=torch.zeros(slots, KV_OUT, r, dtype=dtype, device=device),
        A_o=torch.zeros(slots, r, HIDDEN, dtype=dtype, device=device),
        B_o=torch.zeros(slots, HIDDEN, r, dtype=dtype, device=device),
        A_gu=torch.zeros(slots, 2 * r, HIDDEN, dtype=dtype, device=device),
        B_gate=torch.zeros(slots, MLP_INTER, r, dtype=dtype, device=device),
        B_up=torch.zeros(slots, MLP_INTER, r, dtype=dtype, device=device),
        A_down=torch.zeros(slots, r, MLP_INTER, dtype=dtype, device=device),
        B_down=torch.zeros(slots, HIDDEN, r, dtype=dtype, device=device),
    )


class AdapterPool:
    """Version 1: all adapters resident on GPU; slot table without LRU paging."""

    def __init__(
        self,
        max_slots: int,
        r: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.max_slots = max_slots
        self.r = r
        self.device = device
        self.dtype = dtype
        self.layers: List[LayerPools] = [
            _empty_layer_pools(max_slots, r, device, dtype)
            for _ in range(NUM_LAYERS)
        ]
        self.slot_to_name: Dict[int, str] = {}
        self.name_to_slot: Dict[str, int] = {}
        self._free_slots: List[int] = list(range(max_slots))

    def acquire_slot(self, name: str) -> int:
        if name in self.name_to_slot:
            return self.name_to_slot[name]
        if not self._free_slots:
            raise RuntimeError("AdapterPool: no free slots")
        slot = self._free_slots.pop()
        self.name_to_slot[name] = slot
        self.slot_to_name[slot] = name
        return slot

    def release_slot(self, name: str) -> None:
        slot = self.name_to_slot.pop(name, None)
        if slot is None:
            return
        del self.slot_to_name[slot]
        self._free_slots.append(slot)

    def load_adapter(self, name: str, adapter_dir: Path) -> int:
        cfg, per_layer = load_peft_adapter_tensors(adapter_dir, dtype=self.dtype)
        if cfg.r != self.r:
            raise ValueError(f"Adapter rank {cfg.r} != pool rank {self.r}")
        slot = self.acquire_slot(name)
        for layer_idx in range(NUM_LAYERS):
            mods = per_layer.get(layer_idx, {})
            pools = self.layers[layer_idx]
            self._pack_layer(pools, mods, slot, cfg)
        return slot

    def _pack_layer(
        self,
        pools: LayerPools,
        mods: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        slot: int,
        cfg: AdapterConfig,
    ) -> None:
        r = cfg.r

        def put_a(tensor: torch.Tensor, dest: torch.Tensor, row_offset: int = 0) -> None:
            dest[slot, row_offset : row_offset + tensor.shape[0]].copy_(tensor)

        def put_b(tensor: torch.Tensor, dest: torch.Tensor) -> None:
            dest[slot].copy_(tensor)

        if "q_proj" in mods:
            a_q, b_q = mods["q_proj"]
            put_a(a_q, pools.A_qkv, 0)
            put_b(b_q, pools.B_q)
        if "k_proj" in mods:
            a_k, b_k = mods["k_proj"]
            put_a(a_k, pools.A_qkv, r)
            put_b(b_k, pools.B_k)
        if "v_proj" in mods:
            a_v, b_v = mods["v_proj"]
            put_a(a_v, pools.A_qkv, 2 * r)
            put_b(b_v, pools.B_v)
        if "o_proj" in mods:
            a_o, b_o = mods["o_proj"]
            put_a(a_o, pools.A_o)
            put_b(b_o, pools.B_o)
        if "gate_proj" in mods:
            a_g, b_g = mods["gate_proj"]
            put_a(a_g, pools.A_gu, 0)
            put_b(b_g, pools.B_gate)
        if "up_proj" in mods:
            a_u, b_u = mods["up_proj"]
            put_a(a_u, pools.A_gu, r)
            put_b(b_u, pools.B_up)
        if "down_proj" in mods:
            a_d, b_d = mods["down_proj"]
            put_a(a_d, pools.A_down)
            put_b(b_d, pools.B_down)

    def load_random_adapter(
        self,
        name: str,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Pack random weights (same shapes as real adapters) into a slot."""
        slot = self.acquire_slot(name)
        for layer_idx in range(NUM_LAYERS):
            pools = self.layers[layer_idx]
            for dest in (
                pools.A_qkv,
                pools.B_q,
                pools.B_k,
                pools.B_v,
                pools.A_o,
                pools.B_o,
                pools.A_gu,
                pools.B_gate,
                pools.B_up,
                pools.A_down,
                pools.B_down,
            ):
                dest[slot].copy_(
                    torch.randn(dest[slot].shape, device=self.device, dtype=self.dtype, generator=generator)
                    * 0.01
                )
        return slot


def validate_uniform_adapters(adapter_dirs: Sequence[Path]) -> AdapterConfig:
    """Ensure all adapters share r, alpha, and target_modules."""
    configs = [_read_adapter_config(p) for p in adapter_dirs]
    first = configs[0]
    for c in configs[1:]:
        if c.r != first.r or c.lora_alpha != first.lora_alpha:
            raise ValueError("Adapters must share r and lora_alpha")
        if sorted(c.target_modules) != sorted(first.target_modules):
            raise ValueError("Adapters must share target_modules")
    return first
