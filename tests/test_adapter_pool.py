import json
from pathlib import Path

import torch

from persona.adapter_pool import AdapterPool, validate_uniform_adapters


def _write_minimal_adapter(tmp_path: Path, name: str, r: int = 16):
    d = tmp_path / name
    d.mkdir()
    cfg = {
        "r": r,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "use_rslora": False,
    }
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    # Empty safetensors would fail; use random adapter script pattern in integration test.
    return d


def test_pool_random_slot():
    pool = AdapterPool(max_slots=4, r=16, device=torch.device("cpu"), dtype=torch.float16)
    s0 = pool.load_random_adapter("a")
    s1 = pool.load_random_adapter("b")
    assert s0 != s1
    assert pool.name_to_slot["a"] == s0
    pool.release_slot("a")
    assert "a" not in pool.name_to_slot


def test_validate_uniform(tmp_path):
    a = _write_minimal_adapter(tmp_path, "a")
    b = _write_minimal_adapter(tmp_path, "b")
    cfg = validate_uniform_adapters([a, b])
    assert cfg.r == 16
