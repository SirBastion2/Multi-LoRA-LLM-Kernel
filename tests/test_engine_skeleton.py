"""Engine wiring without loading Mistral weights."""

import torch

from persona.adapter_pool import AdapterPool
from persona.engine import Request, StaticBatchEngine


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})


class DummyTok:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompts, **kwargs):
        batch = len(prompts)
        return _Batch(input_ids=torch.zeros(batch, 4, dtype=torch.long))


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = []
        # Parameter so StaticBatchEngine can resolve device via next(parameters())
        self._probe = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


def test_static_batch_plan():
    pool = AdapterPool(4, 16, torch.device("cpu"), torch.float16)
    pool.load_random_adapter("p0")
    engine = StaticBatchEngine(DummyModel(), DummyTok(), pool, max_batch=2)
    req = Request("1", "p0", "hi", slot=pool.name_to_slot["p0"])
    out = engine.run_static_generate([req], use_hf_generate=False)
    assert out["row_slots"].tolist() == [req.slot]
