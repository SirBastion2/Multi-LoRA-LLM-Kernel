"""Engine v1: static batching skeleton (Hugging Face generate + fixed slots)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from persona.adapter_pool import AdapterPool
from persona.lora_layer import LoRAContext
from persona.model_patch import patch_model_with_pools
from persona.sampling import SamplingParams, sample_next_token


@dataclass
class Request:
    request_id: str
    persona_name: str
    prompt: str
    slot: int = -1
    sampling: SamplingParams = field(default_factory=SamplingParams)


class StaticBatchEngine:
    """
    Version 1 static batching: all requests start together; fixed slot per row.

    Requires a patched Hugging Face model and tokenizer on CUDA. This scaffold
    wires slot assignment and LoRAContext; it does not download Mistral weights.

    Owner setup (CachyOS, persona env):
      1. Load NF4 Mistral.
      2. pool.load_adapter(...) or load_random_adapter(...) per personality.
      3. Construct StaticBatchEngine (patches after pool is loaded).
      4. Construct requests with slot indices from pool.name_to_slot.
      5. Call run_static_generate(model, tokenizer, requests, max_new_tokens=...).
    """

    def __init__(
        self,
        model,
        tokenizer,
        pool: AdapterPool,
        max_batch: int = 8,
        max_seq_len: int = 2048,
        target_modules: Optional[Sequence[str]] = None,
        already_patched: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pool = pool
        device = next(model.parameters()).device
        self.ctx = LoRAContext(max_tokens=max_batch * max_seq_len, device=device)
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        if not already_patched:
            # Adapters must already be in the pool (A_qkv slices are copied at patch).
            patch_model_with_pools(
                model, pool, self.ctx, target_modules=target_modules
            )

    def plan_batch(self, requests: Sequence[Request]) -> torch.Tensor:
        if len(requests) > self.max_batch:
            raise ValueError("Static batch exceeds max_batch")
        slots = torch.tensor(
            [r.slot for r in requests], dtype=torch.int32, device=self.ctx.token_slots.device
        )
        return slots

    @torch.inference_mode()
    def run_static_generate(
        self,
        requests: Sequence[Request],
        max_new_tokens: int = 128,
        use_hf_generate: bool = True,
    ):
        """
        Left-padded batch generate. Sets row_slots for the duration of the call.

        When `use_hf_generate` is True, delegates to `model.generate` (needs GPU +
        weights). When False, returns the planned slot tensor for unit tests.
        """
        row_slots = self.plan_batch(requests)
        self.ctx.row_slots = row_slots

        prompts = [r.prompt for r in requests]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.ctx.token_slots.device)

        batch, seq = enc["input_ids"].shape
        self.ctx.set_batch_slots(row_slots, seq)

        if not use_hf_generate:
            return {"input_ids": enc["input_ids"], "row_slots": row_slots}

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=any(r.sampling.temperature > 0 for r in requests),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        return out

    def decode_step_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """Single-step greedy decode helper for custom loops (v2 precursor)."""
        return sample_next_token(logits, SamplingParams(temperature=0.0))
