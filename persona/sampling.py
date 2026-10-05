"""Per-request sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 128


def sample_next_token(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """
    logits: [batch, vocab]
    Returns: [batch] int64 next tokens
    """
    if params.temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / params.temperature, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > params.top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, idx).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
