"""LoRAContext and MultiLoRALinear modules."""

from __future__ import annotations

import torch
from torch import nn

from persona import ops


class LoRAContext:
    """One per engine. Set once per forward pass; read by every LoRA layer."""

    def __init__(self, max_tokens: int, device: torch.device):
        self.token_slots = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device=device
        )
        self.row_slots: torch.Tensor | None = None
        self.enabled = True

    def set_batch_slots(self, row_slots: torch.Tensor, seq_len: int) -> None:
        """Expand per-request slots to per-token slots for prefill or decode."""
        if row_slots.dtype != torch.int32:
            row_slots = row_slots.to(dtype=torch.int32)
        self.row_slots = row_slots
        batch = row_slots.shape[0]
        ntok = batch * seq_len
        self.token_slots[:ntok] = row_slots.repeat_interleave(seq_len)


class MultiLoRALinear(nn.Module):
    """Wraps a base linear (e.g. Linear4bit) and applies pooled BGMV LoRA."""

    def __init__(
        self,
        base: nn.Module,
        ctx: LoRAContext,
        a_pool: torch.Tensor,
        b_pool: torch.Tensor,
    ):
        super().__init__()
        self.base = base
        self.ctx = ctx
        self.a_pool = a_pool
        self.b_pool = b_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.ctx.enabled:
            return y
        x2d = x.reshape(-1, x.shape[-1])
        y2d = y.reshape(-1, y.shape[-1])
        t = x2d.shape[0]
        slots = self.ctx.token_slots[:t]
        v = ops.bgmv_shrink(x2d.contiguous(), self.a_pool, slots)
        # y2d is a view into y; expand adds in place. Do not re-contiguous() a copy.
        ops.bgmv_expand(v, self.b_pool, slots, y2d)
        return y


def make_slot_hook(ctx: LoRAContext):
    """Forward pre-hook: expand per-request slots to per-token slots once."""

    def hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None or ctx.row_slots is None:
            return
        batch, seq = input_ids.shape
        ctx.token_slots[: batch * seq] = ctx.row_slots.repeat_interleave(seq)

    return hook
