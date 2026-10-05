"""CUDA op loader and FP32 PyTorch reference implementations for BGMV."""

from __future__ import annotations

import torch


def has_cuda_ops() -> bool:
    return hasattr(torch.ops, "persona") and hasattr(torch.ops.persona, "bgmv_shrink")


def bgmv_shrink_ref(
    x: torch.Tensor,
    a_pool: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Reference shrink: v[t, r] = A[slot[t], r, :] @ x[t, :] in FP32.

    x: [T, K] fp16/bf16
    a_pool: [S, R, K]
    slots: [T] int32, -1 means zero row
    """
    T, K = x.shape
    S, R, K2 = a_pool.shape
    assert K == K2
    x32 = x.float()
    a32 = a_pool.float()
    v = torch.zeros(T, R, dtype=torch.float32, device=x.device)
    for t in range(T):
        slot = int(slots[t].item())
        if slot < 0:
            continue
        v[t] = torch.matmul(a32[slot], x32[t])
    return v


def bgmv_expand_ref(
    v: torch.Tensor,
    b_pool: torch.Tensor,
    slots: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Reference expand: y[t, n] += B[slot[t], n, :] @ v[t] (in-place on y)."""
    T, R = v.shape
    S, N, R2 = b_pool.shape
    assert R == R2
    y32 = y.float()
    b32 = b_pool.float()
    for t in range(T):
        slot = int(slots[t].item())
        if slot < 0:
            continue
        y32[t] += torch.matmul(b32[slot], v[t])  # B[N,R] @ v[R] -> [N]
    out = y32.to(dtype=y.dtype)
    y.copy_(out)  # honor in-place contract used by CUDA path / tests
    return y


def bgmv_forward_ref(
    x: torch.Tensor,
    a_pool: torch.Tensor,
    b_pool: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Full LoRA delta: expand(shrink(x))."""
    v = bgmv_shrink_ref(x, a_pool, slots)
    y = torch.zeros(x.shape[0], b_pool.shape[1], dtype=x.dtype, device=x.device)
    return bgmv_expand_ref(v, b_pool, slots, y)


def bgmv_shrink(x: torch.Tensor, a_pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    v = torch.empty(
        (x.shape[0], a_pool.shape[1]), dtype=torch.float32, device=x.device
    )
    if has_cuda_ops() and x.is_cuda:
        torch.ops.persona.bgmv_shrink(x.contiguous(), a_pool, slots, v)
        return v
    return bgmv_shrink_ref(x, a_pool, slots)


def bgmv_expand(
    v: torch.Tensor,
    b_pool: torch.Tensor,
    slots: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    if has_cuda_ops() and v.is_cuda:
        torch.ops.persona.bgmv_expand(v, b_pool, slots, y)
        return y
    return bgmv_expand_ref(v, b_pool, slots, y)
