"""CPU reference math for BGMV (section 4 correctness plan)."""

import pytest
import torch

from persona.ops import bgmv_expand_ref, bgmv_forward_ref, bgmv_shrink_ref


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("R", [8, 16])
@pytest.mark.parametrize("T", [1, 4, 8])
def test_shrink_expand_no_adapter(dtype, R, T):
    K = 128
    S = 4
    x = torch.randn(T, K, dtype=dtype)
    a = torch.randn(S, R, K, dtype=dtype)
    b = torch.randn(S, K, R, dtype=dtype)  # use square N=K for test
    slots = torch.full((T,), -1, dtype=torch.int32)
    v = bgmv_shrink_ref(x, a, slots)
    assert torch.allclose(v, torch.zeros_like(v))
    y = torch.ones(T, K, dtype=dtype)
    y_before = y.clone()
    bgmv_expand_ref(v, b, slots, y)
    assert torch.equal(y, y_before)


@pytest.mark.parametrize("R", [8, 16])
def test_forward_matches_manual(R):
    T, K, N, S = 3, 64, 64, 2
    x = torch.randn(T, K, dtype=torch.float16)
    a = torch.randn(S, R, K, dtype=torch.float16) * 0.01
    b = torch.randn(S, N, R, dtype=torch.float16) * 0.01
    slots = torch.tensor([0, 1, 0], dtype=torch.int32)
    y = bgmv_forward_ref(x, a, b, slots)
    v = bgmv_shrink_ref(x, a, slots)
    y2 = torch.zeros(T, N, dtype=torch.float16)
    bgmv_expand_ref(v, b, slots, y2)
    torch.testing.assert_close(y, y2, rtol=1e-3, atol=1e-3)


def test_adapter_isolation():
    T, K, R, N, S = 2, 128, 16, 128, 2
    x = torch.randn(T, K, dtype=torch.float16)
    a = torch.zeros(S, R, K, dtype=torch.float16)
    b = torch.zeros(S, N, R, dtype=torch.float16)
    a[1, 0, 0] = 1.0
    b[1, 0, 0] = 1.0
    slots = torch.tensor([0, 1], dtype=torch.int32)
    y = bgmv_forward_ref(x, a, b, slots)
    assert y[0].abs().sum() == 0
    assert y[1, 0].abs() > 0
