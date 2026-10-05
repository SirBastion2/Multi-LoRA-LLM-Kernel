"""CUDA kernel tests vs FP32 reference (skip without GPU / extension)."""

import pytest
import torch

from persona import ops


pytestmark = pytest.mark.cuda


def _require_cuda_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not ops.has_cuda_ops():
        pytest.skip("persona._C extension not built")


@pytest.mark.parametrize("R", [8, 16])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_shrink_cuda_vs_ref(R, dtype):
    _require_cuda_ops()
    T, K, S = 8, 256, 4
    x = torch.randn(T, K, dtype=dtype, device="cuda")
    a = torch.randn(S, R, K, dtype=dtype, device="cuda") * 0.01
    slots = torch.tensor([0, 1, 2, 3, 0, 1, -1, 2], dtype=torch.int32, device="cuda")
    v_ref = ops.bgmv_shrink_ref(x.cpu(), a.cpu(), slots.cpu())
    v = ops.bgmv_shrink(x, a, slots)
    torch.testing.assert_close(v.cpu(), v_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("R", [16])
def test_expand_cuda_vs_ref(R):
    _require_cuda_ops()
    T, N, S = 8, 256, 4
    v = torch.randn(T, R, dtype=torch.float32, device="cuda")
    b = torch.randn(S, N, R, dtype=torch.float16, device="cuda") * 0.01
    slots = torch.tensor([0, 1, 2, 3, 0, 1, -1, 2], dtype=torch.int32, device="cuda")
    y = torch.randn(T, N, dtype=torch.float16, device="cuda")
    y_ref = y.cpu().clone()
    ops.bgmv_expand_ref(v.cpu(), b.cpu(), slots.cpu(), y_ref)
    ops.bgmv_expand(v, b, slots, y)
    torch.testing.assert_close(y.cpu(), y_ref, rtol=1e-2, atol=1e-2)
