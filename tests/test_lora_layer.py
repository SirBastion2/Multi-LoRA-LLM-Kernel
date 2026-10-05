import torch

from persona.lora_layer import LoRAContext, MultiLoRALinear


class IdentityBase(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_multi_lora_disabled():
    ctx = LoRAContext(max_tokens=4, device=torch.device("cpu"))
    ctx.enabled = False
    a = torch.randn(2, 8, 16)
    b = torch.randn(2, 16, 8)
    layer = MultiLoRALinear(IdentityBase(), ctx, a, b)
    x = torch.randn(2, 16)
    y = layer(x)
    assert torch.allclose(y, torch.zeros_like(x))
