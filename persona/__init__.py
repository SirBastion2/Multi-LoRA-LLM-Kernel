"""Persona-Serve: batched LoRA serving with custom BGMV kernels."""

from persona.ops import has_cuda_ops

__all__ = ["has_cuda_ops"]

# Register torch.ops.persona when the extension is built.
try:
    from persona import _C  # noqa: F401
except ImportError:
    pass
