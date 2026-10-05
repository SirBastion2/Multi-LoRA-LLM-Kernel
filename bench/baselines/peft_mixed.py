"""S2 baseline: PEFT per-sample adapter_names mixed batch.

PEFT 0.21 `_mixed_batch_forward` splits the batch by adapter and runs one
sub-forward per unique adapter inside each LoRA layer. It only accepts
nn.Linear / Embedding / Conv* as `original_module` — **not** bitsandbytes
Linear4bit. On an NF4 Mistral base this path raises TypeError.

True simultaneous multi-adapter decode is the persona BGMV path
(`bench/e2e.py --mode persona-generate`).
"""

from __future__ import annotations


def run_peft_mixed_benchmark():
    raise NotImplementedError(
        "PEFT adapter_names mixed batch is unsupported with bitsandbytes "
        "Linear4bit (NF4). Use bench/e2e.py --mode persona-generate for true "
        "simultaneous multi-adapter decode, or peft-sequential for the S1 baseline."
    )
