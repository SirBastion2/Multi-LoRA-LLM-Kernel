"""S1 baseline: serve one adapter group at a time.

Real implementation lives in bench/e2e.py --mode peft-sequential.
"""

from __future__ import annotations


def run_sequential_benchmark(**kwargs):
    raise SystemExit(
        "Use: python bench/e2e.py --mode peft-sequential "
        "(loads NF4 Mistral + PEFT adapters one at a time)."
    )
