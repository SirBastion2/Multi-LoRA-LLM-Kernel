"""BGMV kernel microbenchmark on real Mistral-7B LoRA shapes (RTX 3060).

Measures shrink+expand wall time for D=1 vs D>1 (distinct adapter slots per
token) using CUDA events. Does not invent numbers — prints measured values.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from persona import ops

# Mistral-7B module geometry
SHAPES = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 1024),
    "v_proj": (4096, 1024),
    "o_proj": (4096, 4096),
    "gate_proj": (4096, 14336),
    "up_proj": (4096, 14336),
    "down_proj": (14336, 4096),
}


@dataclass
class Row:
    shape: str
    T: int
    D: int
    R: int
    mode: str  # cuda | ref
    mean_ms: float
    std_ms: float
    iters: int


def _require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for microbench")
    if not ops.has_cuda_ops():
        print("WARNING: persona CUDA ops not loaded; will time FP32 reference only", file=sys.stderr)


def make_slots(T: int, D: int, S: int, device: str) -> torch.Tensor:
    """D distinct adapters cycling across T tokens. D must be <= S."""
    assert 1 <= D <= S
    slots = [i % D for i in range(T)]
    return torch.tensor(slots, dtype=torch.int32, device=device)


def time_bgmv(
    K: int,
    N: int,
    T: int,
    D: int,
    S: int,
    R: int,
    warmup: int,
    iters: int,
    use_cuda_ops: bool,
) -> tuple[float, float]:
    device = "cuda"
    dtype = torch.float16
    g = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(T, K, generator=g, dtype=dtype).to(device)
    a = (torch.randn(S, R, K, generator=g, dtype=dtype) * 0.01).to(device)
    b = (torch.randn(S, N, R, generator=g, dtype=dtype) * 0.01).to(device)
    slots = make_slots(T, D, S, device)
    y = torch.zeros(T, N, dtype=dtype, device=device)

    def once():
        y.zero_()
        if use_cuda_ops:
            v = ops.bgmv_shrink(x, a, slots)
            ops.bgmv_expand(v, b, slots, y)
        else:
            v = ops.bgmv_shrink_ref(x, a, slots)
            ops.bgmv_expand_ref(v, b, slots, y)
        return y

    # warmup
    for _ in range(warmup):
        once()
    torch.cuda.synchronize()

    starts = []
    ends = []
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    import statistics

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def time_full_step_proxy(
    T: int,
    D: int,
    S: int,
    R: int,
    modules: list[str],
    layers: int,
    warmup: int,
    iters: int,
    use_cuda_ops: bool,
) -> tuple[float, float]:
    """Time (shrink+expand) × modules × layers — launch-bound proxy for one decode step's LoRA work."""
    device = "cuda"
    dtype = torch.float16
    g = torch.Generator(device="cpu").manual_seed(7)
    slots = make_slots(T, D, S, device)
    bundles = []
    for name in modules:
        K, N = SHAPES[name]
        x = torch.randn(T, K, generator=g, dtype=dtype).to(device)
        a = (torch.randn(S, R, K, generator=g, dtype=dtype) * 0.01).to(device)
        b = (torch.randn(S, N, R, generator=g, dtype=dtype) * 0.01).to(device)
        y = torch.zeros(T, N, dtype=dtype, device=device)
        bundles.append((x, a, b, y))

    def once():
        for _layer in range(layers):
            for x, a, b, y in bundles:
                y.zero_()
                if use_cuda_ops:
                    v = ops.bgmv_shrink(x, a, slots)
                    ops.bgmv_expand(v, b, slots, y)
                else:
                    v = ops.bgmv_shrink_ref(x, a, slots)
                    ops.bgmv_expand_ref(v, b, slots, y)

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    import statistics

    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


def env_meta() -> dict:
    meta = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "has_cuda_ops": ops.has_cuda_ops(),
        "time_unix": time.time(),
    }
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        meta["nvidia_smi"] = smi
    except Exception as e:
        meta["nvidia_smi_err"] = str(e)
    try:
        meta["git"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=str(Path(__file__).resolve().parents[1])
        ).strip()
    except Exception:
        meta["git"] = "not-a-git-repo"
    return meta


def main():
    p = argparse.ArgumentParser(description="BGMV microbenchmarks (Mistral shapes)")
    p.add_argument("--shape", default="all", help="module name or 'all' or 'step'")
    p.add_argument("--T", type=int, default=8, help="tokens in batch")
    p.add_argument("--D-list", default="1,4,8", help="comma list of distinct adapters")
    p.add_argument("--S", type=int, default=8, help="pool slots (>= max D)")
    p.add_argument("--R", type=int, default=16)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--layers", type=int, default=32, help="for --shape step")
    p.add_argument(
        "--modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="comma modules for step proxy (match original train.py targets by default)",
    )
    p.add_argument("--ref", action="store_true", help="time FP32 reference instead of CUDA ops")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    _require_cuda()
    use_cuda = ops.has_cuda_ops() and not args.ref
    D_list = [int(x) for x in args.D_list.split(",") if x.strip()]
    if max(D_list) > args.S:
        raise SystemExit(f"max D={max(D_list)} > S={args.S}")

    rows: list[dict] = []
    meta = env_meta()
    print("=== env ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  mode: {'cuda_ops' if use_cuda else 'fp32_ref'}")
    print()

    if args.shape == "step":
        modules = [m.strip() for m in args.modules.split(",") if m.strip()]
        print(f"=== step LoRA proxy: {args.layers} layers × {modules} ===")
        print(f"{'D':>4} {'T':>4} {'mean_ms':>10} {'std_ms':>10} {'ms/tok':>10}")
        for D in D_list:
            mean, std = time_full_step_proxy(
                args.T, D, args.S, args.R, modules, args.layers, args.warmup, args.iters, use_cuda
            )
            print(f"{D:4d} {args.T:4d} {mean:10.3f} {std:10.3f} {mean/args.T:10.3f}")
            rows.append(
                {
                    "kind": "step_proxy",
                    "D": D,
                    "T": args.T,
                    "R": args.R,
                    "layers": args.layers,
                    "modules": modules,
                    "mean_ms": mean,
                    "std_ms": std,
                    "mode": "cuda" if use_cuda else "ref",
                }
            )
    else:
        shapes = list(SHAPES.keys()) if args.shape == "all" else [args.shape]
        print(f"{'shape':>12} {'D':>4} {'T':>4} {'R':>4} {'mean_us':>10} {'std_us':>10}")
        for name in shapes:
            K, N = SHAPES[name]
            for D in D_list:
                mean_ms, std_ms = time_bgmv(
                    K, N, args.T, D, args.S, args.R, args.warmup, args.iters, use_cuda
                )
                print(
                    f"{name:>12} {D:4d} {args.T:4d} {args.R:4d} {mean_ms*1000:10.1f} {std_ms*1000:10.1f}"
                )
                rows.append(
                    {
                        "kind": "single",
                        "shape": name,
                        "K": K,
                        "N": N,
                        "D": D,
                        "T": args.T,
                        "R": args.R,
                        "mean_ms": mean_ms,
                        "std_ms": std_ms,
                        "mode": "cuda" if use_cuda else "ref",
                    }
                )

    # 1-vs-N summary for q_proj if we have both
    by_d = {}
    for r in rows:
        if r.get("shape") == "q_proj" or r.get("kind") == "step_proxy":
            by_d.setdefault(r["D"], r)
    if 1 in by_d and any(d > 1 for d in by_d):
        print("\n=== 1-vs-N (same T) ===")
        base = by_d[1]["mean_ms"]
        for d, r in sorted(by_d.items()):
            print(f"  D={d}: {r['mean_ms']:.3f} ms  (vs D=1: {r['mean_ms']/base:.2f}x)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
