"""End-to-end / PEFT baseline: 1 vs N personality decode timing on NF4 Mistral.

Uses mistralai/Mistral-7B-Instruct-v0.2 (same as SirBastion2/Mistral-7B-LLM-custom-trained).
Adapters default to adapters/random (SYNTHETIC random LoRA weights, real shapes).
Personas: adapters/synthetic_personas.json (synthetic system prompts).

Honest modes:
  --mode peft-sequential  : load one PEFT adapter at a time, generate separately (S1 baseline)
  --mode peft-batch-one   : single adapter, batch size = N prompts (same personality) [via peft-sequential]
  --mode peft-mixed       : try PEFT adapter_names mixed batch (expected fail on NF4/bnb)
  --mode kernel-pool      : load AdapterPool + BGMV micro proxy only (no HF generate)
  --mode persona-generate : NF4 + MultiLoRALinear + BGMV true mixed-adapter batch generate
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def env_meta() -> dict:
    meta = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "time_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    try:
        meta["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as e:
        meta["nvidia_smi_err"] = str(e)
    try:
        import transformers, peft, bitsandbytes

        meta["transformers"] = transformers.__version__
        meta["peft"] = peft.__version__
        meta["bitsandbytes"] = bitsandbytes.__version__
    except Exception as e:
        meta["import_err"] = str(e)
    try:
        from persona import ops

        meta["has_cuda_ops"] = ops.has_cuda_ops()
    except Exception:
        meta["has_cuda_ops"] = False
    return meta


def load_personas(path: Path) -> dict:
    return json.loads(path.read_text())


def load_nf4_base(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tok


def build_prompt(tok, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"


@torch.inference_mode()
def timed_generate(model, tok, prompts: list[str], max_new_tokens: int, warmup: bool = False):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    # Left-padded batches share one padded prompt length; per-row attention_mask
    # sums differ and would over-count new tokens for shorter prompts.
    prompt_pad_len = enc["input_ids"].shape[1]
    new_per_row = max(0, int(out.shape[1] - prompt_pad_len))
    new_tokens = new_per_row * out.shape[0]
    texts = []
    for i in range(out.shape[0]):
        gen = out[i, prompt_pad_len:]
        texts.append(tok.decode(gen, skip_special_tokens=True))
    return {
        "wall_s": dt,
        "new_tokens": new_tokens,
        "tok_per_s": (new_tokens / dt) if dt > 0 else 0.0,
        "batch": len(prompts),
        "texts": texts if not warmup else [],
    }


def run_peft_sequential(args, personas_doc: dict):
    from peft import PeftModel

    model, tok = load_nf4_base(args.model)
    print("base loaded; allocated_gb", round(torch.cuda.memory_allocated() / 1e9, 3))
    user = personas_doc["user_prompt"]
    personas = personas_doc["personas"][: args.n]
    adapter_root = Path(args.adapters)

    results = []

    # Warmup with first adapter
    first = personas[0]
    path0 = adapter_root / first["adapter"]
    model_p = PeftModel.from_pretrained(model, str(path0), adapter_name=first["id"])
    prompt0 = build_prompt(tok, first["system"], user)
    print("warmup...")
    timed_generate(model_p, tok, [prompt0], max_new_tokens=8, warmup=True)

    # D=1: single personality, measure
    print("\n=== D=1 PEFT single adapter generate ===")
    r1 = timed_generate(model_p, tok, [prompt0], max_new_tokens=args.max_new_tokens)
    print(
        f"  persona={first['id']} wall_s={r1['wall_s']:.3f} new_tokens={r1['new_tokens']} tok/s={r1['tok_per_s']:.2f}"
    )
    print("  sample:", (r1["texts"][0][:160] + "...") if r1["texts"] else "")
    results.append({"mode": "peft_d1", **{k: r1[k] for k in ("wall_s", "new_tokens", "tok_per_s", "batch")}, "persona": first["id"]})

    # D=N sequential: switch adapters, generate one prompt each, sum wall time
    print(f"\n=== D={len(personas)} PEFT sequential (load/switch + generate each) ===")
    # already have first loaded; for others set_adapter / load
    seq_wall = 0.0
    seq_tokens = 0
    for i, p in enumerate(personas):
        ap = adapter_root / p["adapter"]
        if p["id"] not in model_p.peft_config:
            t_load0 = time.perf_counter()
            model_p.load_adapter(str(ap), adapter_name=p["id"])
            torch.cuda.synchronize()
            load_s = time.perf_counter() - t_load0
        else:
            load_s = 0.0
        model_p.set_adapter(p["id"])
        prompt = build_prompt(tok, p["system"], user)
        r = timed_generate(model_p, tok, [prompt], max_new_tokens=args.max_new_tokens)
        seq_wall += r["wall_s"] + load_s
        seq_tokens += r["new_tokens"]
        print(
            f"  [{i}] {p['id']}: gen_s={r['wall_s']:.3f} load_s={load_s:.3f} tokens={r['new_tokens']} tok/s={r['tok_per_s']:.2f}"
        )
        print("       sample:", (r["texts"][0][:120] + "...") if r["texts"] else "")
        results.append(
            {
                "mode": "peft_seq_item",
                "persona": p["id"],
                "wall_s": r["wall_s"],
                "load_s": load_s,
                "new_tokens": r["new_tokens"],
                "tok_per_s": r["tok_per_s"],
            }
        )

    print(
        f"\n  SEQUENTIAL TOTAL wall_s={seq_wall:.3f} tokens={seq_tokens} aggregate_tok/s={seq_tokens/seq_wall if seq_wall else 0:.2f}"
    )
    results.append(
        {
            "mode": "peft_sequential_total",
            "D": len(personas),
            "wall_s": seq_wall,
            "new_tokens": seq_tokens,
            "tok_per_s": seq_tokens / seq_wall if seq_wall else 0,
        }
    )

    # Same-adapter batched N prompts (upper bound if personalities identical)
    print(f"\n=== batch={len(personas)} same adapter (personality={first['id']}) ===")
    model_p.set_adapter(first["id"])
    prompts = [build_prompt(tok, first["system"], user) for _ in personas]
    rb = timed_generate(model_p, tok, prompts, max_new_tokens=args.max_new_tokens)
    print(f"  wall_s={rb['wall_s']:.3f} tokens={rb['new_tokens']} tok/s={rb['tok_per_s']:.2f}")
    results.append({"mode": "peft_same_adapter_batch", **{k: rb[k] for k in ("wall_s", "new_tokens", "tok_per_s", "batch")}})

    print("\n=== 1-vs-N table (PEFT) ===")
    print(f"{'case':<28} {'D/batch':>8} {'wall_s':>10} {'tokens':>8} {'tok/s':>10}")
    print(f"{'D=1 single':<28} {1:8d} {r1['wall_s']:10.3f} {r1['new_tokens']:8d} {r1['tok_per_s']:10.2f}")
    print(
        f"{'D=N sequential sum':<28} {len(personas):8d} {seq_wall:10.3f} {seq_tokens:8d} {seq_tokens/seq_wall if seq_wall else 0:10.2f}"
    )
    print(
        f"{'batch=N same adapter':<28} {rb['batch']:8d} {rb['wall_s']:10.3f} {rb['new_tokens']:8d} {rb['tok_per_s']:10.2f}"
    )
    print(
        "Note: random adapters + synthetic system prompts; output quality is NOT a personality eval."
    )
    return results


def run_kernel_pool(args, personas_doc: dict):
    """Pack adapters into AdapterPool and time BGMV step proxy (attn modules)."""
    import statistics
    from persona.adapter_pool import AdapterPool
    from persona import ops

    device = torch.device("cuda")
    personas = personas_doc["personas"][: args.n]
    pool = AdapterPool(max_slots=max(8, args.n), r=args.rank, device=device)
    for p in personas:
        ap = Path(args.adapters) / p["adapter"]
        if (ap / "adapter_config.json").exists():
            slot = pool.load_adapter(p["id"], ap)
        else:
            slot = pool.load_random_adapter(p["id"])
        print(f"  loaded {p['id']} -> slot {slot} from {ap if ap.exists() else 'random'}")

    # Build per-module tensors from pool layer 0 shapes for timing (use synthetic x)
    # Time full 32-layer × 4 attn modules shrink+expand for D=1 and D=N
    T = args.batch_tokens
    rows = []
    modules_spec = [
        ("q", pool.layers[0].A_qkv[:, : args.rank, :], pool.layers[0].B_q),
        ("k", pool.layers[0].A_qkv[:, args.rank : 2 * args.rank, :], pool.layers[0].B_k),
        ("v", pool.layers[0].A_qkv[:, 2 * args.rank : 3 * args.rank, :], pool.layers[0].B_v),
        ("o", pool.layers[0].A_o, pool.layers[0].B_o),
    ]
    # Materialize independent pool copies per layer for realistic traffic
    layer_mods = []
    for li in range(32):
        lp = pool.layers[li]
        layer_mods.append(
            [
                (lp.A_qkv[:, : args.rank, :].contiguous(), lp.B_q),
                (lp.A_qkv[:, args.rank : 2 * args.rank, :].contiguous(), lp.B_k),
                (lp.A_qkv[:, 2 * args.rank : 3 * args.rank, :].contiguous(), lp.B_v),
                (lp.A_o, lp.B_o),
            ]
        )

    for D in (1, len(personas)):
        slots = torch.tensor([i % D for i in range(T)], dtype=torch.int32, device=device)
        # prebuild x/y per module using layer0 shapes
        xs, ys = [], []
        for a, b in layer_mods[0]:
            K = a.shape[-1]
            N = b.shape[1]
            xs.append(torch.randn(T, K, dtype=torch.float16, device=device))
            ys.append(torch.zeros(T, N, dtype=torch.float16, device=device))

        def once():
            for mods in layer_mods:
                for (a, b), x, y in zip(mods, xs, ys):
                    y.zero_()
                    v = ops.bgmv_shrink(x, a, slots)
                    ops.bgmv_expand(v, b, slots, y)

        for _ in range(5):
            once()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            st = torch.cuda.Event(True)
            en = torch.cuda.Event(True)
            st.record()
            once()
            en.record()
            torch.cuda.synchronize()
            times.append(st.elapsed_time(en))
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  pool step proxy D={D} T={T}: {mean:.3f}±{std:.3f} ms  (32L × qkvo)")
        rows.append({"mode": "kernel_pool_step", "D": D, "T": T, "mean_ms": mean, "std_ms": std})

    if len(rows) == 2:
        print(f"  1-vs-N ratio (D={rows[1]['D']}/D=1): {rows[1]['mean_ms']/rows[0]['mean_ms']:.3f}x")
    return rows




def run_peft_mixed(args, personas_doc: dict):
    """Attempt PEFT adapter_names mixed batch; record success or the real error."""
    from peft import PeftModel

    model, tok = load_nf4_base(args.model)
    user = personas_doc["user_prompt"]
    personas = personas_doc["personas"][: args.n]
    adapter_root = Path(args.adapters)

    first = personas[0]
    model_p = PeftModel.from_pretrained(
        model, str(adapter_root / first["adapter"]), adapter_name=first["id"]
    )
    for p in personas[1:]:
        model_p.load_adapter(str(adapter_root / p["adapter"]), adapter_name=p["id"])

    prompts = [build_prompt(tok, p["system"], user) for p in personas]
    names = [p["id"] for p in personas]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc = {k: v.to(model_p.device) for k, v in enc.items()}

    rows = []
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model_p.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            adapter_names=names,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        prompt_pad_len = enc["input_ids"].shape[1]
        new_tokens = max(0, int(out.shape[1] - prompt_pad_len)) * out.shape[0]
        row = {
            "mode": "peft_mixed_adapter_names",
            "status": "ok",
            "D": len(personas),
            "wall_s": dt,
            "new_tokens": new_tokens,
            "tok_per_s": new_tokens / dt if dt else 0.0,
            "note": "PEFT splits sub-batches per unique adapter (not true BGMV)",
        }
        print(f"  peft-mixed OK wall_s={dt:.3f} tok/s={row['tok_per_s']:.2f}")
        rows.append(row)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  peft-mixed FAILED (expected on NF4/bnb): {msg}")
        rows.append(
            {
                "mode": "peft_mixed_adapter_names",
                "status": "failed",
                "D": len(personas),
                "error": msg,
                "note": "bitsandbytes Linear4bit not in PEFT mixed-batch SUPPORTED_MODULES",
            }
        )
    return rows


def run_persona_generate(args, personas_doc: dict):
    """True simultaneous multi-adapter batch via AdapterPool + MultiLoRALinear + BGMV."""
    from persona.adapter_pool import AdapterPool
    from persona.lora_layer import LoRAContext
    from persona.model_patch import patch_model_with_pools

    model, tok = load_nf4_base(args.model)
    print("base loaded; allocated_gb", round(torch.cuda.memory_allocated() / 1e9, 3))

    user = personas_doc["user_prompt"]
    # Support D up to args.n; synthesize extra persona entries if needed
    base_personas = list(personas_doc["personas"])
    while len(base_personas) < args.n:
        i = len(base_personas)
        src = base_personas[i % len(personas_doc["personas"])]
        base_personas.append(
            {
                "id": f"{src['id']}_extra{i}",
                "adapter": f"adapter_{i}",
                "system": src["system"],
            }
        )
    personas = base_personas[: args.n]
    adapter_root = Path(args.adapters)
    device = torch.device("cuda")

    # Target modules from first existing adapter config (qkvo for random/)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    cfg_path = adapter_root / personas[0]["adapter"] / "adapter_config.json"
    if cfg_path.exists():
        targets = list(json.loads(cfg_path.read_text())["target_modules"])

    pool = AdapterPool(max_slots=max(8, args.n), r=args.rank, device=device)
    for p in personas:
        ap = adapter_root / p["adapter"]
        if (ap / "adapter_config.json").exists() and (
            (ap / "adapter_model.safetensors").exists()
        ):
            slot = pool.load_adapter(p["id"], ap)
            src = str(ap)
        else:
            slot = pool.load_random_adapter(p["id"])
            src = "random_fill"
        print(f"  pool slot {slot}: {p['id']} <- {src}")

    print("pool allocated_gb", round(torch.cuda.memory_allocated() / 1e9, 3))

    ctx = LoRAContext(max_tokens=max(args.n, 1) * 2048, device=device)
    patch_model_with_pools(model, pool, ctx, target_modules=targets)
    print("patched MultiLoRALinear on", targets)

    def gen_batch(batch_personas, label: str):
        prompts = [build_prompt(tok, p["system"], user) for p in batch_personas]
        slots = torch.tensor(
            [pool.name_to_slot[p["id"]] for p in batch_personas],
            dtype=torch.int32,
            device=device,
        )
        ctx.row_slots = slots
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        seq = enc["input_ids"].shape[1]
        ctx.set_batch_slots(slots, seq)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        prompt_pad_len = enc["input_ids"].shape[1]
        new_per_row = max(0, int(out.shape[1] - prompt_pad_len))
        new_tokens = new_per_row * out.shape[0]
        texts = []
        for i in range(out.shape[0]):
            gen = out[i, prompt_pad_len:]
            texts.append(tok.decode(gen, skip_special_tokens=True))
        distinct = len({p["id"] for p in batch_personas})
        row = {
            "mode": label,
            "D": distinct,
            "batch": len(batch_personas),
            "wall_s": dt,
            "new_tokens": new_tokens,
            "tok_per_s": new_tokens / dt if dt else 0.0,
            "slots": slots.tolist(),
            "personas": [p["id"] for p in batch_personas],
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
        print(
            f"  {label}: D={distinct} batch={len(batch_personas)} "
            f"wall_s={dt:.3f} tokens={new_tokens} tok/s={row['tok_per_s']:.2f}"
        )
        for i, t in enumerate(texts):
            print(f"    [{batch_personas[i]['id']}] {(t[:100] + '...') if t else ''}")
        return row

    rows = []
    # Warmup D=1
    print("warmup...")
    try:
        gen_batch(personas[:1], "warmup")
    except Exception as e:
        print(f"persona-generate FAILED during warmup: {type(e).__name__}: {e}")
        rows.append(
            {
                "mode": "persona_generate",
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }
        )
        return rows

    torch.cuda.reset_peak_memory_stats()

    # D=1 single adapter batch
    print("\n=== persona-generate D=1 ===")
    rows.append(gen_batch(personas[:1], "persona_mixed_D1"))

    # D=N distinct adapters in one batch (true mixed)
    print(f"\n=== persona-generate D={len(personas)} mixed adapters ===")
    rows.append(gen_batch(personas, f"persona_mixed_D{len(personas)}"))

    # Same-adapter batch upper bound (all rows → first slot)
    print(f"\n=== persona-generate batch={len(personas)} same adapter ===")
    same = [{**personas[0], "id": personas[0]["id"]} for _ in personas]
    # Force same slot by reusing first persona id for all prompts but keep one slot
    prompts = [build_prompt(tok, personas[0]["system"], user) for _ in personas]
    slots = torch.tensor(
        [pool.name_to_slot[personas[0]["id"]]] * len(personas),
        dtype=torch.int32,
        device=device,
    )
    ctx.row_slots = slots
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}
    ctx.set_batch_slots(slots, enc["input_ids"].shape[1])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    prompt_pad_len = enc["input_ids"].shape[1]
    new_per_row = max(0, int(out.shape[1] - prompt_pad_len))
    new_tokens = new_per_row * out.shape[0]
    row = {
        "mode": "persona_same_adapter_batch",
        "D": 1,
        "batch": len(personas),
        "wall_s": dt,
        "new_tokens": new_tokens,
        "tok_per_s": new_tokens / dt if dt else 0.0,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    print(f"  same-adapter batch: wall_s={dt:.3f} tok/s={row['tok_per_s']:.2f}")
    rows.append(row)

    print("\n=== persona mixed-batch table ===")
    print(f"{'case':<32} {'D':>4} {'batch':>6} {'wall_s':>10} {'tokens':>8} {'tok/s':>10}")
    for r in rows:
        if "wall_s" in r:
            print(
                f"{r['mode']:<32} {r.get('D', 0):4d} {r.get('batch', 0):6d} "
                f"{r['wall_s']:10.3f} {r['new_tokens']:8d} {r['tok_per_s']:10.2f}"
            )
    print(
        "Note: random LoRA weights + synthetic system prompts; not a personality quality eval."
    )
    return rows



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["peft-sequential", "kernel-pool", "persona-generate", "peft-mixed"], default="peft-sequential")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapters", type=Path, default=ROOT / "adapters" / "random")
    p.add_argument("--personas", type=Path, default=ROOT / "adapters" / "synthetic_personas.json")
    p.add_argument("--n", type=int, default=4, help="number of personalities")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch-tokens", type=int, default=8)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    meta = env_meta()
    print("=== env ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    personas_doc = load_personas(args.personas)
    print("personas note:", personas_doc.get("note", "")[:120])

    if args.mode == "peft-sequential":
        rows = run_peft_sequential(args, personas_doc)
    elif args.mode == "kernel-pool":
        rows = run_kernel_pool(args, personas_doc)
    elif args.mode == "peft-mixed":
        rows = run_peft_mixed(args, personas_doc)
    elif args.mode == "persona-generate":
        rows = run_persona_generate(args, personas_doc)
    else:
        raise SystemExit(f"unknown mode {args.mode}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
        print("Wrote", args.json_out)


if __name__ == "__main__":
    main()
