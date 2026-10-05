# Benchmarks — Multi-LoRA LLM Kernel

**Date:** October 5, 2026 (EDT)  
**GPU:** NVIDIA GeForce RTX 3060 (12 GB) on CachyOS Linux  
**What this file is:** I stress-tested multiple LoRA adapter scenarios (1, 4, and 8 distinct adapters; mixed vs same-adapter batches; sequential vs batched) and benchmarked tokens per second for each. Every number below was measured on my machine.

---

## Results overview

This project serves **one shared copy** of a 7B chat model (Mistral) and applies a small LoRA adapter per request, so different "personas" can share one base model.

**Question:** if four or eight requests that each use a **different** adapter run at the same time, does my custom GPU path stay fast?

| Scenario | Throughput (tokens/s) | Notes |
|----------|-------------------:|-------|
| One adapter, single request | ~23–25 | Single-request reference |
| Four adapters, **one after another** (standard PEFT) | ~21 | Naive multi-adapter serving without a mixed batch |
| Four **different** adapters **in one batch** (my BGMV path) | **~33** | Main result: ~1.6× the sequential PEFT path |
| Eight **different** adapters **in one batch** | **~56** | Still fits in 12 GB VRAM |
| Four / eight requests sharing the **same** adapter | ~39 / ~67 | Upper bound: batching helps most when adapters match |

**Honesty note:** the adapters in this campaign were **random weights** (correct shapes, not trained). Persona "styles" came from short **synthetic system prompts**. This is an **engineering / throughput** report, not a persona-quality evaluation.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Base model** | The shared Mistral-7B network every request uses |
| **NF4** | 4-bit quantized storage for the base, so it fits in ~4 GB of GPU memory |
| **LoRA / adapter** | A small low-rank add-on that steers the base without copying the whole model |
| **D** | Number of **distinct** adapters active (D=4 means four different adapters) |
| **Batch** | Number of prompts processed **together** in one `generate` call |
| **Mixed-adapter batch** | One batch where **each row uses a different adapter** (the hard case) |
| **Same-adapter batch** | One batch where **every row uses the same adapter** (easier; usually faster) |
| **Sequential** | Finish adapter 1, then 2, then 3… (no mixing) |
| **tokens/s** | Tokens generated per second (higher = faster); aggregate over the whole batch when batching |
| **BGMV** | Batched gather matrix–vector: my custom CUDA kernels that apply the right adapter to each token in one GPU launch |
| **PEFT** | Hugging Face's standard adapter library (the baseline) |
| **µs / ms** | Microseconds / milliseconds — used for kernel-only timings |

---

## 1. Hardware and software

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB |
| Driver | 615.71.09 |
| Typical idle VRAM | ~532–860 MiB used by the desktop |
| Python env | micromamba env `persona` |
| PyTorch | 2.6.0+cu124 |
| transformers | 5.18.0 |
| peft | 0.21.2 |
| bitsandbytes | 0.50.2 |
| Base model | `mistralai/Mistral-7B-Instruct-v0.2` loaded as NF4 (~**4.13 GB** on GPU) |

---

## 2. Real vs synthetic assets

| Asset | Real or synthetic? | Why it matters |
|-------|--------------------|----------------|
| Mistral-7B base weights | **Real** (downloaded from Hugging Face) | Same model family the project targets |
| LoRA files under `adapters/random/` | **Synthetic** — random numbers, correct shapes | Valid for **throughput**; not for quality demos |
| Prompts in `adapters/synthetic_personas.json` | **Synthetic** — coach / tutor / sarcastic / calm / etc. | Makes outputs look different; not trained behavior |
| Trained persona adapters | **Not used** in this campaign | Planned as a later step |

---

## 3. Methodology

### 3.1 Baseline vs engine tests

Two different code paths produce the full-generate numbers in §6. They are not interchangeable:

| | **Baseline — Hugging Face PEFT** | **Engine — Multi-LoRA / BGMV path** |
|---|---|---|
| Script | `bench/e2e.py --mode peft-sequential` | `bench/e2e.py --mode persona-generate` |
| How adapters are applied | Stock PEFT `PeftModel` on the NF4 base; adapters added with `load_adapter`, one active at a time via `set_adapter` | My Multi-LoRA modules patched into the NF4 layers; all adapters live in one shared GPU pool |
| Multiple adapters | Run **one after another**, switching the active adapter between generates (switch cost included) | Run **in one batch**; each row carries an adapter slot index |
| Kernel | PyTorch / PEFT matmuls per adapter | Custom CUDA BGMV shrink + expand kernels: one launch per layer covers every row's adapter |
| Same-adapter batch | Supported (one adapter, batch of 4) | Supported (D=1, batch of 4 or 8) |
| Mixed-adapter batch | **Not available** on NF4 (see §3.6) | **Yes** — the headline test |
| Role in this report | Reference point for "what standard tooling gives you" | The thing this project is about |

### 3.2 Kernel micro-benchmark (`bench/micro.py`)

Times only the custom adapter op (not a full generate).  
Question: does the op get slower as more distinct adapters are in the pool?

### 3.3 Layer-stack proxy (`bench/micro.py` / `bench/e2e.py --mode kernel-pool`)

Repeats the adapter op across many layers with synthetic activations. Still **not** a full model generate — useful for scaling trends only.

### 3.4 PEFT full generate (`bench/e2e.py --mode peft-sequential`)

- **One adapter, single request**
- **Several adapters, one after another** (includes adapter switch time)
- **Several prompts, same adapter, one batch**

Each generate produces **64 new tokens** (greedy decoding).

### 3.5 Mixed-adapter full generate (`bench/e2e.py --mode persona-generate`) — headline test

Load several adapters into a shared pool, patch the NF4 layers with my Multi-LoRA modules, then run **one** `generate` where **each row uses a different adapter**.

Token counting for left-padded batches uses:  
`batch_size × (output_length − padded_prompt_length)`.  
An earlier counter over-counted short prompts; the tokens/s below use corrected token totals with **unchanged wall-clock times**.

### 3.6 Why there is no PEFT mixed-batch baseline

PEFT's `adapter_names` mixed-batch path does **not** work with bitsandbytes NF4 (`Linear4bit`). Even on plain Linear layers, it splits the batch by adapter inside each layer rather than issuing one BGMV launch. See `bench/baselines/peft_mixed.py`.

---

## 4. Kernel-only timing — do more adapters slow the op?

Source: `profiles/micro_modules_2026-10-05.json`  
One attention projection (`q_proj`), 8 tokens, rank 16.

| Distinct adapters (D) | Average time | vs 1 adapter |
|----------------------:|-------------:|-------------:|
| 1 | 0.195 ms (195 µs) | 1.00× |
| 4 | 0.189 ms (189 µs) | 0.97× |
| 8 | 0.191 ms (191 µs) | 0.98× |

**Takeaway:** at this size, adding adapters barely changes kernel time — the op is dominated by launch overhead, not by the number of distinct adapters.

Other projections (k/v/o/gate/up/down) looked similar (~180–200 µs).

---

## 5. Layer-stack proxy (not full generate)

Adapter math across many layers with synthetic activations. Use for trends, not as tokens per second.

### Attention projections only (32 layers)

| D | Mean time (ms) | Rough ms per token (÷8) |
|--:|---------------:|------------------------:|
| 1 | 25.24 | 3.16 |
| 4 | 25.82 | 3.23 |
| 8 | 22.35 | 2.79 |

### All seven LoRA targets

| D | Mean time (ms) |
|--:|---------------:|
| 1 | 46.03 |
| 4 | 40.54 |

### Pool already loaded (attention only)

| D | Mean time (ms) | vs D=1 |
|--:|---------------:|-------:|
| 1 | 20.75 ± 0.53 | 1.00× |
| 4 | 21.26 ± 1.11 | 1.03× |

---

## 6. Full-generate throughput (tokens per second)

Setup for every row below:

- Base: Mistral-7B-Instruct-v0.2 in NF4  
- New tokens per request: **64**  
- Decoding: greedy  
- Adapters: synthetic/random (see §2)

### 6.1 Baseline — Hugging Face PEFT (no mixed BGMV)

| Scenario | Adapters | Execution | Wall time (s) | Tokens | tokens/s |
|----------|---------:|-----------|--------------:|-------:|------:|
| Single request | 1 | Single generate | 2.57 | 64 | **24.93** |
| Four adapters, one after another | 4 | Sequential (includes adapter switches) | 12.23 | 256 | **20.94** |
| Four requests, **same** adapter, batched | 4 | One batch, one adapter | 5.23 | 256 | **48.99** |

Source: `profiles/e2e_peft_sequential_2026-10-05.json`

### 6.2 Engine — mixed vs same-adapter batch

**D=4** — `profiles/e2e_persona_mixed_D4_2026-10-05.json`

| Scenario | Distinct adapters (D) | Batch size | Wall time (s) | Tokens | tokens/s |
|----------|----------------------:|-----------:|--------------:|-------:|------:|
| Single request (engine path) | 1 | 1 | 2.82 | 64 | **22.72** |
| **Four different adapters, one batch** | 4 | 4 | 7.70 | 256 | **33.25** |
| Four requests, same adapter, batched | 1 | 4 | 6.50 | 256 | **39.41** |

Peak GPU memory ≈ 4.8 GB (base + adapter pool).

**D=8** — `profiles/e2e_persona_mixed_D8_2026-10-05.json`

| Scenario | Distinct adapters (D) | Batch size | Wall time (s) | Tokens | tokens/s |
|----------|----------------------:|-----------:|--------------:|-------:|------:|
| Single request (engine path) | 1 | 1 | 2.85 | 64 | **22.49** |
| **Eight different adapters, one batch** | 8 | 8 | 9.13 | 512 | **56.11** |
| Eight requests, same adapter, batched | 1 | 8 | 7.64 | 512 | **67.05** |

### 6.3 Headline comparison

| Serving style | D=4 tokens/s | D=8 tokens/s | What it shows |
|---------------|----------:|----------:|---------------|
| PEFT: one after another | 20.94 | — | Naive multi-adapter path without a mixed batch |
| PEFT: same adapter, batched | 48.99 | — | Batching helps a lot when every request shares one adapter |
| **Engine: different adapters, one batch** | **33.25** | **56.11** | **The core claim of this project** |
| Engine: same adapter, batched | 39.41 | 67.05 | Same engine; headroom when adapters match |

**Readouts:**

- Mixed D=4 is about **1.6×** the aggregate tokens/s of PEFT sequential D=4.  
- Mixed D=4 is about **0.84×** my own same-adapter batch (expected: mixing is harder than one adapter).  
- Mixed D=8 still fits in 12 GB and keeps scaling aggregate throughput.  
- My single-request path (~22.5 tokens/s) is slightly slower than PEFT single (~24.9): expected, because the engine always runs the multi-adapter kernel path, even for D=1.

---

## 7. Code notes

What made mixed generate work:

1. **`persona/model_patch.py`** — packs/slices adapter weights into the contiguous per-module views the CUDA ops require; adapters are loaded **before** patching.  
2. **`bench/e2e.py --mode persona-generate`** — pool → patch → set per-row adapter slots → Hugging Face `generate`.  
3. PEFT `adapter_names` was **not** used for the mixed headline numbers (NF4 incompatibility).

---

## 8. Limitations

- **Random LoRA weights + synthetic prompts** — throughput study only; not a persona-quality eval.  
- No comparison yet to vLLM / Punica / TensorRT-LLM (separate env planned).  
- Fused BGMV kernel is still a stub; the engine uses separate shrink + expand kernels.  
- Hugging Face `generate` with a fixed batch — not a production scheduler (no paged KV cache, no continuous batching).  
- A token-count bug on left-padded batches was corrected for tokens/s; wall times were not changed.  
- The tree was not git-initialized when results were logged (no commit hash in artifacts).  
- The desktop compositor uses some VRAM and can add timing noise.

Raw JSON and logs: `profiles/`.
