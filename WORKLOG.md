# Persona-Serve worklog

Every optimization and experiment: **change → numbers → hardware-level reason**.  
Record environment: `nvidia-smi`, `nvcc --version`, `torch.__version__`, library versions, git commit hash.

---

## Template (copy per entry)

### YYYY-MM-DD — Short title

**Phase:** P0 | P1 | P2 | …  
**Commit:** `abc1234`

**Change:**  
What changed (one paragraph).

**Before / after:**

| Metric | Before | After |
|--------|--------|-------|
| … | … | … |

**Hardware reason:**  
Why this happened on Ampere / 3060 / memory vs launch bound.

**Nsight / profiler:**  
Link or path under `profiles/` (if any).

**Notes:**  
Correctness tolerances, surprises, next step.

---

<!-- Entries below -->

### 2026-10-05 — Real 1-vs-N micro + PEFT sequential on RTX 3060

**Phase:** P0/P1 measurement (scaffold benches fleshed; persona mixed-batch generate still incomplete)  
**Commit:** Multi-LoRA-LLM-Kernel tree is **not a git repo** on this machine (no commit hash). Original source repo cloned at `/mnt/S/LinuxHome/Projects/Mistral-7B-LLM-custom-trained` (scripts only; no adapters in-tree).  
**ops.py note:** local `persona/ops.py` + `lora_layer.py` forward fix exist on disk; nothing pushed (workspace not git / Origin auth noted by owner).

**Env:**
- GPU: NVIDIA GeForce RTX 3060 12288 MiB, driver 615.71.09
- torch 2.6.0+cu124, transformers 5.18.0, peft 0.21.2, bitsandbytes 0.50.2
- `has_cuda_ops`: True (`bgmv_shrink` + `bgmv_expand`)
- Base HF id: `mistralai/Mistral-7B-Instruct-v0.2` (NF4, allocated ≈4.13 GB)
- Adapters: **SYNTHETIC random** LoRA r=16, targets q/k/v/o (match original `train.py`), 4× under `adapters/random/`
- Personas: **SYNTHETIC** system prompts in `adapters/synthetic_personas.json` (coach/tutor/sarcastic/calm) — style differences in samples are from prompts, not trained LoRA

**Change:** Implemented real `bench/micro.py` and `bench/e2e.py` (peft-sequential + kernel-pool). Fixed MultiLoRALinear.forward (was discarding shrink output). Patched random adapter_config with `peft_type=LORA`.

**Kernel micro (CUDA events, T=8, R=16) — single q_proj shrink+expand:**

| D | mean_ms | vs D=1 |
|---|--------:|-------:|
| 1 | 0.196 | 1.00x |
| 4 | 0.189 | 0.97x |
| 8 | 0.191 | 0.98x |

Per-call ~190 µs; D barely moves time → **launch-bound** at this T. CUDA path ~200 µs vs FP32 ref ~27 ms on same shape.

**Step LoRA proxy (32 layers × qkvo, T=8):**

| D | mean_ms | ms/tok |
|---|--------:|-------:|
| 1 | 25.24 | 3.16 |
| 4 | 25.82 | 3.23 |
| 8 | 22.35 | 2.79 |

Pool-loaded adapters step proxy: D=1 20.75±0.54 ms, D=4 21.26±1.11 ms (1.025x).

**PEFT e2e (NF4 base + random adapters, max_new_tokens=64, greedy):**

| case | D/batch | wall_s | tokens | tok/s |
|------|--------:|-------:|-------:|------:|
| D=1 single | 1 | 2.568 | 64 | 24.93 |
| D=4 sequential sum (incl. adapter load) | 4 | 12.227 | 256 | 20.94 |
| batch=4 same adapter | 4 | 5.226 | 256 | 48.99 |

**Hardware reason:** At T=8, each BGMV call moves little data; overhead is launch/latency, so D=1 vs D=4 looks flat in the kernel microbench. PEFT sequential pays ~4× generate time (+ load) vs one request; same-adapter batch≈2× tok/s of single (batching helps base NF4 path). True mixed-adapter batched decode via persona engine **not yet measured** (patch path still incomplete for fused A_qkv wiring).

**Artifacts:** `profiles/micro_*.json`, `profiles/e2e_*.json`, `profiles/e2e_peft_sequential_2026-10-05.log`

**Blockers / still scaffold:**
- `persona-generate` mixed-batch path not run (MultiLoRALinear + A_qkv packing needs more work)
- PEFT mixed `adapter_names` batch (S2) not implemented
- No trained personality adapters in original repo; random weights only
- Multi-LoRA tree not git-initialized here; cannot record commit; Origin push of ops/test fixes still blocked per owner note

### 2026-10-05 — Mixed-adapter batched decode (BGMV) + BENCHMARKS.md

**Phase:** P1 mixed-batch generate working on 3060  
**Commit:** not a git repo on this machine

**Change:**  
Fixed `model_patch` to slice packed `A_qkv`/`A_gu` into contiguous per-module A pools (load adapters before patch). Implemented `bench/e2e.py --mode persona-generate` for true simultaneous multi-adapter HF generate. Confirmed PEFT `adapter_names` unusable with bitsandbytes Linear4bit. Wrote `docs/BENCHMARKS.md`. Corrected left-pad token counting for mixed batches.

**Before / after:**

| Metric | Before | After |
|--------|--------|-------|
| Mixed-adapter e2e | blocked (A_qkv packing) | D=4 **33.25** tok/s; D=8 **56.11** tok/s |
| PEFT sequential D=4 | 20.94 tok/s | retained |
| PEFT same-batch D=4 | 48.99 tok/s | retained |

**Hardware reason:**  
Kernel micro still flat vs D (launch-bound at T=8). E2E aggregate tok/s rises with batch size because the NF4 base path amortizes across rows; mixed vs same-adapter gap (~16%) is the Multi-LoRA tax on this unfused path.

**Nsight / profiler:**  
`profiles/e2e_persona_mixed_D4_2026-10-05.json`, `profiles/e2e_persona_mixed_D8_2026-10-05.json`, `docs/BENCHMARKS.md`

**Notes:**  
Adapters still synthetic random. D=8 VRAM OK (~4.8 GB allocated). Token counts for mixed D=4/D=8 corrected post-hoc from wall_s (left-pad overcount); walls unchanged.

