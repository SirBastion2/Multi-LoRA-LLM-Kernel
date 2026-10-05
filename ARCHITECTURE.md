# Persona-Serve: Multi-Personality LLM Serving with Custom Batched-LoRA CUDA Kernels

**Project architecture document** · v1.0 · September 30, 2026
**Owner:** Sebastian Villalba
**Hardware:** NVIDIA GeForce RTX 3060 12 GB (Ampere, compute capability 8.6)
**OS:** CachyOS, native Linux

---

## 0. What this project is

Persona-Serve serves many fine-tuned "personality" adapters of Mistral-7B at the same time, from one 4-bit base model, on a single 12 GB consumer GPU. Each personality is a LoRA adapter trained with the existing QLoRA pipeline. The naive options are running one personality at a time or keeping a full model copy per personality; the second doesn't fit in 12 GB. Instead, a single decode batch mixes requests for different personalities. The base model's weights are shared, and custom CUDA C++ kernels apply each request's own adapter inside one kernel launch.

The technique comes from Punica (MLSys 2024), which introduced batched-gather LoRA kernels. It now ships in production servers, including vLLM and NVIDIA TensorRT-LLM. This project re-implements it from first principles on consumer hardware, measures it against those references, and pushes it further with fused kernels. Credit the technique to Punica (and S-LoRA, a separate system from the same period) in the README and in interviews. The engineering, measurements, and optimizations are yours.

| Deliverable | What it proves |
|---|---|
| Custom CUDA kernels (shrink, expand, fused variants) with a correctness suite | You can write, test, and debug GPU kernels in C++ |
| Microbenchmark report: µs per call, GB/s, and % of the bandwidth ceiling, vs. PyTorch and Punica | You measure against hardware limits and strong references, not strawmen |
| A batched decode engine that serves N personalities concurrently | You can integrate kernels into a real ML system |
| Charts: LoRA overhead and throughput vs. the number of distinct adapters per batch, plus a vLLM comparison | Your headline claims are defensible |
| A worklog: every optimization with before/after numbers and a hardware-level reason | You understand why each change worked |
| A demo: several personalities chatting at once from one GPU | A recruiter gets it in ten seconds |

---

## 1. Platform decision: native Linux (your CachyOS install)

Build, profile, and benchmark everything on native Linux, not Windows and not WSL2. Keep every benchmark on the same OS, driver, and machine so the numbers stay comparable across the whole project.

| Requirement | Native Linux | WSL2 | Native Windows |
|---|---|---|---|
| vLLM (end-to-end baseline) | Supported | Runs, inside a virtualization layer | Not supported natively |
| Triton (vLLM's LoRA kernels are written in Triton) | Official | Through Linux | Community fork only |
| Building Punica or FlashAttention from source | Primary target | Usually works, slower builds | Frequently painful |
| bitsandbytes 4-bit | Full support | Works | Supported in recent versions |
| Nsight Compute hardware counters | Full, after enabling non-admin profiling | Historically limited | Full |
| Timing stability | Bare metal | Extra virtualization layer | GeForce cards run under the WDDM driver model, which adds launch latency and scheduling noise |
| Your machine | Already working (`nvidia-smi` shows the GPU) | Not needed | Not needed |

Linux wins on every row that matters for this project: the baseline you most need (vLLM) only runs natively on Linux, the profiler you'll live in is least restricted there, and your machine already works.

### 1.1 Rolling-release hardening for CachyOS

CachyOS is Arch-based and rolling. That's great for fresh drivers and bad for reproducibility. Neutralize the risks before writing any code.

| Risk | Symptom | Mitigation |
|---|---|---|
| System GCC is newer than `nvcc` supports | `unsupported GNU version` when compiling `.cu` files | Install a supported GCC inside the project environment and point `nvcc` at it (setup below) |
| System Python is newer than the published wheels | `pip` can't find PyTorch or vLLM wheels | Use a project environment with a Python version both projects publish wheels for |
| CUDA toolkit version ≠ PyTorch's CUDA version | Extension builds fail with a CUDA version-mismatch error | Install `nvcc` in the environment matching `torch.version.cuda`; take only the driver from pacman |
| A driver or kernel update lands mid-project | `nvidia-smi` fails after a reboot | Add the NVIDIA packages to `IgnorePkg` in `/etc/pacman.conf` for the project's duration; update deliberately; if your root is btrfs, snapshot before updating |
| Profiler permissions | `ncu` reports `ERR_NVGPUCTRPERM` | Put `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/nvidia-profiling.conf`, rebuild the initramfs (`sudo mkinitcpio -P`, or `dracut` if your install uses it), and reboot |
| The 3060 also drives your display | The compositor takes VRAM and adds timing jitter | Note idle VRAM in `nvidia-smi`; run benchmarks from a TTY or a minimal session; close browsers and games; enable persistence mode (`sudo nvidia-smi -pm 1`); lock clocks with `nvidia-smi -lgc` if your driver allows it |

### 1.2 Environment setup

```bash
# 0. Driver: system-wide, from pacman. This must list the RTX 3060.
nvidia-smi

# 1. Project environment (micromamba shown; uv plus a pinned system CUDA also works)
micromamba create -n persona -c conda-forge python=3.12   # choose a version PyTorch AND vLLM publish wheels for
micromamba activate persona
pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>   # pick a CUDA build listed on pytorch.org
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# expected capability: (8, 6)

# 2. nvcc matching torch.version.cuda, plus a host compiler that this nvcc release supports
micromamba install -c nvidia "cuda-toolkit=<torch.version.cuda>"
micromamba install -c conda-forge "gxx_linux-64=<a GCC version supported by that CUDA release>"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.6"          # compile only for the 3060: faster builds
export NVCC_PREPEND_FLAGS="-ccbin $CXX"    # conda's compiler package sets $CXX on activation
nvcc --version

# 3. Libraries for your engine
pip install transformers peft bitsandbytes safetensors accelerate pytest numpy matplotlib nvtx

# 4. Baselines live in a SEPARATE environment, because vLLM pins its own PyTorch version
micromamba create -n vllm-baseline -c conda-forge python=3.12
micromamba activate vllm-baseline
pip install vllm
```

For the standalone C++ harness, pass the same compiler to CMake with `-DCMAKE_CUDA_HOST_COMPILER=$CXX -DCMAKE_CUDA_ARCHITECTURES=86`.

Every benchmark result file records `nvidia-smi` output, `nvcc --version`, `torch.__version__`, library versions, and the git commit hash. A number without its environment is not a result.

---

## 2. Architecture

### 2.1 Data flow

```
 Requests (persona_id, prompt, sampling params)
        │
        ▼
 ┌──────────────┐  acquire / release  ┌───────────────────────┐   async upload   ┌──────────────────┐
 │  Scheduler   │ ──────────────────▶ │  Adapter pool (GPU)   │ ◀─────────────── │ Pinned CPU store │
 │  batch plan  │                     │  S slots, per-layer   │   side stream    │  all adapters    │
 └──────┬───────┘                     │  A / B tensors        │                  └──────────────────┘
        │ token_slots[T] (int32, GPU) └───────────┬───────────┘
        ▼                                         │
 ┌──────────────────────────── One decode step (CUDA Graph) ──────────────────────────────┐
 │ embed → 32 × { RMSNorm → q, k, v = Base4bit(x) + LoRA_qkv(x)                          │
 │                RoPE → KV-cache write → attention → o = Base4bit(a) + LoRA_o(a)         │
 │                RMSNorm → g, u = Base4bit(x) + LoRA_gu(x) → h = SiLU(g) · u             │
 │                down = Base4bit(h) + LoRA_down(h) }                                    │
 │ → final RMSNorm → lm_head → sampler → next token per request                          │
 └───────────────────────────────────────────────────────────────────────────────────────┘

   LoRA_*(x) = BGMV_expand( BGMV_shrink(x, A_pool, token_slots), B_pool, token_slots )
```

Every decode step, the scheduler decides which requests are in the batch and writes one int32 slot index per token into `token_slots` on the GPU. All LoRA layers across the 32 decoder layers read that same tensor, so there is exactly one small host-to-device write per step. The base model's 4-bit weights are shared by every request. Only the small A and B adapter matrices differ per personality, and the BGMV kernels gather the right ones for each token.

### 2.2 Model facts (Mistral-7B)

These are the standard Mistral-7B values; confirm them in your base checkpoint's `config.json`.

| Property | Value |
|---|---|
| Hidden size | 4096 |
| Decoder layers | 32 |
| Attention heads / KV heads | 32 / 8 (grouped-query attention) |
| Head dimension | 128 |
| MLP intermediate size | 14336 |
| Vocabulary | 32000 (v0.1, v0.2) or 32768 (v0.3) |

| Module | In (K) | Out (N) | `lora_A.weight` shape | `lora_B.weight` shape |
|---|---|---|---|---|
| `self_attn.q_proj` | 4096 | 4096 | [r, 4096] | [4096, r] |
| `self_attn.k_proj` | 4096 | 1024 | [r, 4096] | [1024, r] |
| `self_attn.v_proj` | 4096 | 1024 | [r, 4096] | [1024, r] |
| `self_attn.o_proj` | 4096 | 4096 | [r, 4096] | [4096, r] |
| `mlp.gate_proj` | 4096 | 14336 | [r, 4096] | [14336, r] |
| `mlp.up_proj` | 4096 | 14336 | [r, 4096] | [14336, r] |
| `mlp.down_proj` | 14336 | 4096 | [r, 14336] | [4096, r] |

Your adapter may target only some of these modules. `adapter_config.json` records `target_modules`, `r`, `lora_alpha`, and `use_rslora`; the pool code reads all four.

### 2.3 The math

For one request using adapter `a`, each targeted linear layer computes:

```
y = W·x + s_a · B_a · (A_a · x)        s_a = lora_alpha / r     (rsLoRA: lora_alpha / sqrt(r))
```

In a mixed batch, token `t` carries a slot index `σ(t)`, where −1 means "no adapter". The LoRA term splits into two kernels:

```
shrink:  v[t]  = A[σ(t)] · x[t]      K → r    (r is small: 8 to 64)
expand:  y[t] += B'[σ(t)] · v[t]     r → N    (B' = s · B, with the scale folded in once, at load time)
```

This per-token gather is BGMV (batched gather matrix-vector), the decode-specialized kernel in Punica's code. The Punica paper's headline kernel, SGMV (segmented gather matrix-vector), handles contiguous segments of tokens that share one adapter, which is the prefill case. Version 1 uses BGMV for decode and a plain single-adapter matmul path for prefill; SGMV is a later optimization.

The pool layouts deliberately match PEFT's own tensor layouts. `lora_A.weight` is [r, K], so each rank row is contiguous along K, which is exactly what shrink reads. `lora_B.weight` is [N, r], so each output element's r values are contiguous, which is exactly what expand reads. No transposes are needed at load time.

### 2.4 Memory budget (12 GB)

These are estimates. Verify each line with `torch.cuda.memory_allocated()` and `nvidia-smi` in Phase 0.

| Item | Estimate | How it's computed |
|---|---|---|
| Base weights, bitsandbytes NF4 | ≈ 4.1 GB | 6.98B linear-layer params × 0.5 bytes, plus quantization constants, plus FP16 embeddings and lm_head |
| CUDA context and allocator overhead | ≈ 0.6–1.0 GB | Measured |
| Desktop compositor (same GPU) | ≈ 0.2–0.6 GB | Idle `nvidia-smi` |
| KV cache | ≈ 2.0 GB | 128 KiB per token (2 × 32 layers × 8 KV heads × 128 dims × 2 bytes) × 8 sequences × 2048 tokens |
| Adapter pool | ≈ 1.3 GB | ≈ 84 MB per adapter (r = 16, all seven modules) × 16 slots |
| Activations, logits, workspace | ≈ 0.3–0.8 GB | Small during decode; long-prompt prefill is the peak |
| **Total** | **≈ 8.5–9.8 GB** | Leaves room for more adapter slots or longer contexts |

Adapter size formula: `bytes = 2 × r × layers × Σ_modules (K + N)`. For r = 16 on all seven modules, that's 2 × 16 × 32 × 81,920 ≈ 84 MB. Recompute it with your real `r` and `target_modules`.

### 2.5 Performance ceilings

Report every result against a ceiling, not only against a baseline. These are back-of-envelope numbers; your measured versions go in the worklog.

| Quantity | Formula | Value on the 3060 |
|---|---|---|
| Memory bandwidth | Spec | 360 GB/s |
| Batch-1 decode, NF4 base | ≈ 3.86 GB read per token ÷ 360 GB/s | ≈ 10.7 ms per token, so at most ≈ 93 tokens/s |
| LoRA traffic per decode step | D distinct adapters × 84 MB ÷ 360 GB/s | At D = 8: 0.67 GB, so at least ≈ 1.9 ms |
| Kernel launch floor | Measure an empty kernel | A few µs per launch |
| Batched base compute | 2 × 7.1B × T FLOPs vs. ≈ 13 TFLOPS FP32 | Compute time ≈ memory time near T ≈ 9 |
| Example step: T = 8, D = 8, 512-token contexts | (3.86 + 0.67 + 0.5 GB of KV reads) ÷ 360 GB/s | At least 13.9 ms, so at most ≈ 575 tokens/s aggregate |

Two consequences shape the whole design.

First, at small batch sizes each individual LoRA kernel moves very little data. A q_proj shrink at T = 8 with 8 distinct adapters reads about 1 MB, so it's limited by launch overhead and latency, not bandwidth. The unfused design issues 32 layers × 7 modules × 2 kernels = 448 LoRA launches per step. Cutting launches (fused QKV and gate/up shrinks, CUDA Graphs) therefore matters as much as making each kernel fast.

Second, summed over a whole step, adapter bytes grow linearly with the number of distinct adapters. That gives the LoRA path a real bandwidth floor at larger D. Both effects belong in the worklog: per-kernel time against the launch floor, and per-step LoRA time against the bandwidth floor.

---

## 3. Components

### 3.1 CUDA kernels (`csrc/`)

| Kernel | Inputs → output | Version 1 launch shape | Key idea |
|---|---|---|---|
| `bgmv_shrink` | x [T, K], a_pool [S, R, K], slots [T] → v [T, R] in FP32 | grid (T, ⌈R/4⌉), block (32, 4): one warp per (token, rank row) | 16-byte vector loads, FP32 accumulation, warp-shuffle reduction |
| `bgmv_expand` | v [T, R], b_pool [S, N, R], slots → y [T, N], added in place | grid (T, ⌈N/256⌉), block 256: one thread per output element | Each thread reads its R contiguous values, so a warp reads one contiguous span |
| `bgmv_shrink` on stacked A (fused QKV, fused gate/up) | x read once; A stacked as [S, 3r, K] or [S, 2r, K] | Same as shrink | One launch and one read of x for two or three modules |
| `bgmv_expand_slice` | Slices of v → separate output tensors | Same as expand | Pairs with the fused shrink |
| `gemv_fp16` (warm-up only) | x [K], W [N, K] → y [N] | One warp per output row | Learn the pattern first; benchmark it against cuBLAS |

Design invariants, enforced with `TORCH_CHECK` in the host wrappers:

| Invariant | Reason |
|---|---|
| Pool tensors are contiguous, and K and N are multiples of 8 | Enables 16-byte vector loads; Mistral's dimensions are all multiples of 128 |
| R is a template parameter (8, 16, 32, 64) | Loops over R fully unroll and keep values in registers |
| The element type is templated (FP16 and BF16) | Match the compute dtype your adapters were trained with |
| Kernels launch on `at::cuda::getCurrentCUDAStream()` and never synchronize with the host | Required for correctness with streams and for CUDA Graph capture |
| Callers preallocate outputs | No allocation inside the op; keeps it graph-safe |

**Shrink, version 1.** This is a reference to read, understand, and rewrite from a blank page, not final code.

```cuda
// v[t, r] = sum_k x[t, k] * A[slot[t], r, k]      (A packed as [S, R, K], PEFT's lora_A layout)
template <typename T, int WARPS>
__global__ void bgmv_shrink_kernel(const T* __restrict__ x,        // [T_tok, K]
                                   const T* __restrict__ a_pool,   // [S, R, K]
                                   const int* __restrict__ slots,  // [T_tok], -1 = no adapter
                                   float* __restrict__ v,          // [T_tok, R]
                                   int K, int R) {
    const int t    = blockIdx.x;
    const int r    = blockIdx.y * WARPS + threadIdx.y;   // one warp per rank row
    const int lane = threadIdx.x;                        // 0..31
    if (r >= R) return;                                  // r is uniform per warp: the whole warp exits together
    const int slot = slots[t];
    if (slot < 0) { if (lane == 0) v[t * R + r] = 0.f; return; }

    const T* a_row = a_pool + ((size_t)slot * R + r) * K;   // size_t: avoid 32-bit overflow
    const T* x_row = x + (size_t)t * K;

    float acc = 0.f;
    for (int k = lane * 8; k < K; k += 32 * 8) {         // each lane loads 8 values (16 bytes) per iteration
        uint4 av = *reinterpret_cast<const uint4*>(a_row + k);
        uint4 xv = *reinterpret_cast<const uint4*>(x_row + k);
        acc += dot8<T>(av, xv);                          // unpack four half2 pairs each, FMA in FP32
    }
    for (int off = 16; off > 0; off >>= 1)               // warp-level tree reduction, no shared memory
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) v[t * R + r] = acc;
}
// launch: grid(T_tok, ceil(R / WARPS)), block(32, WARPS), stream = current PyTorch stream
```

Why the loads are coalesced: on each iteration, lane i reads 16 bytes at offset 16·i, so the warp reads 512 contiguous bytes of the A row and 512 contiguous bytes of x.

**Expand, version 1.**

```cuda
// y[t, n] += sum_r v[t, r] * B'[slot[t], n, r]    (B' pre-scaled by s; PEFT's lora_B layout [N, R])
template <typename T, int R>
__global__ void bgmv_expand_kernel(const float* __restrict__ v,     // [T_tok, R]
                                   const T* __restrict__ b_pool,    // [S, N, R]
                                   const int* __restrict__ slots,
                                   T* __restrict__ y,               // [T_tok, N], updated in place
                                   int N) {
    const int t = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    const int slot = slots[t];
    if (slot < 0 || n >= N) return;                      // safe: there is no __syncthreads() below

    const T* b_row = b_pool + ((size_t)slot * N + n) * R; // R contiguous values per thread
    float acc = 0.f;
    #pragma unroll
    for (int r = 0; r < R; ++r)                          // v[t, r] is the same address across the warp: broadcast
        acc += v[t * R + r] * to_float(b_row[r]);
    const size_t idx = (size_t)t * N + n;
    y[idx] = from_float<T>(to_float(y[idx]) + acc);
}
// launch: grid(T_tok, ceil(N / 256)), block(256)
```

With R = 16 in FP16, each thread reads 32 contiguous bytes, so a warp reads 1 KB in one contiguous span. The optimized version replaces the scalar loop with two 16-byte loads per thread.

Pitfalls worth knowing cold, because they're classic interview questions:

| Pitfall | What goes wrong | Guard |
|---|---|---|
| Early `return` before `__syncthreads()` | Deadlock or undefined behavior when some threads never reach the barrier | Keep barriers out of divergent paths; expand version 1 uses no shared memory for exactly this reason |
| Misaligned vector loads | `uint4` loads need 16-byte alignment; tensor views and slices can break it | Check `data_ptr() % 16 == 0` in the host wrapper |
| 32-bit index overflow | `slot × R × K` grows fast with large pools | Compute offsets in `size_t` |
| FP16 accumulation | Precision loss and overflow over K = 14336 | Accumulate in FP32 and convert once at the end |
| Default-stream launches | Races with PyTorch's stream; breaks CUDA Graph capture | Always launch on `at::cuda::getCurrentCUDAStream()` |

### 3.2 PyTorch bindings (`csrc/bindings.cpp`)

```cpp
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void bgmv_shrink_cuda(const at::Tensor& x, const at::Tensor& a_pool,
                      const at::Tensor& slots, at::Tensor& v);   // validates, then launches on the current stream
void bgmv_expand_cuda(const at::Tensor& v, const at::Tensor& b_pool,
                      const at::Tensor& slots, at::Tensor& y);

TORCH_LIBRARY(persona, m) {
  m.def("bgmv_shrink(Tensor x, Tensor a_pool, Tensor slots, Tensor(a!) v) -> ()");
  m.def("bgmv_expand(Tensor v, Tensor b_pool, Tensor slots, Tensor(a!) y) -> ()");
}
TORCH_LIBRARY_IMPL(persona, CUDA, m) {
  m.impl("bgmv_shrink", &bgmv_shrink_cuda);
  m.impl("bgmv_expand", &bgmv_expand_cuda);
}
```

Build with `torch.utils.cpp_extension`: `load()` for fast iteration during development, and a `setup.py` for installs. Registering through `TORCH_LIBRARY` (rather than a plain pybind module) makes the ops visible to `torch.ops.persona.*`, and `Tensor(a!)` in the schema declares which argument is mutated. The namespace is `persona` because short names like `mps` collide with existing PyTorch backends. Each host wrapper checks device, dtype, contiguity, shapes, and alignment before launching.

### 3.3 Adapter pool (`persona/adapter_pool.py`)

The pool parses PEFT adapters, validates that they share one configuration, folds `s = lora_alpha / r` into B, converts to the compute dtype, and packs everything into per-layer pool tensors with S slots.

| Tensor (per layer) | Shape | Holds |
|---|---|---|
| `A_qkv` | [S, 3r, 4096] | q, k, v `lora_A` stacked along the rank dimension |
| `B_q`, `B_k`, `B_v` | [S, 4096, r], [S, 1024, r], [S, 1024, r] | Pre-scaled `lora_B` |
| `A_o`, `B_o` | [S, r, 4096], [S, 4096, r] | o_proj |
| `A_gu` | [S, 2r, 4096] | gate and up `lora_A` stacked |
| `B_gate`, `B_up` | [S, 14336, r] each | Pre-scaled |
| `A_down`, `B_down` | [S, r, 14336], [S, 4096, r] | down_proj |

PEFT stores weights under keys like `base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight` (and `lora_B`, and `mlp.*` for the MLP modules) in `adapter_model.safetensors`. Saved adapters are often FP32; convert them once at load time.

Version 1 keeps every adapter resident on the GPU. Version 2 adds paging: pinned CPU master copies, an LRU slot table with reference counts (a slot is never evicted while an in-flight batch uses it), and uploads on a side stream with a CUDA event that the decode stream waits on before using the slot.

`adapters/make_random_adapters.py` creates N adapters with the same shapes as your real ones, for scaling tests. Kernel speed doesn't depend on the values, so random adapters are legitimate for performance and your real personalities are for correctness and the demo.

Train any additional personalities with the same `r`, `target_modules`, and compute dtype as the first one. Uniform adapters pack into one pool without padding; padding smaller ranks up to a maximum R wastes bandwidth.

### 3.4 Model integration (`persona/lora_layer.py`, `persona/model_patch.py`)

Load the base model with `transformers` and a `BitsAndBytesConfig` that matches your QLoRA training setup: NF4, the same compute dtype, and the same double-quantization setting. Don't wrap it in `PeftModel`; the pool owns the adapters. Replace each targeted `Linear4bit` with a `MultiLoRALinear` (version 1: one shrink per module). Later, patch the attention and MLP blocks to use the fused shrinks (optimization step K6 in section 7).

```python
class LoRAContext:
    """One per engine. Set once per forward pass; read by every LoRA layer."""
    def __init__(self, max_tokens, device):
        self.token_slots = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
        self.row_slots = None          # [batch] slot per request, set by the scheduler
        self.enabled = True            # False = base model only (for overhead measurements)

class MultiLoRALinear(torch.nn.Module):
    def __init__(self, base, ctx, a_pool, b_pool):
        super().__init__()
        self.base, self.ctx, self.a_pool, self.b_pool = base, ctx, a_pool, b_pool

    def forward(self, x):
        y = self.base(x)                                   # bitsandbytes Linear4bit now; Marlin later
        if not self.ctx.enabled:
            return y
        x2d, y2d = x.reshape(-1, x.shape[-1]), y.view(-1, y.shape[-1])
        T = x2d.shape[0]
        v = torch.empty((T, self.a_pool.shape[1]), dtype=torch.float32, device=x.device)
        slots = self.ctx.token_slots[:T]
        torch.ops.persona.bgmv_shrink(x2d.contiguous(), self.a_pool, slots, v)
        torch.ops.persona.bgmv_expand(v, self.b_pool, slots, y2d)
        return y

def make_slot_hook(ctx):
    """Forward pre-hook for the top-level model: expand per-request slots to per-token slots once."""
    def hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch, seq = input_ids.shape
        ctx.token_slots[: batch * seq] = ctx.row_slots.repeat_interleave(seq)
    return hook

# model.register_forward_pre_hook(make_slot_hook(ctx), with_kwargs=True)
```

Run everything under `torch.inference_mode()`, which also makes the in-place add into the base output safe. The pre-hook handles both phases: during prefill each request's slot repeats across its prompt length, and during decode there is one token per request.

### 3.5 Base-model path: the decision that determines your end-to-end numbers

The LoRA kernels only add work on top of the base model, so the base path sets the ceiling for everything else.

| Option | Speed with more than one token per call | Quality with your adapters | Effort | Use it for |
|---|---|---|---|---|
| A. bitsandbytes NF4 (version 1) | Likely slow. In the bitsandbytes versions I know, the fused 4-bit GEMV runs only when there is a single token; multi-token inputs dequantize the whole weight to FP16 first, several times the memory traffic. Verify on your installed version with Nsight Systems in Phase 0 | Exact: your adapters were trained against NF4 | None | Correctness reference and LoRA-overhead measurements |
| B. Marlin INT4 (GPTQ-quantized, symmetric, group size 128) | Fast: near-ideal speedups up to batch sizes of 16–32 on Ampere | Some drift expected, because the adapters saw NF4 during training; measure it | Medium: quantize with the GPTQ variant in Marlin's repo, then pack the layers | Throughput benchmarks |
| C. Your own NF4 skinny-GEMM with the LoRA shrink fused in (stretch) | Goal: near the bandwidth ceiling for T ≤ ~8 on CUDA cores; larger batches need tensor cores | Exact | High | The advanced finale (optimization step K9) |

The practical consequence: make **LoRA overhead per decode step**, measured as step time with LoRA minus step time with the LoRA layers disabled, your primary system metric. It's independent of how fast the base happens to be. Quote absolute tokens/s only once you're on option B or C. If option A's batched path turns out to be slow, that measurement is itself a good worklog entry: it explains why production servers use kernels like Marlin.

### 3.6 Decode engine (`persona/engine.py`)

**Engine version 1 (Phase 3): static batching.** Use Hugging Face `generate` with left padding. All requests in a batch start together, and each row keeps a fixed slot for the whole generation. This is the fastest path to the headline chart. Hugging Face's eager `generate` has substantial per-step Python overhead, so treat its absolute numbers as relative comparisons within the same harness.

**Engine version 2 (Phase 5): continuous batching.** Write your own decode loop over the Hugging Face model's modules:

| Piece | Design |
|---|---|
| KV cache | Preallocated per layer as [B_max, max_len, 8 KV heads, 128], with one row per active request and per-row lengths |
| Positions and RoPE | Each row has its own position; apply rotary embeddings with per-row position indices |
| Attention | PyTorch SDPA over the cache with a per-row length mask. FlashAttention's `flash_attn_with_kvcache`, which accepts per-row cache lengths, is an optional faster path |
| Prefill | One request at a time with its single adapter, using plain matmuls (no gather needed), then the request joins the decode batch |
| Decode | All active rows together, mixed adapters, through BGMV |
| Scheduling | Each step: admit waiting requests into free rows, run one decode step, retire rows that hit EOS or max tokens, release their adapter references |
| CUDA Graphs | Capture one decode step per batch-size bucket (1, 2, 4, 8, 16) with static input buffers; replay each step. Test early whether the base-model ops are capture-safe |
| Sampling | Greedy for correctness tests; temperature and top-p per request for the demo |

### 3.7 Demo (`demo/`)

A FastAPI server with a streaming `/chat` endpoint that takes a persona ID and a message, plus a small web page with several chat panes open at once, each on a different personality. Show live stats: tokens/s, current batch size, distinct adapters in the batch, and resident slots. You already built a FastAPI backend at SteelHacks, so reuse that pattern. If the personalities are modeled on real streamers, give them generic names in anything public, or get permission; impersonating real people is a reputational risk a portfolio project doesn't need.

### 3.8 Repository layout

```
persona-serve/
├── README.md                 results, charts, how to reproduce, credit to Punica / S-LoRA
├── WORKLOG.md                every experiment: change → numbers → hardware reason
├── env/                      environment files, exact versions, setup notes
├── csrc/                     CUDA C++: the part you must be able to explain line by line
│   ├── common.cuh            error-check macros, vector-load and dot8 helpers
│   ├── gemv_fp16.cu          warm-up kernel, compared against cuBLAS
│   ├── bgmv_shrink.cu
│   ├── bgmv_expand.cu
│   ├── bgmv_fused.cu         later: fused QKV / gate-up, fused shrink + expand
│   └── bindings.cpp          TORCH_LIBRARY registration
├── standalone/               pure C++/CUDA harness with no Python (this closes the C++ gap)
│   ├── CMakeLists.txt
│   ├── device_buffer.hpp     RAII wrapper around cudaMalloc / cudaFree
│   ├── test_bgmv.cu          CPU reference + randomized tests
│   └── bench_bgmv.cu         cudaEvent timing, L2 flush
├── persona/                  Python package
│   ├── ops.py                loads the extension; PyTorch reference implementations
│   ├── adapter_pool.py       PEFT parsing, packing, slot table, LRU paging
│   ├── lora_layer.py         LoRAContext, MultiLoRALinear
│   ├── model_patch.py        swaps base linears in the Hugging Face model
│   ├── engine.py             v1 static batching; v2 continuous batching + CUDA Graphs
│   └── sampling.py
├── tests/                    pytest: kernels, isolation, layer-level, end-to-end
├── bench/                    micro.py, e2e.py, baselines/ (sequential.py, peft_mixed.py, vllm_client.py)
├── profiles/                 Nsight Systems / Nsight Compute reports referenced in WORKLOG.md
├── adapters/                 real personalities + make_random_adapters.py
└── demo/                     FastAPI server + minimal web page
```

---

## 4. Correctness plan

Nothing gets benchmarked until it passes these tests. The reference implementation for kernel tests is plain PyTorch in FP32: gather each token's A and B, then two matmuls.

| Test | What it checks | Pass criterion |
|---|---|---|
| Kernel vs. reference | Shrink and expand on random T ∈ [1, 64], R ∈ {8, 16, 32, 64}, all seven module shapes, D ∈ {1 … T} distinct adapters, FP16 and BF16 | `torch.testing.assert_close` with tolerances documented in the worklog |
| Adapter isolation | Change adapter s's weights; only tokens assigned to slot s change | Exact |
| No-adapter rows | Tokens with slot −1 leave y bitwise unchanged | Exact |
| Edge cases | T = 1; every token on one slot; every token on a different slot; all slots in use; non-contiguous or misaligned inputs | Correct output, or a clear error from the host wrapper |
| Layer level | `MultiLoRALinear` vs. PEFT's LoRA layer for the same adapter and input | Close within tolerance |
| End to end | Greedy generation with your engine vs. Hugging Face + PEFT for one adapter | Per-step logits within tolerance; identical tokens for at least the first N steps. Later divergence can come from FP16 rounding order; document it |
| Hidden shapes | A shape set you never tuned or benchmarked on | Passes, which guards against kernels that only work on the shapes you optimized |
| Memory safety | `compute-sanitizer --tool memcheck` and `--tool racecheck` on the standalone harness | No errors |

---

## 5. Benchmark plan

### 5.1 Methodology

These rules are non-negotiable. Published kernel-optimization results have been inflated by exactly the mistakes they prevent.

| Rule | Why |
|---|---|
| Warm up at least 10 iterations before timing | First calls include lazy initialization and cold caches |
| Time with CUDA events on the working stream, and synchronize before reading them | Host timers miss asynchronous GPU work |
| Synchronize every stream your code touches before stopping the timer | Work launched on side streams can escape the measurement entirely |
| Report the median and p10/p90 of at least 100 iterations | A single run is noise |
| For cold-cache numbers, flush L2 between iterations by writing a buffer much larger than the 3060's 3 MB L2 (32 MB is plenty); report warm-cache numbers separately | Small adapters otherwise stay cached and look faster than they are |
| Check correctness on shapes you never tuned on | Kernels can overfit to benchmark shapes |
| Fixed environment: persistence mode, stable clocks, nothing else on the GPU; log all versions and the commit hash | Reproducibility |
| Confirm headline numbers in Nsight Systems | Catches harness bugs that timers can't |

### 5.2 Kernel microbenchmarks

| Axis | Values |
|---|---|
| Tokens per call, T | 1, 2, 4, 8, 16, 32, 64 |
| Rank, R | 8, 16, 32, 64 |
| Module shapes | All seven from section 2.2 |
| Distinct adapters, D | 1, 2, 4, 8, T |
| Adapter assignment | Identical (all one adapter), uniform, skewed (Zipf), distinct (one per token), mirroring Punica's workload categories |

For each point, record µs per call, effective bandwidth (unique bytes touched ÷ time), percent of 360 GB/s, and the ratio to the measured launch floor.

| Baseline | What it is | Required? |
|---|---|---|
| M1. PyTorch loop | For each distinct adapter: `index_select` its tokens, two matmuls, `index_add_` back | Required |
| M2. PyTorch gathered bmm | Materialize `A_pool[slots]` and `B_pool[slots]`, then `torch.bmm` | Required |
| M3. Punica's kernels | The original BGMV/SGMV. Prebuilt wheels include compute capability 8.6 but target older CUDA versions, so you may need a source build | Strongly recommended |
| M4. vLLM's LoRA ops | Triton kernels derived from Punica; their Python API changes between versions | Optional |

### 5.3 System benchmarks

| Metric | Definition |
|---|---|
| **LoRA overhead per step (headline)** | Decode-step time with LoRA minus decode-step time with `ctx.enabled = False`, as a function of D |
| Aggregate throughput | Output tokens/s across all requests, as a function of D |
| Latency | Time per output token (p50, p95) and time to first token |
| Capacity | Peak GPU memory; maximum concurrent personalities before running out of memory |

| System | Description |
|---|---|
| S1. Sequential per persona | Group requests by adapter and serve one group at a time: the naive approach |
| S2. PEFT mixed batches | Recent PEFT versions accept per-sample `adapter_names` and loop over adapters internally; use it if your version supports it |
| S3. Persona-Serve | Your engine |
| S4. vLLM with LoRA enabled | Separate environment, same GPU, same prompts and output lengths. Set `--enable-lora`, `--max-loras`, `--max-lora-rank` ≥ your r, and tune `--max-model-len` and `--gpu-memory-utilization` to fit 12 GB. Match the base quantization as closely as your vLLM version allows, and document the exact flags |
| S5. Upper bound | One adapter merged into the base, identical workload only |

Use held-out transcript prompts with a fixed output length (for example, 128 tokens) and the four assignment patterns from 5.2.

### 5.4 Quality check

Compute perplexity for each personality on its held-out transcripts under three setups: Hugging Face + PEFT on NF4 (the reference), your engine on NF4 (should match the reference), and your engine on Marlin INT4 (measures drift from serving on a format the adapters weren't trained on). Report the differences in a table.

### 5.5 Charts for the README

| Chart | Shows |
|---|---|
| Kernel µs vs. T, one line per method, per module shape | Kernel-level win over M1, M2, and M3 |
| **LoRA overhead per step vs. D (headline)** | The core result, independent of base-model speed |
| Aggregate tokens/s vs. D for S1–S4 | End-to-end comparison including vLLM |
| Nsight Systems timeline before and after CUDA Graphs | Where step time went |
| Achieved GB/s vs. the 360 GB/s ceiling | How close each kernel version gets to the hardware limit |

---

## 6. Profiling plan

```bash
# Timeline of decode steps: kernel launches, gaps, copies, CPU overhead
nsys profile -t cuda,nvtx,osrt -o profiles/decode_v1 python bench/e2e.py --steps 50

# Kernel deep-dive: all bgmv kernels, every metric section
ncu --set full -k regex:bgmv -c 20 -o profiles/bgmv_v2 python bench/micro.py --shape q_proj --T 8 --D 8
```

Wrap each layer and each LoRA call in NVTX ranges so the Nsight Systems timeline is readable.

| Question | Where to look |
|---|---|
| How close is the kernel to the bandwidth limit? | Nsight Compute: GPU Speed Of Light (memory throughput) |
| Are the loads coalesced? | Nsight Compute: Memory Workload Analysis (sectors per request vs. the minimum for your access width) |
| Why are warps stalling? | Nsight Compute: Warp State Statistics (long scoreboard means waiting on global memory) |
| Is the GPU full? | Nsight Compute: Occupancy and Launch Statistics (blocks launched vs. the 3060's 28 SMs) |
| Where does step time go? | Nsight Systems: launches per step, gaps between kernels, host-to-device copies |
| Did CUDA Graphs help? | Nsight Systems: step duration and CPU API time before vs. after |

---

## 7. Optimization ladder

Every step follows the same loop: hypothesis, change, measurement, hardware-level explanation, and a worklog entry. Keep a step even if it makes things slower; a well-explained loss is good interview material.

| Step | Change | Hypothesis | Evidence to collect |
|---|---|---|---|
| K0 | Naive shrink: one thread computes a whole dot product | Baseline; expected to be far from the ceiling | µs, GB/s |
| K1 | One warp per (token, rank row), warp-shuffle reduction | Coalesced loads, better parallelism | Sectors per request, GB/s |
| K2 | 16-byte vector loads, FP32 accumulation | Fewer memory instructions per byte | GB/s, instruction counts |
| K3 | Grid-shape tuning: rank tiles, and split-K when T ≤ 2 | At small T, too few blocks to fill 28 SMs | Achieved occupancy, µs |
| K4 | Several rank rows per warp (register tiling) | Reuse each loaded chunk of x across rows | L1 traffic, µs |
| K5 | Order tokens by slot through a permutation array; tokens sharing an adapter reuse its A | Identical and skewed workloads read each adapter once | DRAM bytes vs. D |
| K6 | Fused QKV and fused gate/up shrink with slice expand | Fewer launches, x read once: roughly 256 LoRA launches per step instead of 448 | Launches per step, overhead ms |
| K7 | Fused shrink + expand: each expand block recomputes v, relying on L2 | Half the launches, at the cost of redundant L2 reads; may lose, so measure | µs, L2 hit rate |
| K8 | CUDA Graph for the whole decode step | Removes CPU launch overhead | Nsight Systems gaps, step ms |
| K9 (stretch) | Your own NF4 skinny-GEMM with the LoRA shrink fused in | Read each activation once for base and adapter | Step ms vs. bitsandbytes and Marlin |
| K10 (stretch) | SGMV for prefill | Contiguous segments per adapter make tensor cores usable | Time to first token |

---

## 8. Phases, time, and exit criteria

| Phase | Scope | Estimated hours | Done when |
|---|---|---|---|
| P0. Environment and baseline | Section 1 setup, profiler permissions, Hugging Face + PEFT baseline, measure bitsandbytes' multi-token behavior | 6–10 | Baseline numbers and an Nsight Systems trace are in the worklog |
| P1. Kernels, standalone C++/CUDA | `gemv_fp16` warm-up, shrink, expand, CMake harness, compute-sanitizer | 12–20 | Correctness suite passes; first microbenchmark table exists |
| P2. PyTorch integration | `TORCH_LIBRARY` ops, adapter pool v1, patched model, layer-level tests | 8–12 | Engine output matches PEFT for one adapter |
| P3. Batched serving benchmark (first showable milestone) | Engine v1, baselines S1 and S2, overhead-vs-D chart, demo with 3–5 real personalities | 12–20 | Headline chart, demo video, and README are done |
| P4. Optimization ladder | K1–K8, each with Nsight evidence | 15–30 | Every step logged; best kernel compared against Punica |
| P5. Serving layer | Engine v2 (continuous batching, paging, CUDA Graphs), FastAPI demo, vLLM comparison | 15–25 | S4 comparison chart exists |
| P6. Fast base path | Marlin INT4 plus drift evaluation; stretch: your own fused NF4 kernel | 10–60 | Tokens/s vs. vLLM on a comparable base; drift table |

The minimum credible project is P0–P3, roughly 40–60 hours. The advanced version runs through P5 or P6.

---


## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Toolchain breaks on the rolling release | Medium | High | Pinned project environment, `IgnorePkg`, snapshots before updates |
| bitsandbytes' batched path is too slow for meaningful end-to-end numbers | High | Medium | Lead with the overhead metric; move the base to Marlin in P6 |
| Punica won't build against your CUDA | Medium | Low | It's optional; keep the PyTorch baselines and vLLM |
| vLLM with LoRA doesn't fit in 12 GB | Medium | Medium | Lower `--max-model-len`, `--gpu-memory-utilization`, `--max-loras`, and the batch size |
| CUDA Graph capture fails on the base-model ops | Medium | Medium | Capture only the LoRA path, or defer graphs until the Marlin base |
| Adapters use different ranks or target modules | Medium | Low | Retrain them uniformly, or pad to the maximum rank and measure the bandwidth cost |
| Numerical drift from FP16 or from changing quantization formats | Medium | Medium | FP32 accumulation; the drift table in 5.4 |
| Scope creep | High | High | Phase gates: ship P3 before starting anything else |

---

## 11. How to describe the project

Use only numbers from logged runs. Credit Punica and S-LoRA for the technique. Call the project "in progress" until P3's exit criteria are met.

Resume template, to be filled in only with measured values:

> **Persona-Serve**: multi-personality LLM serving on one 12 GB GPU (C++/CUDA, PyTorch). Custom batched-LoRA CUDA kernels serve N fine-tuned Mistral-7B personalities concurrently from one 4-bit base model; cut multi-adapter overhead from X ms to Y ms per decode step at D adapters; kernels reach Z% of the memory-bandwidth ceiling; benchmarked against Punica and vLLM.

---

## 12. References

| Source | Why it matters here |
|---|---|
| Punica, multi-tenant LoRA serving: github.com/punica-ai/punica and the MLSys 2024 paper (proceedings.mlsys.org) | Origin of the SGMV/BGMV kernel design; baseline M3 |
| S-LoRA (Sheng et al., 2023) | Adapter paging and serving thousands of adapters |
| TensorRT-LLM LoRA documentation: nvidia.github.io/TensorRT-LLM/features/lora.html | NVIDIA's own multi-LoRA support |
| NVIDIA blog, "Deploy Diverse AI Apps with Multi-LoRA Support on RTX AI PCs and Workstations" | Multi-LoRA on consumer RTX GPUs, the exact setting of this project |
| Marlin: github.com/IST-DASLab/marlin | Fast INT4 base path for P6 |
| bitsandbytes: github.com/bitsandbytes-foundation/bitsandbytes | NF4 base path and reference |
| Arseny Kapoulkine, "LLM inference speed of light": zeux.io/2024/03/15/llm-inference-sol | How to compute decode ceilings for Mistral-7B |
| yalm: github.com/andrewkchan/yalm | C++/CUDA LLM inference from scratch; engine-design reference |
| nano-vLLM: github.com/GeeeekExplorer/nano-vllm | Compact reference for a scheduler and KV cache |
| CUDA-L1 (arXiv 2507.14111) and Kernel Contracts (arXiv 2604.22032) | Documented benchmark loopholes that section 5.1 guards against |
| PyTorch "Custom C++ and CUDA Operators" tutorial; Nsight Compute and Nsight Systems documentation | Op registration, profiling metrics |
