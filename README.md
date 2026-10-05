# Multi-LoRA LLM Kernel

I built custom CUDA **BGMV** (batched gather matrix–vector) kernels so multiple LoRA adapters can run in **one mixed decode batch** on a single shared NF4 [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base — without loading a full copy of the model per adapter.

**Hardware:** NVIDIA GeForce RTX 3060 (12 GB), CachyOS Linux  
**Owner:** Sebastian Villalba

## What I did

- Wrote BGMV CUDA kernels (shrink / expand) and Python bindings for applying the right LoRA adapter to each row in a batch inside shared kernel launches.
- Built an adapter pool, Multi-LoRA linear layers, and NF4 model patching so several adapters live in one GPU pool on the shared base.
- Measured end-to-end generate throughput against Hugging Face PEFT sequential serving, and kernel micro-timings that stay flat as adapter count grows.

Adapters used for these numbers were **synthetic** (correct shapes, random weights). This is an engineering / throughput result, not a persona-quality eval.

## Benchmarks (RTX 3060)

Higher tokens/s is better (more tokens generated per second).

| Scenario | Throughput |
|----------|------------|
| PEFT: four adapters, one after another | ~21 tokens/s |
| **My path: four different adapters in one batch** | **~33 tokens/s** (~1.6× sequential PEFT) |
| **My path: eight different adapters in one batch** | **~56 tokens/s** (fits in 12 GB) |
| Same adapter, batched (upper bound) | ~39–67 tokens/s |

Kernel-only adapter op stays ~0.19 ms from 1→8 distinct adapters (launch-bound at this shape).

Full tables, methodology, and profile JSON: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## License

MIT — see [LICENSE](LICENSE).
