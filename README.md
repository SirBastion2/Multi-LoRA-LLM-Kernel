# Persona-Serve
Multi-personality LLM serving on a single consumer GPU: one shared 4-bit Mistral-7B base and many LoRA “personality” adapters in one mixed decode batch. Custom CUDA **BGMV** (batched gather matrix–vector) kernels apply each request’s adapter inside shared kernel launches.

**Owner:** Sebastian Villalba  
**Target hardware:** NVIDIA GeForce RTX 3060 12 GB (Ampere 8.6), **CachyOS** (native Linux)  
**Status:** Scaffold / in progress — see [ARCHITECTURE.md](ARCHITECTURE.md) for phases and exit criteria.
**Benchmarks (RTX 3060):** [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — micro µs vs D, PEFT sequential/same-batch, and **true mixed-adapter** BGMV e2e.

## Credit (technique, not this repo’s measurements)

The batched-gather LoRA idea comes from **[Punica](https://github.com/punica-ai/punica)** (MLSys 2024) and related work such as **S-LoRA**. Production stacks (vLLM, TensorRT-LLM) ship similar kernels. This project re-implements and measures on consumer hardware; **engineering and numbers are the owner’s** once logged in [WORKLOG.md](WORKLOG.md).

## Repository map

See section 3.8 in [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start (development)

### 1. Environment (micromamba)

Detailed pins and CachyOS notes live in `env/`. Summary:

```bash
micromamba create -n persona -c conda-forge python=3.12
micromamba activate persona
# Install PyTorch for your CUDA build — see https://pytorch.org
pip install -e ".[dev]"
pytest tests/ -m "not cuda"   # CPU reference tests
```

Use a **separate** `vllm-baseline` env for vLLM (see `env/README.md`). Do not mix vLLM’s pinned PyTorch with the persona env.

### 2. Build CUDA extension

On the owner machine (GPU + matching `nvcc`):

```bash
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.6"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
pip install -e .
python -c "import torch; import persona; print('ok', torch.cuda.is_available())"
```

### 3. Random adapters (no Mistral weights required)

```bash
python adapters/make_random_adapters.py --out adapters/random --num-adapters 4 --rank 16
```

### 4. Standalone C++/CUDA harness (no Python)

```bash
cd standalone && mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER="$CXX"
cmake --build .
```

### 5. Demo (stub)

```bash
pip install fastapi uvicorn
uvicorn demo.server:app --host 127.0.0.1 --port 9847
```

The demo does **not** load Mistral weights in this scaffold; it shows the API shape for multi-pane chat.

## Benchmarks and claims

- **Do not trust headline numbers in this README until they appear in WORKLOG.md** with environment metadata and commit hash.
- GPU microbenchmarks and Nsight runs are intended for the owner’s **RTX 3060 on CachyOS**, not this cloud scaffold environment.
- Primary system metric (per architecture): **LoRA overhead per decode step** = step time with LoRA minus step with `LoRAContext.enabled = False`.

## Tests

```bash
pytest tests/
```

CUDA kernel tests skip automatically when no GPU is available (`pytest -m cuda` to run them).

## License

TBD — add before public release.
