# Environment notes (CachyOS / RTX 3060)

Record exact versions in **WORKLOG.md** for every benchmark.

## `persona` env (kernels + engine)

See ARCHITECTURE.md §1.2. Use Python **3.12** (or another version with PyTorch + vLLM wheels).

```bash
micromamba create -n persona -c conda-forge python=3.12
micromamba activate persona
pip install torch --index-url https://download.pytorch.org/whl/cu124  # match your driver/CUDA
pip install -e ".[dev]"
```

Pin host compiler for `nvcc`:

```bash
micromamba install -c nvidia "cuda-toolkit=<match torch.version.cuda>"
micromamba install -c conda-forge "gxx_linux-64=13"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.6"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
```

## `vllm-baseline` env (separate PyTorch pin)

```bash
micromamba create -n vllm-baseline -c conda-forge python=3.12
micromamba activate vllm-baseline
pip install vllm
```

Do **not** install vLLM into `persona`.

## CachyOS rolling-release mitigations

- `IgnorePkg` for nvidia packages during a benchmark campaign (see ARCHITECTURE.md §1.1).
- Profiler: `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in modprobe.d.
- Benchmarks: TTY or minimal session; note idle VRAM in `nvidia-smi`.
