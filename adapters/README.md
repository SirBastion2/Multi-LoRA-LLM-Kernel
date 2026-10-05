# Adapters

- **Real personalities:** place each trained QLoRA export in its own subdirectory (`adapter_config.json` + `adapter_model.safetensors`). Do not commit private weights unless you intend to publish them.
- **Random adapters:** for kernel and pool scaling tests (values do not affect bandwidth logic):

```bash
python adapters/make_random_adapters.py --out adapters/random --num-adapters 8 --rank 16
```

Load into the pool with `AdapterPool.load_adapter(name, path)` on the owner GPU machine.
