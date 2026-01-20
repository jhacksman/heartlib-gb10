# heartlib-gb10

HeartMuLa heartlib optimized for NVIDIA GB10 (DGX Spark, Asus Ascent GX10, Jetson Thor).

## Installation

### With CUDA support (recommended for GB10)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[cuda]"
```

### CPU only

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[cpu]"
```

## Verify GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
