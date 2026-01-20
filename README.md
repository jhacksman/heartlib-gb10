# heartlib-gb10

HeartMuLa heartlib optimized for NVIDIA GB10 (DGX Spark, Asus Ascent GX10, Jetson Thor).

## Requirements

- NVIDIA GB10 GPU (aarch64)
- CUDA 12.8+
- Python 3.9+

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

This will automatically fetch PyTorch nightly builds with CUDA 12.8 support for aarch64.

## Verify GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected output: `NVIDIA GB10`
