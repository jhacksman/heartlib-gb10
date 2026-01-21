# heartlib-gb10

HeartMuLa heartlib optimized for NVIDIA GB10 (DGX Spark, Asus Ascent GX10, Jetson Thor).

## Requirements

- NVIDIA GB10 GPU (aarch64)
- CUDA 12.8+
- Python 3.9+

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install -e .
```

## Verify GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected output: `NVIDIA GB10`
