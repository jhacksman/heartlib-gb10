# heartlib-gb10

HeartMuLa heartlib optimized for NVIDIA GB10 (DGX Spark, Asus Ascent GX10, Jetson Thor).

## Requirements

- NVIDIA GB10 GPU (aarch64)
- CUDA 12.8+
- Python 3.9+
- Node.js 20+

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

## Web App

### Backend Setup

```bash
cd web/backend
cp .env.example .env
uv pip install fastapi 'uvicorn[standard]' python-multipart pydantic pydantic-settings aiofiles python-dotenv demucs soundfile librosa
```

### Frontend Setup

```bash
cd web/frontend
cp .env.example .env
# Edit .env with your server IP:
# VITE_API_URL=http://<YOUR_IP>:8000
# VITE_BACKEND_URLS=http://<YOUR_IP>:8000,http://<YOUR_IP>:8001

npm install
npm run build
npm install -g serve
```

### Running

```bash
./scripts/start.sh   # Start all services
./scripts/stop.sh    # Stop all services
```

Services:
- Backend 1: http://0.0.0.0:8000
- Backend 2: http://0.0.0.0:8001
- Frontend: http://0.0.0.0:3000

Logs:
- `tail -f /tmp/heartlib-backend-1.log`
- `tail -f /tmp/heartlib-backend-2.log`
- `tail -f /tmp/heartlib-frontend.log`
