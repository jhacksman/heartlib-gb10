# HeartLib Web Backend

FastAPI backend for the AI Music Studio web interface. Integrates HeartLib for music generation, Demucs v4 for stem separation, and YingMusic-SVC for voice conversion.

## Setup

### Prerequisites

1. Install HeartLib from the parent directory:
```bash
cd ../..
pip install -e .
```

2. Install backend dependencies:
```bash
cd web/backend
pip install -e .
```

3. Clone YingMusic-SVC for voice conversion:
```bash
git clone https://github.com/GiantAILab/YingMusic-SVC.git
cd YingMusic-SVC
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and adjust as needed:
```bash
cp .env.example .env
```

### Running

Start the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /api/health` - Detailed health check

### Music Generation
- `POST /api/generate` - Generate music with optional voice cloning
  - `prompt` (required): Description of the music
  - `tags`: Comma-separated style tags
  - `lyrics`: Song lyrics with section markers
  - `duration_ms`: Target duration (default: 30000)
  - `pitch_shift`: Voice pitch shift in semitones (-12 to +12)
  - `flow_steps`: Quality/speed tradeoff (5-20)
  - `temperature`: Generation temperature (0.5-2.0)
  - `cfg_scale`: Guidance scale (1.0-2.0)
  - `voice_reference`: Audio file for voice cloning (optional)

### Voice Conversion
- `POST /api/convert-voice` - Convert vocals in existing audio
  - `source_audio` (required): Audio file to convert
  - `voice_reference` (required): Reference audio for target voice
  - `pitch_shift`: Voice pitch shift in semitones

### Job Management
- `GET /api/job/{job_id}` - Get job status
- `GET /api/download/{job_id}` - Download generated audio
- `DELETE /api/job/{job_id}` - Delete job and files

## Architecture

```
User Request
     │
     ▼
┌─────────────────┐
│   FastAPI       │
│   Backend       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   HeartLib      │  ← Music Generation
│   Pipeline      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Demucs v4     │  ← Stem Separation
│   (htdemucs_ft) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YingMusic-SVC  │  ← Voice Conversion
│                 │
└────────┬────────┘
         │
         ▼
    Mixed Output
```

## VRAM Usage

| Component | VRAM (FP16) |
|-----------|-------------|
| HeartMuLa 3B | ~6 GB |
| HeartCodec | ~2 GB |
| Demucs v4 | ~2-4 GB |
| YingMusic-SVC | ~4-6 GB |
| **Total** | **~14-18 GB** |

GB10 has 128GB unified memory, providing ample headroom.
