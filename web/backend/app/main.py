"""
HeartLib Web Backend - AI Music Studio

FastAPI backend for the AI Music Studio web interface.
MVP features: music generation, extend at any timestamp, crop, download.
"""

import os
import uuid
import asyncio
import hashlib
import secrets
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from .config import settings

# Storage for generated files
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory storage (for MVP - would use SQLite with volume for production)
jobs: dict = {}
users: dict = {}
sessions: dict = {}

# Security
security = HTTPBearer(auto_error=False)


class UserCreate(BaseModel):
    email: str
    password: str
    name: str = ""


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    credits: int


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    output_url: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: Optional[str] = None


class SongResponse(BaseModel):
    id: str
    name: str
    prompt: str
    tags: str
    duration_ms: int
    status: str
    output_url: Optional[str]
    created_at: str


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_token() -> str:
    return secrets.token_urlsafe(32)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    if not credentials:
        return None
    token = credentials.credentials
    if token not in sessions:
        return None
    user_id = sessions[token]
    return users.get(user_id)


async def require_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# Global pipeline instance
pipeline = None

# Lock to prevent concurrent generations (HeartLib model isn't thread-safe)
generation_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline
    
    if settings.ENABLE_PIPELINE:
        try:
            from .music_pipeline import MusicPipeline
            pipeline = MusicPipeline(device=settings.DEVICE)
            print("Music pipeline initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize pipeline: {e}")
            print("Running in demo mode without actual generation")
            pipeline = None
    else:
        print("Pipeline disabled, running in demo mode")
        pipeline = None
    
    yield
    
    if pipeline:
        del pipeline


app = FastAPI(
    title="HeartLib Music Studio API",
    description="AI Music Generation - Create, Extend, Edit, Download",
    version="0.1.0",
    lifespan=lifespan
)

# CORS configuration - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Health Endpoints ==============

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "HeartLib Music Studio API",
        "pipeline_ready": pipeline is not None
    }


@app.get("/api/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "device": settings.DEVICE,
    }


# ============== Auth Endpoints ==============

@app.post("/api/auth/signup", response_model=AuthResponse)
async def signup(data: UserCreate):
    """Create a new user account."""
    if any(u["email"] == data.email for u in users.values()):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    user = {
        "id": user_id,
        "email": data.email,
        "name": data.name or data.email.split("@")[0],
        "password_hash": hash_password(data.password),
        "credits": 100,  # Starting credits
        "created_at": datetime.utcnow().isoformat()
    }
    users[user_id] = user
    
    token = create_token()
    sessions[token] = user_id
    
    return AuthResponse(
        token=token,
        user=UserResponse(
            id=user_id,
            email=user["email"],
            name=user["name"],
            credits=user["credits"]
        )
    )


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(data: UserLogin):
    """Login with email and password."""
    user = None
    for u in users.values():
        if u["email"] == data.email and u["password_hash"] == hash_password(data.password):
            user = u
            break
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_token()
    sessions[token] = user["id"]
    
    return AuthResponse(
        token=token,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            credits=user["credits"]
        )
    )


@app.post("/api/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout and invalidate token."""
    if credentials and credentials.credentials in sessions:
        del sessions[credentials.credentials]
    return {"status": "logged out"}


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(user: dict = Depends(require_user)):
    """Get current user info."""
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        credits=user["credits"]
    )


# ============== Music Generation Endpoints ==============

async def process_generation(
    job_id: str,
    user_id: str,
    prompt: str,
    tags: str,
    lyrics: str,
    duration_ms: int,
    flow_steps: int,
    temperature: float,
    cfg_scale: float,
    topk: int,
):
    """Background task for music generation."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1
        jobs[job_id]["message"] = "Starting generation..."
        
        if pipeline is None:
            # Demo mode - simulate generation
            await asyncio.sleep(3)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["message"] = "Demo mode - no actual audio generated"
            jobs[job_id]["output_url"] = f"/api/songs/{job_id}/download"
            return
        
        jobs[job_id]["progress"] = 0.2
        jobs[job_id]["message"] = "Waiting for GPU..."
        
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        # Acquire lock to prevent concurrent generations (HeartLib model isn't thread-safe)
        async with generation_lock:
            jobs[job_id]["message"] = "Generating music..."
            await asyncio.to_thread(
                pipeline.generate,
                prompt=prompt,
                tags=tags,
                lyrics=lyrics,
                duration_ms=duration_ms,
                flow_steps=flow_steps,
                temperature=temperature,
                cfg_scale=cfg_scale,
                topk=topk,
                output_path=str(output_path)
            )
        
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Generation complete!"
        jobs[job_id]["output_url"] = f"/api/songs/{job_id}/download"
        
        # Deduct credits
        if user_id in users:
            users[user_id]["credits"] = max(0, users[user_id]["credits"] - 10)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Generation failed: {str(e)}"


@app.post("/api/songs/generate", response_model=GenerationResponse)
async def generate_music(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    tags: str = Form(""),
    lyrics: str = Form(""),
    duration_ms: int = Form(30000),
    flow_steps: int = Form(10),
    temperature: float = Form(1.0),
    cfg_scale: float = Form(1.5),
    topk: int = Form(50),
    user: dict = Depends(require_user)
):
    """
    Generate a new song.
    
    - **prompt**: Song name (used for naming only, not for conditioning)
    - **tags**: Comma-separated style tags (e.g., "polka,happy,accordion")
    - **lyrics**: Song lyrics with section markers ([Verse], [Chorus], etc.)
    - **duration_ms**: Target duration in milliseconds (default: 30000 = 30s)
    - **flow_steps**: Quality setting (5-20, higher = better quality but slower)
    - **temperature**: Creativity (0.5-2.0, higher = more creative)
    - **cfg_scale**: Style adherence (1.0-3.0, higher = stronger tag/lyric conditioning)
    - **topk**: Top-k sampling (default: 50, HeartLib's recommended value)
    
    HeartLib Recommended Settings:
        - cfg_scale: 1.5 (controls how strongly tags/lyrics affect output)
        - temperature: 1.0
        - topk: 50
        - Tags should be short keywords like: piano,happy,wedding,synthesizer,romantic
    """
    if user["credits"] < 10:
        raise HTTPException(status_code=402, detail="Insufficient credits")
    
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "id": job_id,
        "user_id": user["id"],
        "name": prompt[:50] + ("..." if len(prompt) > 50 else ""),
        "prompt": prompt,
        "tags": tags,
        "lyrics": lyrics,
        "duration_ms": duration_ms,
        "status": "pending",
        "progress": 0.0,
        "message": "Job queued",
        "output_url": None,
        "created_at": datetime.utcnow().isoformat()
    }
    
    background_tasks.add_task(
        process_generation,
        job_id=job_id,
        user_id=user["id"],
        prompt=prompt,
        tags=tags,
        lyrics=lyrics,
        duration_ms=duration_ms,
        flow_steps=flow_steps,
        temperature=temperature,
        cfg_scale=cfg_scale,
        topk=topk,
    )
    
    return GenerationResponse(
        job_id=job_id,
        status="pending",
        message="Generation started"
    )


async def process_extend(
    job_id: str,
    user_id: str,
    source_job_id: str,
    extend_from_ms: int,
    extend_duration_ms: int,
    prompt: str,
    direction: str,
    flow_steps: int,
    temperature: float,
    cfg_scale: float,
    topk: int,
):
    """Background task for extending a song."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1
        jobs[job_id]["message"] = "Preparing to extend..."
        
        source_path = OUTPUT_DIR / f"{source_job_id}.wav"
        if not source_path.exists():
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = "Source audio not found"
            return
        
        if pipeline is None:
            # Demo mode
            await asyncio.sleep(3)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["message"] = "Demo mode - extension simulated"
            jobs[job_id]["output_url"] = f"/api/songs/{job_id}/download"
            return
        
        jobs[job_id]["progress"] = 0.3
        jobs[job_id]["message"] = "Waiting for GPU..."
        
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        # Acquire lock to prevent concurrent generations (HeartLib model isn't thread-safe)
        async with generation_lock:
            jobs[job_id]["message"] = f"Extending {direction} from {extend_from_ms}ms..."
            await asyncio.to_thread(
                pipeline.extend,
                source_path=str(source_path),
                extend_from_ms=extend_from_ms,
                extend_duration_ms=extend_duration_ms,
                prompt=prompt,
                direction=direction,
                flow_steps=flow_steps,
                temperature=temperature,
                cfg_scale=cfg_scale,
                topk=topk,
                output_path=str(output_path)
            )
        
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Extension complete!"
        jobs[job_id]["output_url"] = f"/api/songs/{job_id}/download"
        
        if user_id in users:
            users[user_id]["credits"] = max(0, users[user_id]["credits"] - 5)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Extension failed: {str(e)}"


@app.post("/api/songs/{song_id}/extend", response_model=GenerationResponse)
async def extend_song(
    song_id: str,
    background_tasks: BackgroundTasks,
    extend_from_ms: int = Form(...),
    extend_duration_ms: int = Form(30000),
    prompt: str = Form(""),
    direction: str = Form("after"),
    flow_steps: int = Form(10),
    temperature: float = Form(1.0),
    cfg_scale: float = Form(1.5),
    topk: int = Form(50),
    user: dict = Depends(require_user)
):
    """
    Extend a song from a specific timestamp.
    
    - **song_id**: ID of the song to extend
    - **extend_from_ms**: Timestamp in milliseconds to extend from
    - **extend_duration_ms**: Duration of the extension (default: 30000 = 30s)
    - **prompt**: Optional prompt for the extension (uses original if empty)
    - **direction**: "before" or "after" the timestamp
    - **flow_steps**: Quality setting (5-20)
    - **temperature**: Creativity (0.5-2.0)
    - **cfg_scale**: Style adherence (1.0-3.0, higher = stronger conditioning)
    - **topk**: Top-k sampling (default: 50, HeartLib's recommended value)
    """
    if song_id not in jobs:
        raise HTTPException(status_code=404, detail="Song not found")
    
    if jobs[song_id]["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not your song")
    
    if user["credits"] < 5:
        raise HTTPException(status_code=402, detail="Insufficient credits")
    
    source_job = jobs[song_id]
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "id": job_id,
        "user_id": user["id"],
        "name": f"Extended: {source_job['name'][:30]}",
        "prompt": prompt or source_job["prompt"],
        "tags": source_job["tags"],
        "lyrics": source_job.get("lyrics", ""),
        "duration_ms": source_job["duration_ms"] + extend_duration_ms,
        "status": "pending",
        "progress": 0.0,
        "message": "Extension queued",
        "output_url": None,
        "parent_id": song_id,
        "created_at": datetime.utcnow().isoformat()
    }
    
    background_tasks.add_task(
        process_extend,
        job_id=job_id,
        user_id=user["id"],
        source_job_id=song_id,
        extend_from_ms=extend_from_ms,
        extend_duration_ms=extend_duration_ms,
        prompt=prompt or source_job["prompt"],
        direction=direction,
        flow_steps=flow_steps,
        temperature=temperature,
        cfg_scale=cfg_scale,
        topk=topk,
    )
    
    return GenerationResponse(
        job_id=job_id,
        status="pending",
        message="Extension started"
    )


@app.post("/api/songs/{song_id}/crop", response_model=GenerationResponse)
async def crop_song(
    song_id: str,
    start_ms: int = Form(...),
    end_ms: int = Form(...),
    user: dict = Depends(require_user)
):
    """
    Crop a song to a specific time range.
    
    - **song_id**: ID of the song to crop
    - **start_ms**: Start timestamp in milliseconds
    - **end_ms**: End timestamp in milliseconds
    """
    if song_id not in jobs:
        raise HTTPException(status_code=404, detail="Song not found")
    
    if jobs[song_id]["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not your song")
    
    source_path = OUTPUT_DIR / f"{song_id}.wav"
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    source_job = jobs[song_id]
    job_id = str(uuid.uuid4())
    
    try:
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(str(source_path))
        
        # Convert ms to samples
        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)
        
        # Crop
        cropped = audio[start_sample:end_sample]
        
        # Save
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        sf.write(str(output_path), cropped, sr)
        
        jobs[job_id] = {
            "id": job_id,
            "user_id": user["id"],
            "name": f"Cropped: {source_job['name'][:30]}",
            "prompt": source_job["prompt"],
            "tags": source_job["tags"],
            "lyrics": source_job.get("lyrics", ""),
            "duration_ms": end_ms - start_ms,
            "status": "completed",
            "progress": 1.0,
            "message": "Crop complete",
            "output_url": f"/api/songs/{job_id}/download",
            "parent_id": song_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return GenerationResponse(
            job_id=job_id,
            status="completed",
            message="Crop complete"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop failed: {str(e)}")


# ============== Song Management Endpoints ==============

@app.get("/api/songs", response_model=List[SongResponse])
async def list_songs(user: dict = Depends(require_user)):
    """List all songs for the current user."""
    user_songs = [
        SongResponse(
            id=job["id"],
            name=job["name"],
            prompt=job["prompt"],
            tags=job["tags"],
            duration_ms=job["duration_ms"],
            status=job["status"],
            output_url=job.get("output_url"),
            created_at=job["created_at"]
        )
        for job in jobs.values()
        if job.get("user_id") == user["id"]
    ]
    return sorted(user_songs, key=lambda x: x.created_at, reverse=True)


@app.get("/api/songs/{song_id}", response_model=JobStatus)
async def get_song(song_id: str, user: dict = Depends(require_user)):
    """Get song details and status."""
    if song_id not in jobs:
        raise HTTPException(status_code=404, detail="Song not found")
    
    job = jobs[song_id]
    if job.get("user_id") != user["id"]:
        raise HTTPException(status_code=403, detail="Not your song")
    
    return JobStatus(
        job_id=job["id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        output_url=job.get("output_url"),
        duration_ms=job.get("duration_ms"),
        created_at=job.get("created_at")
    )


@app.get("/api/songs/{song_id}/download")
async def download_song(song_id: str, user: dict = Depends(require_user)):
    """Download the audio file for a song."""
    if song_id not in jobs:
        raise HTTPException(status_code=404, detail="Song not found")
    
    job = jobs[song_id]
    if job.get("user_id") != user["id"]:
        raise HTTPException(status_code=403, detail="Not your song")
    
    output_path = OUTPUT_DIR / f"{song_id}.wav"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(output_path),
        media_type="audio/wav",
        filename=f"{job['name'][:30]}.wav"
    )


@app.delete("/api/songs/{song_id}")
async def delete_song(song_id: str, user: dict = Depends(require_user)):
    """Delete a song."""
    if song_id not in jobs:
        raise HTTPException(status_code=404, detail="Song not found")
    
    job = jobs[song_id]
    if job.get("user_id") != user["id"]:
        raise HTTPException(status_code=403, detail="Not your song")
    
    # Delete audio file
    output_path = OUTPUT_DIR / f"{song_id}.wav"
    if output_path.exists():
        output_path.unlink()
    
    del jobs[song_id]
    
    return {"status": "deleted", "song_id": song_id}


# ============== Legacy Endpoints (for backwards compatibility) ==============

@app.get("/api/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status (legacy endpoint)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        output_url=job.get("output_url"),
        duration_ms=job.get("duration_ms"),
        created_at=job.get("created_at")
    )
