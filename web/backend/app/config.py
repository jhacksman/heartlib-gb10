"""
Configuration settings for HeartLib Web Backend.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Device configuration
    DEVICE: str = "cuda"
    
    # Pipeline settings
    ENABLE_PIPELINE: bool = True
    
    # HeartLib model paths (optional, uses defaults if not set)
    HEARTMULA_MODEL_PATH: str = ""
    HEARTCODEC_MODEL_PATH: str = ""
    
    # YingMusic-SVC settings
    YINGMUSIC_SVC_PATH: str = "./YingMusic-SVC"
    
    # Audio settings
    SAMPLE_RATE: int = 48000
    MAX_DURATION_MS: int = 180000  # 3 minutes max
    
    # File storage
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MAX_UPLOAD_SIZE_MB: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
