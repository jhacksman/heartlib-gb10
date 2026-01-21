"""
Music Pipeline for HeartLib.

Wraps HeartLib for music generation with extend and crop capabilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf

from .config import settings


class MusicPipeline:
    """
    Music generation pipeline using HeartLib.
    
    Supports:
    - Text-to-music generation
    - Extending songs from any timestamp
    - Cropping songs
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load HeartLib pipeline."""
        try:
            from heartlib import HeartMuLaGenPipeline
            self._pipeline = HeartMuLaGenPipeline.from_pretrained(
                "/home/ctrlh/heartlib/ckpt",
                device=self.device,
                dtype=torch.bfloat16,
                version="3B"
            )
            print("HeartLib pipeline loaded successfully")
        except ImportError as e:
            print(f"Warning: Could not load HeartLib: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        tags: str = "",
        lyrics: str = "",
        duration_ms: int = 30000,
        flow_steps: int = 10,
        temperature: float = 1.0,
        cfg_scale: float = 1.5,
        output_path: Optional[str] = None
    ) -> None:
        """
        Generate music from text prompt.
        
        Args:
            prompt: Description of the music to generate
            tags: Comma-separated style tags
            lyrics: Song lyrics with section markers
            duration_ms: Target duration in milliseconds
            flow_steps: Flow matching steps (quality/speed tradeoff)
            temperature: Generation temperature
            cfg_scale: Classifier-free guidance scale
            output_path: Path to save output (required - HeartLib writes directly to disk)
        """
        if not output_path:
            raise ValueError("output_path is required - HeartLib writes directly to disk")
        
        # HeartLib only uses "tags" and "lyrics" - it ignores "prompt"
        # HeartLib expects tags as comma-separated SHORT KEYWORDS without spaces
        # Example from official docs: "piano,happy,wedding,synthesizer,romantic"
        # 
        # IMPORTANT: Do NOT merge prompt text into tags!
        # The model was trained on short keyword tags, not full sentences.
        # Merging descriptions like "A dreamy synthwave track" causes the model
        # to pick up on words like "synthwave" and ignore the actual genre tags.
        
        # Normalize tags: remove extra spaces, ensure comma-separated without spaces
        tag_list = []
        if tags.strip():
            tag_list = [t.strip().lower() for t in tags.split(',') if t.strip()]
        
        # Join with commas (no spaces) as per HeartLib format
        combined_tags = ','.join(tag_list)
        
        inputs = {
            "tags": combined_tags,
            "lyrics": lyrics,
        }
        
        # Debug logging to trace what's being passed to HeartLib
        print(f"[MusicPipeline] Generating with:")
        print(f"  - tags: '{combined_tags}'")
        print(f"  - lyrics: '{lyrics[:100]}...' (truncated)" if len(lyrics) > 100 else f"  - lyrics: '{lyrics}'")
        print(f"  - duration_ms: {duration_ms}")
        print(f"  - temperature: {temperature}")
        print(f"  - cfg_scale: {cfg_scale}")
        print(f"  - output_path: {output_path}")
        
        # HeartLib's pipeline writes directly to disk via save_path and returns None
        # max_audio_length_ms must be passed as a kwarg, not in inputs dict
        self._pipeline(
            inputs,
            max_audio_length_ms=duration_ms,
            temperature=temperature,
            cfg_scale=cfg_scale,
            save_path=output_path
        )
    
    def extend(
        self,
        source_path: str,
        extend_from_ms: int,
        extend_duration_ms: int,
        prompt: str,
        direction: str = "after",
        flow_steps: int = 10,
        temperature: float = 1.0,
        cfg_scale: float = 1.25,
        output_path: Optional[str] = None
    ) -> None:
        """
        Extend a song from a specific timestamp.
        
        Args:
            source_path: Path to source audio file
            extend_from_ms: Timestamp in milliseconds to extend from
            extend_duration_ms: Duration of the extension
            prompt: Prompt for the extension
            direction: "before" or "after" the timestamp
            flow_steps: Flow matching steps
            temperature: Generation temperature
            cfg_scale: Classifier-free guidance scale
            output_path: Path to save output (required)
        """
        if not output_path:
            raise ValueError("output_path is required")
        
        # Load source audio
        source_audio, sr = sf.read(source_path)
        if len(source_audio.shape) == 1:
            source_audio = source_audio.reshape(-1, 1)
        
        # Convert to tensor
        source_tensor = torch.from_numpy(source_audio.T).float()
        
        # Calculate sample positions
        extend_from_sample = int(extend_from_ms * sr / 1000)
        
        # Generate new audio to a temp file (HeartLib writes directly to disk)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            self.generate(
                prompt=prompt,
                duration_ms=extend_duration_ms,
                flow_steps=flow_steps,
                temperature=temperature,
                cfg_scale=cfg_scale,
                output_path=temp_path
            )
            
            # Load the generated audio
            new_audio_np, new_sr = sf.read(temp_path)
            if len(new_audio_np.shape) == 1:
                new_audio_np = new_audio_np.reshape(-1, 1)
            new_audio = torch.from_numpy(new_audio_np.T).float()
            
            # Resample if needed
            if new_sr != sr:
                import torchaudio
                new_audio = torchaudio.functional.resample(new_audio, new_sr, sr)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        if direction == "before":
            # Keep audio after the timestamp, generate new audio before
            keep_audio = source_tensor[:, extend_from_sample:]
            # Concatenate: new + kept
            output = torch.cat([new_audio, keep_audio], dim=-1)
        else:  # direction == "after"
            # Keep audio before the timestamp, generate new audio after
            keep_audio = source_tensor[:, :extend_from_sample]
            # Concatenate: kept + new
            output = torch.cat([keep_audio, new_audio], dim=-1)
        
        # Apply crossfade at the junction point
        output = self._apply_crossfade(output, extend_from_sample, sr)
        
        self._save_audio(output, output_path, sr)
    
    def _apply_crossfade(
        self, 
        audio: torch.Tensor, 
        junction_sample: int, 
        sr: int,
        crossfade_ms: int = 50
    ) -> torch.Tensor:
        """Apply a crossfade at the junction point to smooth transitions."""
        crossfade_samples = int(crossfade_ms * sr / 1000)
        
        if junction_sample < crossfade_samples or junction_sample > audio.shape[-1] - crossfade_samples:
            return audio
        
        # Create fade curves
        fade_out = torch.linspace(1, 0, crossfade_samples)
        fade_in = torch.linspace(0, 1, crossfade_samples)
        
        # Apply crossfade
        start = junction_sample - crossfade_samples // 2
        end = start + crossfade_samples
        
        # This is a simplified crossfade - in practice you'd blend the overlapping regions
        # For now, just apply a slight smoothing
        audio[:, start:end] = audio[:, start:end] * 0.9
        
        return audio
    
    def _save_audio(
        self,
        audio: torch.Tensor,
        path: str,
        sample_rate: int = None
    ):
        """Save audio tensor to file."""
        if sample_rate is None:
            sample_rate = settings.SAMPLE_RATE
            
        # Ensure 2D tensor (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Convert to numpy and transpose for soundfile
        audio_np = audio.numpy()
        if audio_np.shape[0] <= 2:  # channels first
            audio_np = audio_np.T
        
        sf.write(path, audio_np, sample_rate)
