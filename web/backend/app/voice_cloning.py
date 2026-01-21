"""
Voice Cloning Pipeline for HeartLib.

Integrates:
- HeartLib for music generation
- Demucs v4 (htdemucs_ft) for stem separation
- YingMusic-SVC for voice conversion
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import soundfile as sf

from .config import settings


class StemSeparator:
    """Demucs v4 stem separation for extracting vocals and instrumental."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Lazy load Demucs model."""
        if self._model is None:
            try:
                from demucs.pretrained import get_model
                self._model = get_model("htdemucs_ft")
                self._model.to(self.device)
                self._model.eval()
                print("Demucs htdemucs_ft model loaded successfully")
            except ImportError:
                raise ImportError(
                    "Demucs not installed. Install with: pip install demucs"
                )
    
    def separate(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Separate audio into vocals and instrumental.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Tuple of (vocals, instrumental, sample_rate)
        """
        self._load_model()
        
        from demucs.apply import apply_model
        
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        
        # Resample to model's expected sample rate if needed
        if sr != self._model.samplerate:
            resampler = torchaudio.transforms.Resample(sr, self._model.samplerate)
            wav = resampler(wav)
            sr = self._model.samplerate
        
        wav = wav.to(self.device)
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(self._model, wav[None], device=self.device)[0]
        
        # htdemucs_ft outputs: drums, bass, other, vocals
        vocals = sources[3]  # vocals
        instrumental = sources[0] + sources[1] + sources[2]  # drums + bass + other
        
        return vocals.cpu(), instrumental.cpu(), sr


class VoiceConverter:
    """YingMusic-SVC voice conversion wrapper."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._model_path = settings.YINGMUSIC_SVC_PATH
    
    def _load_model(self):
        """Lazy load YingMusic-SVC model."""
        if self._model is None:
            # Check if YingMusic-SVC is available
            if not os.path.exists(self._model_path):
                print(f"Warning: YingMusic-SVC not found at {self._model_path}")
                print("Voice conversion will be disabled. Clone from:")
                print("https://github.com/GiantAILab/YingMusic-SVC")
                return False
            
            try:
                sys.path.insert(0, self._model_path)
                # Import will depend on actual YingMusic-SVC API
                # This is a placeholder that will need adjustment
                from inference import YingMusicSVC
                self._model = YingMusicSVC.from_pretrained(device=self.device)
                print("YingMusic-SVC model loaded successfully")
                return True
            except ImportError as e:
                print(f"Warning: Could not load YingMusic-SVC: {e}")
                return False
        return True
    
    def convert(
        self,
        source_vocals: torch.Tensor,
        reference_audio: str,
        pitch_shift: int = 0,
        sample_rate: int = 48000
    ) -> torch.Tensor:
        """
        Convert source vocals to match reference voice timbre.
        
        Args:
            source_vocals: Tensor of source vocal audio
            reference_audio: Path to reference audio file
            pitch_shift: Semitones to shift pitch (-12 to +12)
            sample_rate: Sample rate of source audio
            
        Returns:
            Converted vocals tensor
        """
        if not self._load_model():
            print("Voice conversion unavailable, returning original vocals")
            return source_vocals
        
        try:
            # Save source vocals to temp file for model input
            temp_source = "/tmp/temp_source_vocals.wav"
            torchaudio.save(temp_source, source_vocals, sample_rate)
            
            # Convert using YingMusic-SVC
            # API will depend on actual implementation
            converted = self._model.convert(
                source=temp_source,
                reference=reference_audio,
                pitch_shift=pitch_shift,
                use_f0_adaptor=True
            )
            
            # Load converted audio
            converted_wav, _ = torchaudio.load(converted)
            return converted_wav
            
        except Exception as e:
            print(f"Voice conversion failed: {e}")
            return source_vocals


class VoiceCloningPipeline:
    """
    Complete voice cloning pipeline integrating HeartLib, Demucs, and YingMusic-SVC.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.stem_separator = StemSeparator(device=device)
        self.voice_converter = VoiceConverter(device=device)
        self._heartlib_pipeline = None
    
    def _load_heartlib(self):
        """Lazy load HeartLib pipeline."""
        if self._heartlib_pipeline is None:
            try:
                from heartlib.pipelines import MusicGenerationPipeline
                self._heartlib_pipeline = MusicGenerationPipeline(device=self.device)
                print("HeartLib pipeline loaded successfully")
            except ImportError as e:
                print(f"Warning: Could not load HeartLib: {e}")
                raise
    
    def generate(
        self,
        prompt: str,
        tags: str = "",
        lyrics: str = "",
        voice_reference: Optional[str] = None,
        duration_ms: int = 30000,
        pitch_shift: int = 0,
        flow_steps: int = 10,
        temperature: float = 1.0,
        cfg_scale: float = 1.25,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate music with optional voice cloning.
        
        Args:
            prompt: Description of the music to generate
            tags: Comma-separated style tags
            lyrics: Song lyrics with section markers
            voice_reference: Path to reference audio for voice cloning
            duration_ms: Target duration in milliseconds
            pitch_shift: Semitones to shift voice
            flow_steps: Flow matching steps (quality/speed tradeoff)
            temperature: Generation temperature
            cfg_scale: Classifier-free guidance scale
            output_path: Optional path to save output
            
        Returns:
            Generated audio tensor
        """
        self._load_heartlib()
        
        # Step 1: Generate with HeartLib
        print(f"Generating music: {prompt[:50]}...")
        
        inputs = {
            "prompt": prompt,
            "tags": tags,
            "lyrics": lyrics,
            "max_audio_length_ms": duration_ms,
        }
        
        audio = self._heartlib_pipeline(
            inputs,
            num_steps=flow_steps,
            temperature=temperature,
            cfg_scale=cfg_scale
        )
        
        # If no voice reference, return as-is
        if voice_reference is None:
            if output_path:
                self._save_audio(audio, output_path)
            return audio
        
        # Step 2: Separate stems (Demucs v4)
        print("Separating stems...")
        
        # Save generated audio to temp file for stem separation
        temp_generated = "/tmp/temp_generated.wav"
        self._save_audio(audio, temp_generated)
        
        vocals, instrumental, sr = self.stem_separator.separate(temp_generated)
        
        # Step 3: Convert vocals (YingMusic-SVC)
        print("Converting vocals...")
        converted_vocals = self.voice_converter.convert(
            source_vocals=vocals,
            reference_audio=voice_reference,
            pitch_shift=pitch_shift,
            sample_rate=sr
        )
        
        # Step 4: Mix back
        print("Mixing audio...")
        output = self._mix_audio(converted_vocals, instrumental, sr)
        
        if output_path:
            self._save_audio(output, output_path, sr)
        
        return output
    
    def convert_voice_only(
        self,
        source_path: str,
        voice_reference: str,
        pitch_shift: int = 0,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Convert vocals in existing audio to match reference voice.
        
        Args:
            source_path: Path to source audio file
            voice_reference: Path to reference audio for voice cloning
            pitch_shift: Semitones to shift voice
            output_path: Optional path to save output
            
        Returns:
            Converted audio tensor
        """
        # Step 1: Separate stems
        print("Separating stems...")
        vocals, instrumental, sr = self.stem_separator.separate(source_path)
        
        # Step 2: Convert vocals
        print("Converting vocals...")
        converted_vocals = self.voice_converter.convert(
            source_vocals=vocals,
            reference_audio=voice_reference,
            pitch_shift=pitch_shift,
            sample_rate=sr
        )
        
        # Step 3: Mix back
        print("Mixing audio...")
        output = self._mix_audio(converted_vocals, instrumental, sr)
        
        if output_path:
            self._save_audio(output, output_path, sr)
        
        return output
    
    def _mix_audio(
        self,
        vocals: torch.Tensor,
        instrumental: torch.Tensor,
        sample_rate: int,
        vocal_gain: float = 1.0
    ) -> torch.Tensor:
        """Mix vocals and instrumental with gain control."""
        # Ensure same length
        min_len = min(vocals.shape[-1], instrumental.shape[-1])
        vocals = vocals[..., :min_len]
        instrumental = instrumental[..., :min_len]
        
        # Apply gain and mix
        vocals = vocals * vocal_gain
        mixed = vocals + instrumental
        
        # Prevent clipping
        max_val = torch.max(torch.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val * 0.95
        
        return mixed
    
    def _save_audio(
        self,
        audio: torch.Tensor,
        path: str,
        sample_rate: int = 48000
    ):
        """Save audio tensor to file."""
        # Ensure 2D tensor (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Use soundfile for better format support
        sf.write(path, audio.numpy().T, sample_rate)
