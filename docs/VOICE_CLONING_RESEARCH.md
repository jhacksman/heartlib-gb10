# Voice Cloning for HeartLib: Research & Implementation Plan

## Executive Summary

This document outlines the research findings and implementation paths for adding voice cloning capabilities to HeartLib. The goal is to enable users to upload a reference audio sample and have HeartLib generate music with vocals that match the reference voice's timbre.

**Key Finding:** Zero-shot Singing Voice Conversion (SVC) is a mature technology. The 2025 Singing Voice Conversion Challenge demonstrated that top systems achieve "comparable singer identity scores to ground truth samples." Multiple open-source implementations exist that can run on 4-8GB VRAM, making integration with HeartLib on GB10 (128GB unified memory) highly feasible.

---

## SELECTED TOOLS (Final Decision)

After comprehensive research and evaluation for GB10 hardware (ARM CPU + Blackwell GPU, 128GB unified memory):

| Component | Selected Tool | Rationale |
|-----------|---------------|-----------|
| **Voice Conversion** | **YingMusic-SVC** | Latest research (Dec 2025), Flow-GRPO architecture compatible with HeartCodec, robust to harmony interference, F0-aware timbre adaptor, actively maintained |
| **Stem Separation** | **Demucs v4 (htdemucs_ft)** | State-of-the-art quality (SDR 9.20 dB), MIT license, hybrid transformer architecture, well-tested |

### Tool Comparison Summary

**SVC Models Evaluated:**

| Tool | Quality | VRAM | Status | Why Selected/Rejected |
|------|---------|------|--------|----------------------|
| **YingMusic-SVC** | Best | ~4-6GB | Active (Dec 2025) | **SELECTED** - Latest research, handles real-world audio robustly |
| HQ-SVC | Excellent | ~4GB | Active (Nov 2025) | Backup option - efficient, also does super-resolution |
| seed-vc | Good | 4-8GB | Archived (Nov 2025) | Proven but unmaintained |
| RVC | Very Good | ~4-6GB | Active | Requires fine-tuning (not zero-shot) |
| LHQ-SVC | Good | ~2GB | Research | CPU-optimized, lower quality than GPU models |

**Stem Separation Models Evaluated:**

| Tool | Quality | VRAM | Status | Why Selected/Rejected |
|------|---------|------|--------|----------------------|
| **Demucs v4 (htdemucs_ft)** | Best (SDR 9.20 dB) | ~2-4GB | Stable | **SELECTED** - State-of-the-art, MIT license |
| Spleeter | Good | ~1-2GB | Maintained | Faster but lower quality |

### VRAM Budget (Final)

| Component | VRAM (FP16) |
|-----------|-------------|
| HeartMuLa 3B | ~6 GB |
| HeartCodec | ~2 GB |
| Demucs v4 | ~2-4 GB |
| YingMusic-SVC | ~4-6 GB |
| **Total** | **~14-18 GB** |
| **Available (GB10)** | **128 GB** |

**Headroom:** 110+ GB available for batch processing, longer audio, or parallel generations.

---

## 1. Current State of HeartLib

### 1.1 Existing Architecture

HeartLib consists of two main components:

| Component | Parameters | Function |
|-----------|------------|----------|
| HeartMuLa | 3B (backbone) + 300M (decoder) | Autoregressive token generation |
| HeartCodec | ~500M | Flow matching detokenization |

### 1.2 Reference Audio Infrastructure (Partially Implemented)

HeartLib's codebase shows **planned but unimplemented** support for reference audio:

```python
# From src/heartlib/pipelines/music_generation.py (lines 86-89)
ref_audio = inputs.get("ref_audio", None)
if ref_audio is not None:
    raise NotImplementedError("ref_audio is not supported yet.")
muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype)
```

The `muq_embed` (Music Understanding embedding) and `muq_mulan` parameters exist in the architecture, suggesting the original design anticipated style/timbre conditioning. This is a strong foundation for Path B implementation.

---

## 2. Voice Cloning Technology Landscape

### 2.1 Zero-Shot Singing Voice Conversion (SVC)

Zero-shot SVC transforms vocals from a source singer to a target singer's timbre using only a short reference audio sample (10-30 seconds), without requiring model fine-tuning.

**State of the Art (2025-2026):**

| System | Architecture | Key Innovation | Open Source |
|--------|--------------|----------------|-------------|
| YingMusic-SVC | Flow-GRPO + RVC | Robust to harmony interference | Yes (GitHub) |
| seed-vc | DiT + Flow Matching | Real-time support, 4-8GB VRAM | Yes (Archived) |
| HQ-SVC | Decoupled Codec + Diffusion | High-quality, efficient | Yes (GitHub) |
| Vevo2 | AR + Flow Matching | Unified speech/singing | Planned release |
| Everyone-Can-Sing | Diffusion | Voice clone from speech sample | Research |

### 2.2 Key Research Papers

**arXiv:2508.16332 - Vevo2: Unified Framework for Speech and Singing Voice Generation**
- Two tokenizers: prosody tokenizer + content-style tokenizer
- AR content-style modeling + flow-matching acoustic modeling
- Enables timbre disentanglement (critical for voice cloning)
- Joint speech-singing training bridges domains

**arXiv:2501.13870 - Everyone-Can-Sing: Zero-Shot SVS/SVC with Speech Reference**
- Can clone singing voice from just a SPEECH sample
- Uses diffusion-based generator
- Trained on mixed speech + singing datasets
- Addresses scarcity of singing training data

**arXiv:2512.04793 - YingMusic-SVC: Robust Zero-Shot SVC**
- Flow-GRPO reinforcement learning
- F0-aware timbre adaptor for dynamic expression
- Handles real-world challenges: harmony interference, F0 errors
- Production-ready quality

**arXiv:2509.15629 - Singing Voice Conversion Challenge 2025**
- Top systems achieved comparable identity scores to ground truth
- Singing style modeling (vibrato, glissando, breathy) remains challenging
- 26 systems evaluated in controlled environment

---

## 3. Implementation Paths

### 3.1 Path A: Post-Processing SVC Pipeline

**Timeline:** 1-2 weeks
**Complexity:** Low
**Quality:** Good (proven technology)

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Voice Cloning Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Input                                                      │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │ Text/Lyrics  │    │ Voice Ref    │                           │
│  │   + Tags     │    │  (10-30s)    │                           │
│  └──────┬───────┘    └──────┬───────┘                           │
│         │                   │                                    │
│         ▼                   │                                    │
│  ┌──────────────┐           │                                    │
│  │  HeartLib    │           │                                    │
│  │  Generation  │           │                                    │
│  └──────┬───────┘           │                                    │
│         │                   │                                    │
│         ▼                   │                                    │
│  ┌──────────────┐           │                                    │
│  │    Demucs    │           │                                    │
│  │ (Stem Split) │           │                                    │
│  └──────┬───────┘           │                                    │
│         │                   │                                    │
│    ┌────┴────┐              │                                    │
│    ▼         ▼              │                                    │
│ Vocals   Instrumental       │                                    │
│    │                        │                                    │
│    ▼                        ▼                                    │
│  ┌──────────────────────────────┐                               │
│  │      SVC Model               │                               │
│  │  (seed-vc / YingMusic-SVC)   │                               │
│  └──────────────┬───────────────┘                               │
│                 │                                                │
│                 ▼                                                │
│          Converted Vocals                                        │
│                 │                                                │
│                 ▼                                                │
│  ┌──────────────────────────────┐                               │
│  │         Audio Mixer          │                               │
│  │   (Vocals + Instrumental)    │                               │
│  └──────────────┬───────────────┘                               │
│                 │                                                │
│                 ▼                                                │
│           Final Output                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation Steps (Using Selected Tools)

1. **Integrate Demucs v4 (htdemucs_ft) for Stem Separation**
   ```python
   # pip install demucs
   import torch
   from demucs.pretrained import get_model
   from demucs.apply import apply_model
   import torchaudio
   
   class StemSeparator:
       def __init__(self, device="cuda"):
           self.model = get_model("htdemucs_ft")
           self.model.to(device)
           self.device = device
       
       def separate(self, audio_path):
           """Separate audio into vocals and instrumental."""
           wav, sr = torchaudio.load(audio_path)
           wav = wav.to(self.device)
           
           # Apply model
           with torch.no_grad():
               sources = apply_model(self.model, wav[None], device=self.device)[0]
           
           # htdemucs_ft outputs: drums, bass, other, vocals
           vocals = sources[3]  # vocals
           instrumental = sources[0] + sources[1] + sources[2]  # drums + bass + other
           
           return vocals, instrumental, sr
   ```

2. **Integrate YingMusic-SVC for Voice Conversion**
   ```python
   # Clone: https://github.com/GiantAILab/YingMusic-SVC
   import sys
   sys.path.append("./YingMusic-SVC")
   from inference import YingMusicSVC
   
   class VoiceConverter:
       def __init__(self, device="cuda"):
           self.model = YingMusicSVC.from_pretrained(device=device)
           self.device = device
       
       def convert(self, source_vocals, reference_audio, pitch_shift=0):
           """Convert source vocals to match reference voice timbre."""
           return self.model.convert(
               source=source_vocals,
               reference=reference_audio,
               pitch_shift=pitch_shift,
               # YingMusic-SVC specific: F0-aware timbre adaptor
               use_f0_adaptor=True
           )
   ```

3. **Create Voice Cloning Pipeline**
   ```python
   import torchaudio
   import torch
   
   class VoiceCloningPipeline:
       def __init__(self, heartlib_pipeline, device="cuda"):
           self.heartlib = heartlib_pipeline
           self.stem_separator = StemSeparator(device=device)
           self.voice_converter = VoiceConverter(device=device)
           self.device = device
       
       def generate(self, prompt, tags, lyrics, voice_reference=None, 
                    duration_ms=30000, pitch_shift=0):
           """Generate music with optional voice cloning."""
           # Step 1: Generate with HeartLib
           audio = self.heartlib.generate(
               prompt=prompt,
               tags=tags,
               lyrics=lyrics,
               max_audio_length_ms=duration_ms
           )
           
           if voice_reference is None:
               return audio
           
           # Step 2: Separate stems (Demucs v4)
           vocals, instrumental, sr = self.stem_separator.separate(audio)
           
           # Step 3: Convert vocals (YingMusic-SVC)
           converted_vocals = self.voice_converter.convert(
               source_vocals=vocals,
               reference_audio=voice_reference,
               pitch_shift=pitch_shift
           )
           
           # Step 4: Mix back with proper gain matching
           output = self._mix_audio(converted_vocals, instrumental, sr)
           
           return output
       
       def _mix_audio(self, vocals, instrumental, sr, vocal_gain=1.0):
           """Mix vocals and instrumental with gain control."""
           # Normalize and mix
           vocals = vocals * vocal_gain
           mixed = vocals + instrumental
           
           # Prevent clipping
           max_val = torch.max(torch.abs(mixed))
           if max_val > 1.0:
               mixed = mixed / max_val * 0.95
           
           return mixed, sr
   ```

#### VRAM Budget (GB10) - Updated with Selected Tools

| Component | VRAM (FP16) |
|-----------|-------------|
| HeartMuLa 3B | ~6 GB |
| HeartCodec | ~2 GB |
| Demucs v4 (htdemucs_ft) | ~2-4 GB |
| YingMusic-SVC | ~4-6 GB |
| **Total** | **~14-18 GB** |
| **Available (GB10)** | **128 GB** |

**Headroom:** 110+ GB available for batch processing, longer audio, or parallel generations.

#### Pros & Cons

**Pros:**
- No changes to HeartLib core
- Proven, production-ready technology
- Can be implemented in 1-2 weeks
- Each component can be upgraded independently

**Cons:**
- Two-stage pipeline (generation → conversion)
- Requires stem separation (potential quality loss)
- Latency: adds ~10-20 seconds to generation time
- Voice may not perfectly match musical style

---

### 3.2 Path B: Timbre Embedding Integration

**Timeline:** 1-2 months
**Complexity:** Medium
**Quality:** Better (single-stage)

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Timbre-Conditioned Generation                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Text/Lyrics  │    │    Tags      │    │ Voice Ref    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Text      │    │     Tag      │    │   Speaker    │       │
│  │   Encoder    │    │   Encoder    │    │   Encoder    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                             ▼                                    │
│                    ┌──────────────┐                              │
│                    │  HeartMuLa   │                              │
│                    │  (Modified)  │                              │
│                    │              │                              │
│                    │ muq_embed ←──┼── Timbre Embedding           │
│                    └──────┬───────┘                              │
│                           │                                      │
│                           ▼                                      │
│                    ┌──────────────┐                              │
│                    │  HeartCodec  │                              │
│                    └──────┬───────┘                              │
│                           │                                      │
│                           ▼                                      │
│                    Voice-Cloned Audio                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation Steps

1. **Train/Integrate Speaker Encoder**
   
   Options:
   - **ECAPA-TDNN** (SpeechBrain): State-of-the-art speaker verification
   - **Resemblyzer**: Lightweight, easy to integrate
   - **WavLM + Projection**: Self-supervised, high quality

   ```python
   from speechbrain.pretrained import EncoderClassifier
   
   speaker_encoder = EncoderClassifier.from_hparams(
       source="speechbrain/spkrec-ecapa-voxceleb"
   )
   
   def extract_timbre_embedding(audio_path):
       # Returns 192-dim speaker embedding
       return speaker_encoder.encode_batch(audio_path)
   ```

2. **Modify HeartMuLa to Accept Timbre Embedding**

   The `muq_embed` infrastructure already exists. Modify `music_generation.py`:

   ```python
   def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):
       # ... existing code ...
       
       # NEW: Process voice reference
       voice_ref = inputs.get("voice_reference", None)
       if voice_ref is not None:
           # Extract timbre embedding
           timbre_embed = self.speaker_encoder(voice_ref)
           # Project to muq_dim
           muq_embed = self.timbre_projection(timbre_embed)
       else:
           muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype)
       
       # ... rest of code ...
   ```

3. **Train Timbre Projection Layer**

   Need to train a projection from speaker embedding space to HeartMuLa's muq space:

   ```python
   class TimbreProjection(nn.Module):
       def __init__(self, speaker_dim=192, muq_dim=1024):
           super().__init__()
           self.projection = nn.Sequential(
               nn.Linear(speaker_dim, 512),
               nn.ReLU(),
               nn.Linear(512, muq_dim)
           )
       
       def forward(self, speaker_embed):
           return self.projection(speaker_embed)
   ```

4. **Fine-tune HeartMuLa**

   Options:
   - **LoRA fine-tuning**: Add low-rank adapters, train on paired data
   - **Full fine-tuning**: Requires significant compute and data
   - **Prompt tuning**: Learn soft prompts for timbre conditioning

#### Data Requirements

For training the timbre projection:
- Paired data: same song, different singers
- Or: singer-labeled dataset with multiple songs per singer
- Estimated: 100-1000 hours of labeled singing data

#### Pros & Cons

**Pros:**
- Single-stage generation
- Better voice-music coherence
- No stem separation needed
- Leverages existing HeartLib architecture

**Cons:**
- Requires training/fine-tuning
- Need paired or labeled data
- 1-2 months development time
- May need hyperparameter tuning

---

### 3.3 Path C: Full Voice Cloning Integration (Vevo2-style)

**Timeline:** 3-6 months
**Complexity:** High
**Quality:** Best

This path involves building a unified framework similar to Vevo2, with:
- Content-style tokenizer for timbre disentanglement
- Prosody tokenizer for melody/rhythm preservation
- Joint AR + flow-matching architecture

**Not recommended for initial implementation** due to complexity and timeline. Consider after Path A/B are proven.

---

## 4. Recommended Implementation Plan

### Phase 1: Path A Implementation (Weeks 1-2) - IMMEDIATE PRIORITY

**Selected Tools:** YingMusic-SVC + Demucs v4 (htdemucs_ft)

1. **Week 1: Backend Integration**
   - Clone and set up YingMusic-SVC from GitHub
   - Install Demucs v4 via pip (`pip install demucs`)
   - Create `VoiceCloningPipeline` wrapper class
   - Add FastAPI endpoints for voice cloning:
     - `POST /api/generate-with-voice` - Generate music with voice reference
     - `POST /api/convert-voice` - Convert existing audio to target voice
   - Test pipeline end-to-end on GB10

2. **Week 2: Frontend Integration**
   - Add "Voice Reference" upload component to web UI
   - Implement audio preview for reference samples
   - Add pitch shift slider (-12 to +12 semitones)
   - Create progress indicators for multi-stage pipeline
   - Test full user flow

### Phase 2: Path B Research (Weeks 3-6) - OPTIONAL

Only pursue if Path A quality is insufficient:

1. **Week 3-4:**
   - Evaluate speaker encoders (ECAPA-TDNN, Resemblyzer, WavLM)
   - Collect/prepare training data
   - Design timbre projection architecture

2. **Week 5-6:**
   - Train timbre projection layer
   - Fine-tune HeartMuLa with LoRA
   - A/B test against Path A

### Phase 3: Production (Weeks 7-8)

1. Deploy best-performing approach
2. Add quality controls and fallbacks
3. User testing and iteration

---

## 5. Open Source Resources

### SVC Models

| Repository | URL | Status | Notes |
|------------|-----|--------|-------|
| **YingMusic-SVC** | github.com/GiantAILab/YingMusic-SVC | **SELECTED** | Latest (Dec 2025), Flow-GRPO, robust |
| HQ-SVC | github.com/ShawnPi233/HQ-SVC | Backup | High quality, efficient |
| seed-vc | github.com/Plachtaa/seed-vc | Archived | Proven but unmaintained |
| so-vits-svc-fork | github.com/voicepaw/so-vits-svc-fork | Active | Requires training |

### Speaker Encoders (For Path B)

| Model | Source | Embedding Dim |
|-------|--------|---------------|
| ECAPA-TDNN | SpeechBrain | 192 |
| Resemblyzer | github.com/resemble-ai/Resemblyzer | 256 |
| WavLM | HuggingFace | 768-1024 |

### Stem Separation

| Model | URL | Status | Quality |
|-------|-----|--------|---------|
| **Demucs v4 (htdemucs_ft)** | github.com/facebookresearch/demucs | **SELECTED** | Best (SDR 9.20 dB) |
| Spleeter | github.com/deezer/spleeter | Alternative | Fast but lower quality |

---

## 6. Technical Considerations

### 6.1 Audio Quality Preservation

- Use 48kHz sample rate throughout pipeline
- Avoid multiple encode/decode cycles
- Use high-quality resampling (librosa, torchaudio)

### 6.2 Pitch Handling

SVC models need pitch information:
- Extract F0 from source vocals
- Optionally shift to match reference singer's range
- Preserve relative pitch contour

### 6.3 Real-time Considerations

For live/streaming use cases:
- seed-vc supports real-time conversion
- Chunk-based processing for low latency
- GPU memory management for concurrent users

### 6.4 Ethical Considerations

- Require consent for voice cloning
- Watermark generated audio
- Implement voice verification to prevent misuse
- Clear attribution in generated content

---

## 7. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Timbre Similarity | >0.8 | Speaker verification cosine similarity |
| Audio Quality | MOS >4.0 | Mean Opinion Score listening tests |
| Latency | <30s total | End-to-end generation time |
| VRAM Usage | <20GB | Peak memory during generation |

---

## 8. Conclusion

Voice cloning for HeartLib is highly feasible with current technology.

### Final Tool Selection

| Component | Tool | Why |
|-----------|------|-----|
| **Voice Conversion** | YingMusic-SVC | Latest research (Dec 2025), Flow-GRPO compatible with HeartCodec, robust to real-world audio |
| **Stem Separation** | Demucs v4 (htdemucs_ft) | State-of-the-art quality (SDR 9.20 dB), MIT license |

### Implementation Path

1. **Immediate (Path A, Weeks 1-2):** Implement post-processing SVC pipeline using **YingMusic-SVC + Demucs v4**. This provides working voice cloning with proven quality and minimal risk.

2. **Optional (Path B, Weeks 3-6):** If Path A quality is insufficient, develop timbre embedding integration to leverage HeartLib's existing `muq_embed` infrastructure for cleaner, single-stage generation.

3. **Long-term (Path C):** Consider full Vevo2-style integration only after validating user demand and quality requirements.

### Resource Summary

- **Total VRAM:** ~14-18 GB (HeartLib + Demucs + YingMusic-SVC)
- **Available (GB10):** 128 GB unified memory
- **Headroom:** 110+ GB for batch processing or parallel generations

The GB10's 128GB unified memory provides ample headroom for running HeartLib + SVC models concurrently, making this a technically straightforward integration.

---

## References

1. arXiv:2508.16332 - Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation
2. arXiv:2501.13870 - Everyone-Can-Sing: Zero-Shot Singing Voice Synthesis and Conversion with Speech Reference
3. arXiv:2512.04793 - YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion
4. arXiv:2509.15629 - The Singing Voice Conversion Challenge 2025
5. arXiv:2502.12572 - TechSinger: Technique Controllable Multilingual Singing Voice Synthesis
6. arXiv:2511.08496 - HQ-SVC: High-Quality Zero-Shot Singing Voice Conversion
7. arXiv:2409.08583 - LHQ-SVC: Lightweight and High Quality Singing Voice Conversion Modeling

---

*Document Version: 2.0*
*Last Updated: 2026-01-21*
*Author: Devin (Research Assistant)*
