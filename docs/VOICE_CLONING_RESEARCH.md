# Voice Cloning for HeartLib: Research & Implementation Plan

## Executive Summary

This document outlines the research findings and implementation paths for adding voice cloning capabilities to HeartLib. The goal is to enable users to upload a reference audio sample and have HeartLib generate music with vocals that match the reference voice's timbre.

**Key Finding:** Zero-shot Singing Voice Conversion (SVC) is a mature technology. The 2025 Singing Voice Conversion Challenge demonstrated that top systems achieve "comparable singer identity scores to ground truth samples." Multiple open-source implementations exist that can run on 4-8GB VRAM, making integration with HeartLib on GB10 (128GB unified memory) highly feasible.

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

#### Implementation Steps

1. **Integrate Demucs for Stem Separation**
   ```python
   # pip install demucs
   from demucs import separate
   
   def separate_stems(audio_path):
       # Returns: vocals, drums, bass, other
       stems = separate.main(["--two-stems", "vocals", audio_path])
       return stems["vocals"], stems["no_vocals"]
   ```

2. **Integrate SVC Model (seed-vc recommended)**
   ```python
   # Clone: https://github.com/Plachtaa/seed-vc
   from seed_vc import SeedVC
   
   svc_model = SeedVC.from_pretrained("seed-vc-v1")
   
   def convert_voice(source_vocals, reference_audio):
       return svc_model.convert(
           source=source_vocals,
           reference=reference_audio,
           pitch_shift=0  # Auto-detect
       )
   ```

3. **Create Pipeline Wrapper**
   ```python
   class VoiceCloningPipeline:
       def __init__(self, heartlib_pipeline, svc_model, demucs_model):
           self.heartlib = heartlib_pipeline
           self.svc = svc_model
           self.demucs = demucs_model
       
       def generate(self, prompt, tags, lyrics, voice_reference=None):
           # Step 1: Generate with HeartLib
           audio = self.heartlib(prompt, tags, lyrics)
           
           if voice_reference is None:
               return audio
           
           # Step 2: Separate stems
           vocals, instrumental = self.demucs.separate(audio)
           
           # Step 3: Convert vocals
           converted_vocals = self.svc.convert(vocals, voice_reference)
           
           # Step 4: Mix back
           return self.mix(converted_vocals, instrumental)
   ```

#### VRAM Budget (GB10)

| Component | VRAM (FP16) |
|-----------|-------------|
| HeartMuLa 3B | ~6 GB |
| HeartCodec | ~2 GB |
| Demucs | ~1 GB |
| seed-vc | ~2-4 GB |
| **Total** | **~11-13 GB** |
| **Available** | **128 GB** |

Plenty of headroom for concurrent processing.

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

### Phase 1: Path A (Weeks 1-2)

1. **Week 1:**
   - Integrate Demucs for stem separation
   - Set up seed-vc or YingMusic-SVC
   - Create basic pipeline wrapper
   - Add API endpoint for voice reference upload

2. **Week 2:**
   - Integrate into web frontend
   - Add "Voice Reference" upload in UI
   - Test end-to-end pipeline
   - Optimize for latency

### Phase 2: Path B Research (Weeks 3-6)

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

| Repository | URL | Notes |
|------------|-----|-------|
| seed-vc | github.com/Plachtaa/seed-vc | Archived but functional |
| YingMusic-SVC | github.com/GiantAILab/YingMusic-SVC | Latest, robust |
| HQ-SVC | github.com/ShawnPi233/HQ-SVC | High quality |
| so-vits-svc-fork | github.com/voicepaw/so-vits-svc-fork | Community maintained |

### Speaker Encoders

| Model | Source | Embedding Dim |
|-------|--------|---------------|
| ECAPA-TDNN | SpeechBrain | 192 |
| Resemblyzer | github.com/resemble-ai/Resemblyzer | 256 |
| WavLM | HuggingFace | 768-1024 |

### Stem Separation

| Model | URL | Quality |
|-------|-----|---------|
| Demucs v4 | github.com/facebookresearch/demucs | Best |
| Spleeter | github.com/deezer/spleeter | Fast |

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

Voice cloning for HeartLib is highly feasible with current technology. The recommended approach is:

1. **Immediate (Path A):** Implement post-processing SVC pipeline using seed-vc or YingMusic-SVC. This provides working voice cloning in 1-2 weeks with proven quality.

2. **Medium-term (Path B):** Develop timbre embedding integration to leverage HeartLib's existing `muq_embed` infrastructure for cleaner, single-stage generation.

3. **Long-term (Path C):** Consider full Vevo2-style integration only after validating user demand and quality requirements.

The GB10's 128GB unified memory provides ample headroom for running HeartLib + SVC models concurrently, making this a technically straightforward integration.

---

## References

1. arXiv:2508.16332 - Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation
2. arXiv:2501.13870 - Everyone-Can-Sing: Zero-Shot Singing Voice Synthesis and Conversion with Speech Reference
3. arXiv:2512.04793 - YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion
4. arXiv:2509.15629 - The Singing Voice Conversion Challenge 2025
5. arXiv:2502.12572 - TechSinger: Technique Controllable Multilingual Singing Voice Synthesis
6. arXiv:2511.08496 - HQ-SVC: High-Quality Zero-Shot Singing Voice Conversion

---

*Document Version: 1.0*
*Last Updated: 2026-01-21*
*Author: Devin (Research Assistant)*
