# HeartLib GB10 Optimization Plan

## Executive Summary

This document outlines a comprehensive optimization strategy for running HeartMuLa (a 3B parameter music generation model) on the GB10 ARM64 board. The goal is to achieve maximum inference speed **without quantization**, focusing on parallelization, speculative decoding, and ARM64-specific optimizations.

---

## 1. Architecture Analysis

### 1.1 HeartMuLa System Overview

HeartMuLa is a **token-based autoregressive music generation system**, NOT a diffusion model. It consists of:

| Component | Type | Parameters | Bottleneck Level |
|-----------|------|------------|------------------|
| HeartMuLa LM | Llama 3.2 (3B) | ~3B | **HIGH** - Autoregressive generation |
| HeartMuLa Decoder | Llama 3.2 (300M) | ~300M | MEDIUM |
| HeartCodec FlowMatching | Transformer + RVQ | ~500M | **HIGH** - Euler solver loop |
| HeartCodec ScalarModel | CNN Encoder-Decoder | ~50M | LOW |

### 1.2 How Music Generation Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    HeartMuLa Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Tags + Lyrics                                           │
│           ↓                                                     │
│  ┌─────────────────┐                                            │
│  │   Tokenizer     │  Text → Token IDs                          │
│  └────────┬────────┘                                            │
│           ↓                                                     │
│  ┌─────────────────┐                                            │
│  │  HeartMuLa 3B   │  Autoregressive frame-by-frame generation  │
│  │  (Backbone +    │  Generates 8 codebook tokens per frame     │
│  │   Decoder)      │  @ 12.5 Hz (12.5 frames/second of audio)   │
│  └────────┬────────┘                                            │
│           ↓                                                     │
│  Audio Tokens: [B, 8 codebooks, T frames]                       │
│           ↓                                                     │
│  ┌─────────────────┐                                            │
│  │ HeartCodec      │  Tokens → Latent (via ResidualVQ lookup)   │
│  │ FlowMatching    │  Latent → Refined Latent (Euler ODE)       │
│  │ (Transformer)   │  20 steps of flow matching                 │
│  └────────┬────────┘                                            │
│           ↓                                                     │
│  ┌─────────────────┐                                            │
│  │ HeartCodec      │  Latent → Waveform (CNN decoder)           │
│  │ ScalarModel     │  @ 48kHz sample rate                       │
│  └────────┬────────┘                                            │
│           ↓                                                     │
│  Output: Audio Waveform                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Current Performance Baseline

From GitHub Issue #14:
- RTX 3090 (24GB): ~1.0 RTF (3 min song = 3 min generation)
- RTX 5080 (16GB): ~11 min for 5 seconds (severe VRAM bottleneck)
- After PR #5 fix: ~5 min for 3 min song

**GB10 Target**: Achieve sub-2x RTF on ARM64 without quantization.

---

## 2. Bottleneck Analysis

### 2.1 Primary Bottlenecks (Ranked by Impact)

#### Bottleneck #1: Autoregressive LM Generation (HeartMuLa 3B)
- **Problem**: Sequential token generation, cannot parallelize across time
- **Impact**: ~60% of total inference time
- **Solution**: Speculative decoding (see Section 3.1)

#### Bottleneck #2: Flow Matching Euler Solver
- **Problem**: 20 sequential ODE steps, each requiring full transformer forward pass
- **Impact**: ~25% of total inference time
- **Location**: `flow_matching.py:solve_euler()` lines 85-120
- **Solution**: Reduce steps + parallel segment processing (see Section 3.2)

#### Bottleneck #3: CFG Duplication
- **Problem**: Classifier-free guidance requires 2x forward passes (conditional + unconditional)
- **Impact**: ~15% overhead when CFG != 1.0
- **Location**: `flow_matching.py` line 104-115
- **Solution**: Batch CFG pairs together (see Section 3.3)

### 2.2 Memory Access Patterns

| Operation | Memory Pattern | ARM64 Concern |
|-----------|---------------|---------------|
| Attention Q/K/V | Large matmul | Needs tiled GEMM |
| RoPE | Sequential access | Cache-friendly already |
| FFN (SwiGLU) | Large matmul | Needs tiled GEMM |
| Conv1D (codec) | Strided access | NEON vectorization |
| VQ Lookup | Random access | Prefetch needed |

---

## 3. Optimization Strategies (No Quantization)

### 3.1 Speculative Decoding for HeartMuLa

**Source**: [arXiv:2410.13839](https://arxiv.org/abs/2410.13839) - "Accelerating Codec-based Speech Synthesis with Multi-Token Prediction and Speculative Decoding"

**Expected Speedup**: 4-5x reduction in token generation time

#### 3.1.1 Multi-Token Prediction Heads

Add K prediction heads to generate K tokens per forward pass:

```
Current:     [Frame N] → Forward Pass → [Frame N+1]
Proposed:    [Frame N] → Forward Pass → [Frame N+1, N+2, N+3, N+4]
```

**Implementation**:
1. Add 4 additional output projection heads to HeartMuLa decoder
2. Train heads to predict future frames (can use knowledge distillation)
3. Use Viterbi algorithm to select optimal token sequence

**No Retraining Option**: Use draft model approach:
- Use the 300M decoder as "draft" model
- Use 3B backbone to verify/reject speculated tokens
- Accept ~3-4 tokens per verification step on average

#### 3.1.2 Saguaro-style Speculative Speculative Decoding

**Source**: [OpenReview: Saguaro](https://openreview.net/forum?id=aL1Wnml9Ef)

While verification is running, pre-compute likely next speculations:
- 2x faster than standard speculative decoding
- 5x faster than pure autoregressive

### 3.2 Flow Matching Optimization

#### 3.2.1 Reduce ODE Steps

Current: 20 Euler steps
Proposed: Investigate step reduction

| Steps | Quality Impact | Speed Gain |
|-------|----------------|------------|
| 20    | Baseline       | 1.0x       |
| 10    | Minimal        | 2.0x       |
| 5     | Moderate       | 4.0x       |
| 1     | Consistency distillation needed | 20x |

**Recommendation**: Test quality at 10 steps, fall back to 15 if needed.

#### 3.2.2 Parallel Segment Processing

For long audio (>30 seconds), process segments in parallel:

```
Current:
  [Seg1] → [Seg2] → [Seg3] → [Seg4]    (Sequential)

Proposed:
  [Seg1] ─┐
  [Seg2] ─┼→ Parallel Flow Matching → Overlap-add
  [Seg3] ─┤
  [Seg4] ─┘
```

HeartCodec already supports `context_latents` for segment continuity. We can:
1. Run LM generation sequentially (required for coherence)
2. Batch all segments' flow matching in parallel
3. Use overlap-add for seamless stitching

### 3.3 CFG Batching Optimization

Instead of 2 sequential forward passes:

```python
# Current (2 passes)
cond_out = model(x, cond)
uncond_out = model(x, uncond)
out = uncond_out + scale * (cond_out - uncond_out)

# Optimized (1 batched pass)
batch_x = torch.cat([x, x], dim=0)
batch_cond = torch.cat([cond, uncond], dim=0)
batch_out = model(batch_x, batch_cond)
cond_out, uncond_out = batch_out.chunk(2)
out = uncond_out + scale * (cond_out - uncond_out)
```

**Note**: FlowMatching already does this (line 104-108). Ensure HeartMuLa does too.

---

## 4. ARM64-Specific Optimizations

### 4.1 NEON SIMD Vectorization

**Source**: [arXiv:2506.10443](https://arxiv.org/abs/2506.10443) - MNN-LLM for ARM mobile deployment

#### 4.1.1 Target Operations

| Operation | Current | ARM64 Optimization |
|-----------|---------|-------------------|
| MatMul | PyTorch default | ARM Compute Library / XNNPACK |
| SiLU activation | Scalar | NEON vfmaq_f32 |
| RMSNorm | Scalar | NEON reduction + rsqrt |
| Softmax | Scalar | NEON exp + reduction |
| Rotary embedding | Scalar | NEON sin/cos LUT |

#### 4.1.2 Snake Activation (Already Optimized)

The codec uses a Snake activation that's already TorchScript-optimized:

```python
@torch.jit.script
def snake(x, alpha):
    # 1.4x speedup from JIT
    ...
```

Extend this pattern to other custom ops.

### 4.2 Memory Layout Optimization

**Source**: MNN-LLM paper Section 4.2

#### 4.2.1 Weight Rearrangement

Rearrange weight matrices for ARM CPU cache efficiency:
- Tile weights into 4x4 or 8x8 blocks for NEON register utilization
- Interleave weights for vectorized FMA operations

#### 4.2.2 Activation Memory

- Use in-place operations where possible
- Implement activation checkpointing for large sequences
- Pre-allocate KV cache buffers

### 4.3 Multi-Core Load Balancing

GB10 has 8 ARM cores. Distribute work:

```
┌────────────────────────────────────────────────┐
│              GB10 Core Allocation              │
├────────────────────────────────────────────────┤
│ Cores 0-5:  Transformer attention/FFN (6 cores)│
│ Core 6:     KV cache management                │
│ Core 7:     Memory prefetch + I/O              │
└────────────────────────────────────────────────┘
```

Use OpenMP or similar for parallel attention head computation:
- 24 attention heads ÷ 6 cores = 4 heads per core

---

## 5. Framework Recommendations

### 5.1 Primary: MNN Framework

**Why**: Purpose-built for ARM LLM inference with NEON optimization

- Pre-built ARM NEON kernels
- Mixed-precision support (FP32/FP16/BF16)
- DRAM-Flash hybrid for memory efficiency
- Up to 8.6x speedup over baseline

**Integration Path**:
1. Export HeartMuLa to ONNX
2. Convert ONNX to MNN
3. Replace attention/FFN with MNN optimized ops
4. Keep PyTorch for pipeline orchestration

### 5.2 Alternative: ExecuTorch (Meta)

**Why**: Official PyTorch edge deployment solution

- Direct PyTorch model export
- ARM backend with XNNPACK
- Easier integration than MNN

### 5.3 Alternative: ONNX Runtime Mobile

**Why**: Cross-platform with ARM optimizations

- XNNPACK execution provider
- Good transformer support
- Easy ONNX conversion from PyTorch

---

## 6. Implementation Phases

### Phase 1: Baseline Measurement (Week 1)
- [ ] Port existing code to GB10
- [ ] Measure baseline inference time
- [ ] Profile hotspots with `torch.profiler`
- [ ] Document memory usage patterns

### Phase 2: Quick Wins (Week 2)
- [ ] Reduce flow matching steps to 10 (if quality acceptable)
- [ ] Ensure CFG batching is used everywhere
- [ ] Enable `torch.compile()` with `inductor` backend
- [ ] TorchScript all custom activations

### Phase 3: Speculative Decoding (Weeks 3-4)
- [ ] Implement draft model approach using 300M decoder
- [ ] Add verification logic to 3B backbone
- [ ] Tune acceptance threshold
- [ ] Measure token acceptance rate

### Phase 4: ARM64 Kernels (Weeks 5-6)
- [ ] Integrate MNN or XNNPACK for matmul
- [ ] Implement NEON RMSNorm
- [ ] Implement NEON Rotary Embeddings
- [ ] Multi-core attention distribution

### Phase 5: Parallel Flow Matching (Week 7)
- [ ] Implement segment batching
- [ ] Add overlap-add stitching
- [ ] Test audio quality at segment boundaries

### Phase 6: Integration & Testing (Week 8)
- [ ] End-to-end pipeline testing
- [ ] Quality evaluation (listening tests)
- [ ] Final performance benchmarks
- [ ] Documentation

---

## 7. Expected Performance Gains

| Optimization | Individual Gain | Cumulative RTF |
|--------------|-----------------|----------------|
| Baseline (GPU RTX 3090) | 1.0x | 1.0 |
| Baseline (GB10 ARM64) | ~0.1-0.2x | 5.0-10.0 |
| + Reduce flow steps (20→10) | 1.5-2.0x | 3.0-5.0 |
| + CFG batching | 1.2x | 2.5-4.0 |
| + Speculative decoding (4x) | 3.0-4.0x | 0.8-1.5 |
| + ARM NEON kernels | 1.5-2.0x | 0.5-1.0 |
| + Multi-core parallelism | 1.3-1.5x | 0.4-0.7 |

**Target**: Achieve **sub-1.0 RTF** on GB10 (real-time or faster)

---

## 8. Research Papers Summary

### Primary References

1. **Multi-Token Prediction + Speculative Decoding for Codec Speech**
   - arXiv:2410.13839
   - 4-5x speedup, Viterbi-based token selection
   - Directly applicable to HeartMuLa

2. **MNN-LLM: ARM Mobile LLM Deployment**
   - arXiv:2506.10443
   - 8.6x speedup with ARM NEON + memory optimization
   - Key techniques: weight rearrangement, multicore balancing

3. **Saguaro: Speculative Speculative Decoding**
   - OpenReview
   - 2x faster than standard speculative decoding
   - Pre-compute next speculations during verification

4. **LLM Inference Acceleration Hardware Survey**
   - arXiv:2410.04466
   - Comprehensive CPU/GPU/FPGA/ASIC comparison
   - Memory bandwidth analysis

5. **VADUSA: High-Quality AR Speech with Speculative Decoding**
   - arXiv:2410.21951
   - Draft heads predict future content auto-regressively
   - Improves both speed AND quality

### Secondary References

6. **KV Cache Management Survey** (arXiv:2412.19442)
   - Token/model/system-level optimizations

7. **2:4 Activation Sparsity** (arXiv:2503.16672)
   - 50% activation sparsity without quality loss
   - May require fine-tuning

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Quality degradation from fewer ODE steps | Medium | High | A/B testing at each step count |
| Speculative decoding low acceptance rate | Low | Medium | Tune temperature, fallback to AR |
| ARM kernel bugs | Medium | Medium | Extensive unit testing |
| Memory OOM on GB10 | High | High | Sequential model loading (PR #5) |
| Segment boundary artifacts | Medium | Medium | Overlap-add with cross-fade |

---

## 10. Success Criteria

1. **Performance**: Achieve ≤1.5x RTF on GB10 for 3-minute songs
2. **Quality**: No perceptible audio quality degradation (blind listening test)
3. **Memory**: Peak VRAM/RAM usage ≤12GB
4. **Stability**: 100 consecutive generations without crash
5. **Latency**: Time-to-first-audio ≤10 seconds

---

## Appendix A: File Locations for Optimization

```
src/heartlib/
├── heartmula/
│   └── modeling_heartmula.py     # LM backbone - speculative decoding target
├── heartcodec/
│   ├── modeling_heartcodec.py    # Codec entry point
│   └── models/
│       ├── flow_matching.py      # solve_euler() - step reduction target
│       ├── transformer.py        # LlamaAttention - NEON target
│       └── sq_codec.py           # snake() already JIT-optimized
└── pipelines/
    └── music_generation.py       # Pipeline orchestration
```

## Appendix B: GB10 Hardware Specs (Assumed)

- CPU: 8-core ARM Cortex-A78 or similar
- RAM: 16GB LPDDR5
- Compute: ARM Compute Library / NEON
- Target power: <25W

---

*Document Version: 1.0*
*Last Updated: 2026-01-20*
*Author: Claude (for jhacksman/heartlib-gb10)*
