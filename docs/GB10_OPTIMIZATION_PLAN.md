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

## 11. Advanced Optimization Strategies (Outside-the-Box)

### 11.1 Consistency Distillation for Flow Matching → 1-Step Generation

**Source**: [arXiv:2503.20349](https://arxiv.org/abs/2503.20349) - Consistency Trajectory Matching

The HeartCodec FlowMatching uses 20 Euler ODE steps. This is the **biggest opportunity** for dramatic speedup.

**Key Insight**: Recent work shows flow matching models can be distilled to **1-4 steps** with minimal quality loss:

| Method | Steps | Speedup | Quality |
|--------|-------|---------|---------|
| Euler (current) | 20 | 1x | Baseline |
| Consistency Distillation | 4 | 5x | ~98% |
| **Consistency Trajectory Matching** | 1 | **20x** | ~95% |

**Implementation for HeartCodec**:
1. Train a student FlowMatching model using CTM loss
2. The student learns the entire ODE trajectory in one step
3. No architectural changes needed - just weight replacement

**Reference**: [arXiv:2505.18825](https://arxiv.org/abs/2505.18825) - "How to Build a Consistency Model" shows Lagrangian methods achieve stable training

### 11.2 Lookahead Decoding (No Draft Model Required)

**Source**: [arXiv:2402.02057](https://arxiv.org/abs/2402.02057) - ICML 2024

Unlike speculative decoding which needs a separate draft model, **Lookahead Decoding** uses Jacobi iteration to parallelize autoregressive generation with the SAME model.

```
Standard AR:     [t1] → [t2] → [t3] → [t4] → [t5]  (5 steps)
Lookahead:       [t1,t2,t3,t4,t5] → verify → accept 3 → [t4,t5,t6,t7,t8] → ...
```

**Speedup**: 1.8x on MT-bench, up to 4x with strong scaling

**Why this is perfect for HeartMuLa**:
- No need to maintain a separate draft model
- Works with existing KV cache
- Compatible with FlashAttention
- The 300M decoder is already there - can use it for verification

### 11.3 Jacobi Forcing: Progressive Distillation for Parallel Decoding

**Source**: [arXiv:2512.14681](https://arxiv.org/abs/2512.14681) - December 2025

**Key Innovation**: Train the model to accept parallel token predictions through progressive distillation.

**Results**:
- **3.8x wall-clock speedup** on coding/math benchmarks
- Multi-block decoding with rejection recycling: **4.5x higher token acceptance**
- Nearly **4.0x wall-clock speedup** overall

**Application to HeartMuLa**:
- Fine-tune HeartMuLa backbone with Jacobi Forcing loss
- Progressive noise schedule handles AR/parallel mismatch
- Each iteration decodes multiple audio frames

### 11.4 LayerSkip + Self-Speculative Decoding

**Source**: [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)

**Radical Idea**: Use EARLY layers of the 3B model as the draft, LATER layers for verification.

```
┌─────────────────────────────────────────┐
│         HeartMuLa 3B (28 layers)        │
├─────────────────────────────────────────┤
│ Layers 1-8:   Draft tokens (fast)       │
│ Layers 9-28:  Verify/correct (accurate) │
└─────────────────────────────────────────┘
```

**Speedup**: Up to 2.16x on summarization, 1.82x on coding

**Why this works**:
- Early transformer layers capture basic patterns
- Later layers refine and correct
- Training with layer dropout makes early exits accurate
- **No separate draft model needed**

### 11.5 Medusa Heads: Multi-Token Prediction Without Draft Model

**Source**: [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)

Add lightweight prediction heads that generate multiple future tokens:

```
HeartMuLa Output → [Head 0: t+1]
                   [Head 1: t+2]
                   [Head 2: t+3]
                   [Head 3: t+4]

Tree attention verifies all candidates in parallel
```

**Speedup**: 2.2-3.6x depending on training approach

**Medusa-1** (lossless): Freeze backbone, train heads only → 2.2x
**Medusa-2** (joint training): Fine-tune everything → 3.6x

### 11.6 KVSwap: Disk-Aware Cache for Edge Devices

**Source**: [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) - November 2025

For GB10's limited RAM, use NVMe/eMMC as KV cache overflow:

```
┌─────────────────────────────────────────┐
│              Memory Hierarchy            │
├─────────────────────────────────────────┤
│ L1/L2 Cache:   Hot attention heads       │
│ RAM (16GB):    Active KV entries         │
│ NVMe/eMMC:     Cold KV entries           │
│ Reuse Buffer:  Recently accessed KV      │
└─────────────────────────────────────────┘
```

**Key Techniques**:
- Compressed K cache reduces memory overhead
- Grouped KV prediction for disk I/O optimization
- Software reuse buffer for low-bandwidth devices (<200 MB/s)

### 11.7 Streaming Sliding Window Attention

**Source**: [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) - SWAA

For long music generation (3+ minutes), full attention is memory-prohibitive.

**SWAA Strategies**:
1. Apply sliding window only during prefilling
2. Preserve "attention sink" tokens (first few tokens)
3. Interleave full/sliding attention layers
4. Fine-tune with sliding window

**Result**: 30-100% speedup with acceptable quality loss

**For HeartMuLa**: Music has strong local structure - sliding window is natural fit.

### 11.8 Streaming Codec Architecture

**Source**: [arXiv:2509.16195](https://arxiv.org/abs/2509.16195) - FocalCodec-Stream

Current HeartCodec processes full audio at once. Streaming architecture enables:
- **80ms theoretical latency**
- Causal processing (no future lookahead)
- Progressive output while generating

**Key Insight**: The ScalarModel in HeartCodec already supports `causal=True` mode. Enable it for streaming.

---

## 12. Aggressive Optimization Stack (Maximum Speed)

Combining the most promising techniques for GB10:

```
┌─────────────────────────────────────────────────────────────┐
│                 AGGRESSIVE OPTIMIZATION STACK                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Consistency Distillation (FlowMatching)                  │
│     └─ 20 steps → 1-2 steps = 10-20x speedup                │
│                                                              │
│  2. Jacobi Forcing + Lookahead Decoding (HeartMuLa)         │
│     └─ AR bottleneck broken = 3-4x speedup                  │
│                                                              │
│  3. LayerSkip Self-Speculation                               │
│     └─ Early layers draft, late layers verify = 1.8x        │
│                                                              │
│  4. Sliding Window Attention                                 │
│     └─ Memory bounded, long context efficient = 1.5x        │
│                                                              │
│  5. KVSwap Disk Offloading                                   │
│     └─ Enables larger batch / longer context                │
│                                                              │
│  6. ARM NEON + MNN Kernels                                   │
│     └─ Hardware-optimized matmul/attention = 2x             │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  COMBINED THEORETICAL SPEEDUP: 50-100x over naive baseline  │
│  TARGET RTF: 0.1-0.3 (3-10x faster than real-time)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Experimental Priority Matrix

| Technique | Implementation Effort | Expected Gain | Risk | Priority |
|-----------|----------------------|---------------|------|----------|
| Reduce ODE steps (20→10) | Low | 2x | Low | **P0** |
| Consistency Distillation | High | 10-20x | Medium | **P1** |
| Lookahead Decoding | Medium | 1.8-4x | Low | **P1** |
| LayerSkip | Medium | 1.8x | Medium | P2 |
| Medusa Heads | Medium | 2-3x | Medium | P2 |
| Sliding Window | Low | 1.5x | Low | P2 |
| KVSwap | High | Memory only | Medium | P3 |
| Jacobi Forcing | High | 3-4x | High | P3 |

**Recommended Order**:
1. P0: Quick wins, no training required
2. P1: High impact, moderate effort
3. P2: Stack on top of P1 gains
4. P3: Only if P1/P2 insufficient

---

## 14. Additional Research References

### Flow Matching Distillation
- [arXiv:2503.20349](https://arxiv.org/abs/2503.20349) - Consistency Trajectory Matching (1-step)
- [arXiv:2505.18825](https://arxiv.org/abs/2505.18825) - How to Build a Consistency Model
- [arXiv:2506.14603](https://arxiv.org/abs/2506.14603) - Align Your Flow (continuous-time distillation)

### Parallel Decoding
- [arXiv:2402.02057](https://arxiv.org/abs/2402.02057) - Lookahead Decoding (ICML 2024)
- [arXiv:2512.14681](https://arxiv.org/abs/2512.14681) - Jacobi Forcing
- [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) - Medusa Heads

### Layer Efficiency
- [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) - LayerSkip
- [arXiv:2504.08850](https://arxiv.org/abs/2504.08850) - SpecEE (Speculative Early Exit)
- [arXiv:2506.21103](https://arxiv.org/abs/2506.21103) - Middle-Outward Layer Skipping

### Memory & Attention
- [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) - KVSwap (disk-aware KV cache)
- [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) - SWAA (sliding window adaptation)
- [arXiv:2509.04377](https://arxiv.org/abs/2509.04377) - PagedEviction

### Streaming Audio
- [arXiv:2509.16195](https://arxiv.org/abs/2509.16195) - FocalCodec-Stream (80ms latency)
- [arXiv:2503.11562](https://arxiv.org/abs/2503.11562) - BRAVE (low-latency neural synthesizer)

---

*Document Version: 2.0*
*Last Updated: 2026-01-20*
*Author: Claude (for jhacksman/heartlib-gb10)*
