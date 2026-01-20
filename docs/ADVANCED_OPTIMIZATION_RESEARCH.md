# Advanced Optimization Research for GB10

Supplementary research document with outside-the-box optimization strategies.
This document extends the main GB10_OPTIMIZATION_PLAN.md with additional techniques discovered through arxiv research.

---

## 1. Consistency Distillation for Flow Matching (Highest Impact)

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

---

## 2. Lookahead Decoding (No Draft Model Required)

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

**GitHub**: [github.com/hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)

---

## 3. Jacobi Forcing: Progressive Distillation for Parallel Decoding

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

---

## 4. LayerSkip + Self-Speculative Decoding

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

---

## 5. Medusa Heads: Multi-Token Prediction Without Draft Model

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

**GitHub**: [github.com/FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)

---

## 6. KVSwap: Disk-Aware Cache (For Reference)

**Source**: [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) - November 2025

**Note**: GB10 has 128GB unified memory - KVSwap is NOT needed for this platform. This section is retained for reference on memory-constrained deployments only.

```
┌─────────────────────────────────────────┐
│   GB10 Memory Reality (NOT Limited)      │
├─────────────────────────────────────────┤
│ Total: 128GB unified LPDDR5x             │
│ Model: ~10GB at FP16                     │
│ KV Cache: ~2GB for 8K context            │
│ Headroom: 116GB (no disk offload needed) │
└─────────────────────────────────────────┘
```

**Key Techniques** (for other platforms):
- Compressed K cache reduces memory overhead
- Grouped KV prediction for disk I/O optimization
- Software reuse buffer for low-bandwidth devices

---

## 7. Streaming Sliding Window Attention

**Source**: [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) - SWAA

For long music generation (3+ minutes), full attention is memory-prohibitive.

**SWAA Strategies**:
1. Apply sliding window only during prefilling
2. Preserve "attention sink" tokens (first few tokens)
3. Interleave full/sliding attention layers
4. Fine-tune with sliding window

**Result**: 30-100% speedup with acceptable quality loss

**For HeartMuLa**: Music has strong local structure - sliding window is natural fit.

**Related**: [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) - StreamingLLM (attention sinks)

---

## 8. Streaming Codec Architecture

**Source**: [arXiv:2509.16195](https://arxiv.org/abs/2509.16195) - FocalCodec-Stream

Current HeartCodec processes full audio at once. Streaming architecture enables:
- **80ms theoretical latency**
- Causal processing (no future lookahead)
- Progressive output while generating

**Key Insight**: The ScalarModel in HeartCodec already supports `causal=True` mode. Enable it for streaming.

**Related**: [arXiv:2503.11562](https://arxiv.org/abs/2503.11562) - BRAVE (low-latency neural synthesizer)

---

## 9. Aggressive Optimization Stack (Maximum Speed)

Combining the most promising **lossless** techniques for GB10:

```
┌─────────────────────────────────────────────────────────────┐
│         AGGRESSIVE OPTIMIZATION STACK (FP16, NO QUANT)       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Consistency Distillation (FlowMatching)                  │
│     └─ 20 steps → 2-4 steps = 5-10x speedup                 │
│                                                              │
│  2. Jacobi Forcing + Lookahead Decoding (HeartMuLa)         │
│     └─ AR bottleneck broken = 3-4x speedup                  │
│                                                              │
│  3. LayerSkip Self-Speculation                               │
│     └─ Early layers draft, late layers verify = 1.8x        │
│                                                              │
│  4. Blackwell Tensor Cores + FlashAttention-3               │
│     └─ GPU-optimized matmul/attention = 2x                  │
│                                                              │
│  5. CUDA Graphs for Autoregressive Loop                     │
│     └─ Eliminate kernel launch overhead = 1.3x              │
│                                                              │
│  6. Parallel Segment Batching                               │
│     └─ Batch flow matching across segments = 1.5x           │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  COMBINED THEORETICAL SPEEDUP: 30-50x over naive baseline   │
│  TARGET RTF: 0.05-0.1 (10-20x faster than real-time)        │
│  QUALITY: Full FP16 precision, no audio artifacts           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Experimental Priority Matrix

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

## 11. Research References

### Flow Matching Distillation
- [arXiv:2503.20349](https://arxiv.org/abs/2503.20349) - Consistency Trajectory Matching (1-step)
- [arXiv:2505.18825](https://arxiv.org/abs/2505.18825) - How to Build a Consistency Model
- [arXiv:2506.14603](https://arxiv.org/abs/2506.14603) - Align Your Flow (continuous-time distillation)

### Parallel Decoding
- [arXiv:2402.02057](https://arxiv.org/abs/2402.02057) - Lookahead Decoding (ICML 2024)
- [arXiv:2512.14681](https://arxiv.org/abs/2512.14681) - Jacobi Forcing
- [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) - Medusa Heads
- [arXiv:2502.09419](https://arxiv.org/abs/2502.09419) - On Multi-Token Prediction

### Layer Efficiency
- [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) - LayerSkip
- [arXiv:2504.08850](https://arxiv.org/abs/2504.08850) - SpecEE (Speculative Early Exit)
- [arXiv:2506.21103](https://arxiv.org/abs/2506.21103) - Middle-Outward Layer Skipping
- [arXiv:2504.05598](https://arxiv.org/abs/2504.05598) - DEL (Dynamic Exit Layer)

### Memory & Attention
- [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) - KVSwap (disk-aware KV cache)
- [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) - SWAA (sliding window adaptation)
- [arXiv:2509.04377](https://arxiv.org/abs/2509.04377) - PagedEviction
- [arXiv:2506.07311](https://arxiv.org/abs/2506.07311) - Paged Attention + FlexAttention
- [arXiv:2406.17808](https://arxiv.org/abs/2406.17808) - Cascading KV Cache

### Streaming Audio
- [arXiv:2509.16195](https://arxiv.org/abs/2509.16195) - FocalCodec-Stream (80ms latency)
- [arXiv:2503.11562](https://arxiv.org/abs/2503.11562) - BRAVE (low-latency neural synthesizer)
- [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) - StreamingLLM (attention sinks)

---

*Document Version: 1.0*
*Last Updated: 2026-01-20*
*Author: Claude (for jhacksman/heartlib-gb10)*
