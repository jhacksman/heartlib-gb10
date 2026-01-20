# HeartLib GB10 Optimization Plan

## Executive Summary

This document outlines a comprehensive optimization strategy for running HeartMuLa (a 3B parameter music generation model) on the GB10 (NVIDIA Grace Blackwell Superchip). The GB10 features a Blackwell GPU with 6,144 CUDA cores and 128GB unified memory, enabling GPU-accelerated inference with optional FP4 quantization for maximum throughput.

**Primary optimization vectors**: speculative decoding, flow matching step reduction, and GPU Tensor Core utilization.

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

**GB10 Target**: Achieve ≤0.1 RTF (10x+ faster than real-time) using GPU Tensor Cores with optional FP4 quantization.

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

## 4. GB10 GPU Optimizations

### 4.1 Blackwell Tensor Core Utilization

The GB10's Blackwell GPU has 5th Generation Tensor Cores optimized for transformer inference.

#### 4.1.1 Precision Strategy

| Precision | Memory | Throughput | Use Case |
|-----------|--------|------------|----------|
| FP16 | ~8GB | Baseline | Quality-sensitive components |
| FP8 | ~4GB | 2x | HeartMuLa backbone |
| **FP4** | **~2GB** | **3-4x** | **Maximum throughput** |

**Recommendation**: Use FP4 for HeartMuLa LM components, FP16 for HeartCodec (quality-sensitive audio).

#### 4.1.2 FlashAttention-3

Blackwell supports FlashAttention-3 with:
- Asynchronous softmax computation
- FP8 accumulation
- Improved SM occupancy

Enable via: `torch.backends.cuda.enable_flash_sdp(True)`

### 4.2 Unified Memory Architecture

GB10's 128GB unified memory eliminates CPU-GPU transfer overhead:

```
┌─────────────────────────────────────────────────────────────┐
│              GB10 UNIFIED MEMORY ADVANTAGE                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Traditional GPU:                                            │
│    CPU RAM ─────PCIe────→ GPU VRAM ─────→ Compute           │
│              (bottleneck)                                    │
│                                                              │
│  GB10 (Grace Blackwell):                                    │
│    Unified 128GB ──────────────────────→ Compute            │
│              (no transfer needed)                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Memory Budget** (HeartMuLa pipeline at FP4):
- HeartMuLa 3B: ~1.5GB
- HeartMuLa 300M decoder: ~0.15GB
- HeartCodec (FP16): ~1.1GB
- KV Cache (8K context): ~2GB
- **Total: ~5GB** (123GB headroom)

### 4.3 CUDA Graphs for Autoregressive Generation

Eliminate kernel launch overhead in the generation loop:

```python
# Compile with CUDA graphs for autoregressive decoding
model = torch.compile(model, mode="max-autotune", fullgraph=True)
```

This is critical for HeartMuLa's frame-by-frame generation where kernel launch overhead dominates.

---

## 5. Framework Recommendations

### 5.1 Primary: TensorRT-LLM

**Why**: NVIDIA's official LLM inference engine, optimized for Blackwell

- Native FP4/FP8 quantization via TensorRT Model Optimizer
- Built-in speculative decoding support
- Paged KV cache (vLLM-style memory efficiency)
- CUDA graph integration

**Integration Path**:
1. Export HeartMuLa to HuggingFace format
2. Convert to TensorRT-LLM engine with FP4 quantization
3. Use TensorRT-LLM Python API for inference
4. Keep HeartCodec on PyTorch with `torch.compile()`

### 5.2 Alternative: vLLM / SGLang

**Why**: Production-ready, easy integration

- PagedAttention for memory efficiency
- Continuous batching for throughput
- Speculative decoding support
- Active community, frequent updates

### 5.3 For HeartCodec (Flow Matching)

The codec transformer should use:
- **torch.compile()** with `mode="max-autotune"` for CUDA graphs
- **FlashAttention-3** (Blackwell optimized)
- **TensorRT** for ScalarModel CNN decoder (optional)

---

## 6. Implementation Phases

### Phase 1: GPU Baseline
- [ ] Run HeartMuLa on GB10 with FP16
- [ ] Measure baseline inference time and memory
- [ ] Profile hotspots with `torch.profiler`
- [ ] Verify CUDA/FlashAttention compatibility

### Phase 2: Quick Wins
- [ ] Reduce flow matching steps to 10 (quality test)
- [ ] Enable `torch.compile()` with `mode="max-autotune"`
- [ ] Ensure CFG batching is used everywhere
- [ ] Enable FlashAttention-3

### Phase 3: Quantization
- [ ] Apply FP4 quantization via TensorRT Model Optimizer
- [ ] Benchmark accuracy on music generation quality
- [ ] A/B test FP4 vs FP16 output quality

### Phase 4: Speculative Decoding
- [ ] Implement draft model approach using 300M decoder
- [ ] Add verification logic to 3B backbone
- [ ] Tune speculation length and acceptance threshold
- [ ] Measure token acceptance rate

### Phase 5: Parallel Flow Matching
- [ ] Implement segment batching for long audio
- [ ] Add overlap-add stitching
- [ ] Test audio quality at segment boundaries

### Phase 6: Integration & Testing
- [ ] End-to-end pipeline testing
- [ ] Quality evaluation (listening tests)
- [ ] Final performance benchmarks
- [ ] Documentation

---

## 7. Expected Performance Gains

| Optimization | Individual Gain | Cumulative RTF |
|--------------|-----------------|----------------|
| Baseline (RTX 3090 FP16) | 1.0x | 1.0 |
| GB10 Blackwell (FP16) | ~1.2-1.5x | 0.7-0.8 |
| + FP4 quantization | 2-3x | 0.25-0.4 |
| + Reduce flow steps (20→10) | 2.0x | 0.12-0.2 |
| + Speculative decoding (4x) | 3.0-4.0x | 0.03-0.07 |
| + CUDA graphs | 1.2-1.5x | 0.02-0.05 |

**Target**: Achieve **0.02-0.1 RTF** on GB10 (10-50x faster than real-time)

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
| Quality degradation from FP4 quantization | Low | Medium | Keep HeartCodec at FP16, calibration dataset |
| Speculative decoding low acceptance rate | Low | Medium | Tune temperature, fallback to AR |
| TensorRT-LLM conversion issues | Medium | Medium | Use vLLM as fallback |
| Segment boundary artifacts | Medium | Medium | Overlap-add with cross-fade |

---

## 10. Success Criteria

1. **Performance**: Achieve ≤0.1 RTF on GB10 for 3-minute songs (10x+ faster than real-time)
2. **Quality**: No perceptible audio quality degradation (blind listening test)
3. **Memory**: Peak usage ≤16GB (leaving 112GB headroom for batching/longer context)
4. **Stability**: 100 consecutive generations without crash
5. **Latency**: Time-to-first-audio ≤3 seconds

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

## Appendix B: GB10 Hardware Specs

**Source**: [NVIDIA DGX Spark Hardware Documentation](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

The GB10 is the **NVIDIA Grace Blackwell Superchip** powering the DGX Spark desktop AI system.

| Component | Specification |
|-----------|---------------|
| **Chip** | GB10 Grace Blackwell Superchip |
| **CPU** | 20-core ARM (10 Cortex-X925 @ 4GHz + 10 Cortex-A725 @ 2.8GHz) |
| **GPU** | NVIDIA Blackwell Architecture, 6,144 CUDA cores, 5th Gen Tensor Cores |
| **Memory** | 128GB LPDDR5x unified, 273 GB/s bandwidth |
| **AI Performance** | 1 PFLOP FP4 sparse, 1,000 TOPS inference |
| **Storage** | 1TB or 4TB NVMe M.2 |
| **Power** | 140W TDP (GB10 SOC), 240W system |
| **Networking** | 10GbE, ConnectX-7 (2x QSFP 200Gb/s aggregate), WiFi 7 |

**Key Implications for Optimization**:
- **GPU-first**: Heavy compute should target Blackwell Tensor Cores, not ARM CPU
- **Unified memory**: 128GB shared between CPU/GPU eliminates transfer overhead
- **FP4 optimal**: Blackwell achieves 1 PFLOP at FP4 precision with sparsity
- **Model capacity**: Can run 200B parameter models locally (405B with two units linked)

---

*Document Version: 2.0*
*Last Updated: 2026-01-20*
*Author: Claude (for jhacksman/heartlib-gb10)*
*Hardware verified from: [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)*
