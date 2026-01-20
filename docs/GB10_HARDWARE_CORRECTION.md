# GB10 Hardware Correction & GPU Optimization Strategy

**CRITICAL UPDATE**: The original optimization plan misidentified the GB10 hardware. This document provides corrected specifications and GPU-optimized recommendations.

---

## 1. Actual GB10 (DGX Spark) Hardware Specifications

| Spec | Original Assumption (WRONG) | Actual DGX Spark (CORRECT) |
|------|----------------------------|---------------------------|
| CPU | 8-core ARM Cortex-A78 | 20-core ARM (10 X925 + 10 A725) |
| Memory | 16GB LPDDR5 | **128GB LPDDR5x unified** |
| GPU | None (CPU-only) | **NVIDIA Blackwell GPU with 5th Gen Tensor Cores** |
| AI Performance | N/A | **1 PFLOP FP4** |
| Power | <25W | ~140W TDP |

**Key Insight**: GB10 is NOT a CPU-only edge device. It's a Grace Blackwell superchip with unified memory architecture - heavy compute belongs on the GPU, not ARM NEON.

---

## 2. What Changes

### 2.1 Invalid Recommendations (Discard)

| Original Recommendation | Why It's Wrong |
|------------------------|----------------|
| ARM NEON SIMD optimization | GPU Tensor Cores are 100x faster for matmul |
| MNN Framework | MNN is for mobile ARM CPU; use TensorRT-LLM instead |
| "No Quantization" goal | FP4 on Blackwell gives 1 PFLOP with ~3.5x memory savings |
| 16GB memory constraints | 128GB unified memory fits entire pipeline at FP16 (~8GB) |
| Multi-core CPU load balancing | GPU handles all heavy compute |

### 2.2 Valid Recommendations (Keep)

| Original Recommendation | Why It's Still Valid |
|------------------------|---------------------|
| Speculative decoding (300M draft) | Works on GPU, proven 4-5x speedup |
| Flow matching step reduction (20→10) | Architecture-agnostic optimization |
| CFG batching | Already implemented, valid for GPU |
| Consistency distillation | Still the biggest win for flow matching |
| Lookahead/Jacobi decoding | GPU-accelerated parallel verification |

---

## 3. Corrected Framework Recommendations

### 3.1 Primary: TensorRT-LLM

**Why**: NVIDIA's official LLM inference engine, optimized for Blackwell

- Native FP4/FP8 quantization support
- Speculative decoding built-in
- Paged KV cache (vLLM-style)
- Multi-GPU support (future scaling)

**Integration Path**:
1. Export HeartMuLa to HuggingFace format
2. Convert to TensorRT-LLM engine with FP4 quantization
3. Use TensorRT-LLM Python API for inference
4. Keep HeartCodec on PyTorch with torch.compile()

### 3.2 Alternative: vLLM / SGLang

**Why**: Production-ready, easy integration

- PagedAttention for memory efficiency
- Continuous batching
- Speculative decoding support
- Active community, frequent updates

### 3.3 For HeartCodec (Flow Matching)

The codec transformer can use:
- **torch.compile()** with `mode="max-autotune"` for CUDA graphs
- **FlashAttention-3** (Blackwell optimized)
- **TensorRT** for CNN decoder (ScalarModel)

---

## 4. Corrected Quantization Strategy

### 4.1 NVFP4 for Blackwell (Recommended)

**Source**: [NVIDIA TensorRT Model Optimizer Documentation](https://docs.nvidia.com/deeplearning/tensorrt/model-optimizer/)

| Format | Memory | Throughput | Accuracy |
|--------|--------|------------|----------|
| FP16 | 8GB | Baseline | 100% |
| FP8 | 4GB | 2x | ~99.5% |
| **FP4** | **2GB** | **3-4x** | **~98%** |

**Why FP4 is right for GB10**:
- Blackwell Tensor Cores are optimized for FP4
- 1 PFLOP theoretical at FP4 vs 500 TFLOP at FP8
- HeartMuLa 3B fits in ~2GB at FP4 (vs 6GB FP16)
- Leaves headroom for KV cache and codec

### 4.2 Mixed Precision Strategy

```
┌─────────────────────────────────────────────────────────────┐
│              RECOMMENDED PRECISION ALLOCATION                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HeartMuLa 3B Backbone:     FP4 weights, FP8 activations    │
│  HeartMuLa 300M Decoder:    FP4 weights, FP8 activations    │
│  HeartCodec FlowMatching:   FP16 (quality-sensitive)        │
│  HeartCodec ScalarModel:    FP16 (small, quality-sensitive) │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Corrected Performance Estimates

### 5.1 Memory Budget (128GB Unified)

| Component | FP16 | FP4 (Recommended) |
|-----------|------|-------------------|
| HeartMuLa 3B | ~6GB | ~1.5GB |
| HeartMuLa 300M (draft) | ~0.6GB | ~0.15GB |
| HeartCodec FlowMatching | ~1GB | ~1GB (keep FP16) |
| HeartCodec ScalarModel | ~0.1GB | ~0.1GB (keep FP16) |
| KV Cache (8K context) | ~4GB | ~2GB |
| **Total** | **~12GB** | **~5GB** |

**Headroom**: 123GB available for batching, longer context, or multiple streams.

### 5.2 Expected Throughput

| Configuration | RTF Estimate | Notes |
|---------------|-------------|-------|
| Naive FP16 on Blackwell | 0.3-0.5 | Already faster than RTX 3090 |
| + FP4 quantization | 0.1-0.2 | 3x memory bandwidth improvement |
| + Speculative decoding | 0.03-0.05 | 4x token throughput |
| + Flow matching 10 steps | 0.02-0.03 | 2x codec speedup |
| + Consistency distillation | **0.01-0.02** | **50-100x faster than real-time** |

**Revised Target**: **0.02-0.05 RTF** (20-50x faster than real-time)

---

## 6. Corrected Implementation Plan

### Phase 1: GPU Baseline
- [ ] Run HeartMuLa on Blackwell with FP16
- [ ] Measure baseline throughput and memory
- [ ] Verify CUDA graphs work with torch.compile()

### Phase 2: Quantization
- [ ] Apply FP4 quantization via TensorRT Model Optimizer
- [ ] Benchmark accuracy on music generation quality
- [ ] Fine-tune if needed (likely unnecessary)

### Phase 3: Speculative Decoding
- [ ] Implement 300M draft model in TensorRT-LLM
- [ ] Or use vLLM's built-in speculative decoding
- [ ] Tune speculation length and acceptance threshold

### Phase 4: Flow Matching Optimization
- [ ] Reduce ODE steps to 10 (quality test)
- [ ] Apply torch.compile() to FlowMatching transformer
- [ ] Consider consistency distillation training

### Phase 5: End-to-End Optimization
- [ ] Profile full pipeline
- [ ] Optimize data transfer (unified memory helps)
- [ ] Batch processing for throughput

---

## 7. GPU-Specific Optimizations

### 7.1 Unified Memory Architecture

DGX Spark's unified memory eliminates CPU-GPU transfer overhead:

```
┌─────────────────────────────────────────────────────────────┐
│              UNIFIED MEMORY ADVANTAGE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Traditional GPU:                                            │
│    CPU RAM ─────PCIe────→ GPU VRAM ─────→ Compute           │
│              (bottleneck)                                    │
│                                                              │
│  DGX Spark (Grace Blackwell):                               │
│    Unified 128GB ──────────────────────→ Compute            │
│              (no transfer needed)                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implication**: Can keep model weights, KV cache, and activations in same memory space. No explicit memory management needed.

### 7.2 FlashAttention-3 for Blackwell

Use FlashAttention-3 which has Blackwell-specific optimizations:
- Asynchronous softmax
- FP8 accumulation
- Better occupancy on Blackwell SM

### 7.3 CUDA Graphs

For autoregressive generation, CUDA graphs eliminate kernel launch overhead:

```python
# Compile with CUDA graphs
model = torch.compile(model, mode="max-autotune", fullgraph=True)
```

---

## 8. Research That Remains Valid

All arxiv papers cited in the original document are valid and applicable to GPU inference:

### Speculative Decoding
- [arXiv:2410.13839](https://arxiv.org/abs/2410.13839) - Multi-token prediction (4-5x) ✓
- [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) - Medusa Heads ✓
- [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) - LayerSkip ✓

### Flow Matching
- Consistency distillation (20→1 steps) ✓
- Step reduction (20→10) ✓

### Parallel Decoding
- [arXiv:2402.02057](https://arxiv.org/abs/2402.02057) - Lookahead Decoding ✓
- [arXiv:2512.14681](https://arxiv.org/abs/2512.14681) - Jacobi Forcing ✓

### KV Cache (For Long Generation)
- [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) - KVSwap (less relevant with 128GB)
- [arXiv:2509.04377](https://arxiv.org/abs/2509.04377) - PagedEviction ✓

---

## 9. Summary of Changes

| Aspect | Original Plan | Corrected Plan |
|--------|--------------|----------------|
| Target Hardware | ARM CPU | Blackwell GPU |
| Framework | MNN | TensorRT-LLM / vLLM |
| Quantization | None | FP4 (NVFP4) |
| Memory Budget | 16GB | 128GB |
| Target RTF | 0.4-0.7 | **0.02-0.05** |
| Key Optimization | NEON SIMD | Tensor Cores + speculative decoding |

---

*Document Version: 1.0*
*Last Updated: 2026-01-20*
*Status: Correction to GB10_OPTIMIZATION_PLAN.md hardware assumptions*
