#!/usr/bin/env python3
"""
Benchmark script for HeartLib on GB10.

Measures:
- Flow matching inference time
- Real-time factor (RTF) 
- Memory usage
- Throughput

Usage:
    python scripts/benchmark.py [--num_steps N] [--duration D] [--warmup W] [--runs R]
"""

import argparse
import time
import torch
import gc
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class BenchmarkResult:
    num_steps: int
    duration_sec: float
    avg_inference_time_ms: float
    std_inference_time_ms: float
    rtf: float  # Real-time factor (inference_time / audio_duration)
    peak_memory_gb: float
    throughput_sec_per_sec: float  # seconds of audio per second of compute
    device: str
    torch_version: str


def benchmark_flow_matching(
    num_steps: int = 10,
    duration: float = 29.76,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    device: str = "cuda",
) -> BenchmarkResult:
    """Benchmark the flow matching detokenization."""
    
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    from heartlib.heartcodec.configuration_heartcodec import HeartCodecConfig
    
    print(f"Benchmarking flow matching with num_steps={num_steps}, duration={duration}s")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    
    # Create model with default config
    config = HeartCodecConfig()
    model = HeartCodec(config).to(device).eval()
    
    # Create synthetic input codes
    # codes shape: [num_quantizers, time_frames]
    # time_frames ~= duration * 12.5 (from the code)
    time_frames = int(duration * 12.5)
    codes = torch.randint(0, config.codebook_size, (config.num_quantizers, time_frames), device=device)
    
    print(f"Input codes shape: {codes.shape}")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Warmup
    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.detokenize(
                codes,
                duration=duration,
                num_steps=num_steps,
                disable_progress=True,
                device=device
            )
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({benchmark_runs} runs)...")
    times = []
    for i in range(benchmark_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            output = model.detokenize(
                codes,
                duration=duration,
                num_steps=num_steps,
                disable_progress=True,
                device=device
            )
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        inference_time = (end - start) * 1000  # ms
        times.append(inference_time)
        print(f"  Run {i+1}: {inference_time:.2f}ms")
    
    # Calculate stats
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # RTF = inference_time / audio_duration
    rtf = (avg_time / 1000) / duration
    
    # Throughput = audio_duration / inference_time
    throughput = duration / (avg_time / 1000)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    result = BenchmarkResult(
        num_steps=num_steps,
        duration_sec=duration,
        avg_inference_time_ms=avg_time,
        std_inference_time_ms=std_time,
        rtf=rtf,
        peak_memory_gb=peak_memory,
        throughput_sec_per_sec=throughput,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
    )
    
    return result


def print_result(result: BenchmarkResult):
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Device: {result.device}")
    print(f"PyTorch: {result.torch_version}")
    print(f"Flow steps: {result.num_steps}")
    print(f"Audio duration: {result.duration_sec:.2f}s")
    print("-" * 60)
    print(f"Avg inference time: {result.avg_inference_time_ms:.2f} ± {result.std_inference_time_ms:.2f} ms")
    print(f"Real-time factor (RTF): {result.rtf:.4f}")
    print(f"Throughput: {result.throughput_sec_per_sec:.2f}x realtime")
    print(f"Peak memory: {result.peak_memory_gb:.2f} GB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark HeartLib flow matching")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of flow steps")
    parser.add_argument("--duration", type=float, default=29.76, help="Audio duration in seconds")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    args = parser.parse_args()
    
    result = benchmark_flow_matching(
        num_steps=args.num_steps,
        duration=args.duration,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
    )
    
    print_result(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
