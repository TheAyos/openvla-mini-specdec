"""
optimize_minivla.py

Single-file script to optimize MiniVLA inference speed.

The goal is to make MiniVLA predict actions as fast as possible.
We test different optimization strategies and report the best one.

Usage:
    python -m experiments.specdec.optimize_minivla_v0_preStaticKV_15Hz
"""

import os
os.environ['PRISMATIC_DATA_ROOT'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import copy

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismatic.models.load import load_vla

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    checkpoint: str = "Stanford-ILIAD/minivla-libero90-prismatic"
    hf_token_path: str = ".hf_token"
    unnorm_key: str = "libero_90"
    num_iterations: int = 50
    warmup_iterations: int = 10
    image_size: int = 224


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_minivla(checkpoint: str, hf_token: str, device: str = "cuda"):
    """Load MiniVLA model."""
    print(f"Loading MiniVLA from {checkpoint}...")
    
    vla = load_vla(checkpoint, hf_token=hf_token, load_for_training=False)
    
    # Verify parameters are in float32 initially
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Parameter not in float32: {param.dtype}"
    
    # Cast to half precision
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(device)
    vla.eval()
    
    return vla


# ============================================================================
# Optimization Strategies
# ============================================================================

def apply_torch_compile_default(model):
    """Apply torch.compile with default mode (no CUDA graphs)."""
    try:
        model.llm_backbone.llm = torch.compile(
            model.llm_backbone.llm,
            mode="default",
            fullgraph=False,
            dynamic=True,  # Allow dynamic shapes for KV cache
        )
        print("  [OK] torch.compile(mode='default') applied to LLM")
        return True
    except Exception as e:
        print(f"  [FAIL] torch.compile failed: {e}")
        return False
    

def apply_torch_compile_default_sdpa_attn(model):
    """Apply torch.compile with default mode and sdpa attn (no CUDA graphs)."""
    try:
        model.llm_backbone.llm = torch.compile(
            model.llm_backbone.llm,
            mode="default",
            fullgraph=False,
            dynamic=True,
        )
        model.llm_backbone.set_attn_implementation("sdpa")
        print("  [OK] torch.compile(mode='default') applied to LLM and sdpa attn set")
        return True
    except Exception as e:
        print(f"  [FAIL] torch.compile failed: {e}")
        return False

def apply_torch_compile_no_cudagraph(model):
    """Apply torch.compile but disable CUDA graphs to avoid KV cache issues."""
    try:
        # Disable CUDA graphs in inductor
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraphs = False
        
        model.llm_backbone.llm = torch.compile(
            model.llm_backbone.llm,
            mode="max-autotune",
            fullgraph=False,
            dynamic=True,
            # options={"triton.cudagraphs": False}
        )
        # print("  [OK] torch.compile(mode='reduce-overhead', cudagraphs=False) applied")
        print("  [OK] torch.compile(mode='max-autotune', cudagraphs=False) applied")
        return True
    except Exception as e:
        print(f"  [FAIL] torch.compile failed: {e}")
        return False


# def apply_torch_compile_vision(model):
#     """Apply torch.compile to vision backbone only (safe, static shapes)."""
#     try:
#         # Vision backbone has static shapes, so CUDA graphs work well
#         model.vision_backbone.dino_featurizer = torch.compile(
#             model.vision_backbone.dino_featurizer,
#             mode="reduce-overhead",
#             fullgraph=True,
#         )
#         model.vision_backbone.siglip_featurizer = torch.compile(
#             model.vision_backbone.siglip_featurizer,
#             mode="reduce-overhead", 
#             fullgraph=True,
#         )
#         print("  [OK] torch.compile(mode='reduce-overhead') applied to vision backbone")
#         return True
#     except Exception as e:
#         print(f"  [FAIL] torch.compile on vision failed: {e}")
#         return False


# ============================================================================
# Benchmarking
# ============================================================================

def create_test_image(size: int = 224) -> Image.Image:
    """Create a random test image."""
    img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(img_array).convert("RGB")


def benchmark_model(
    model,
    image: Image.Image,
    instruction: str,
    unnorm_key: str,
    num_iterations: int,
    warmup_iterations: int,
    description: str = "Model",
) -> Dict[str, float]:
    """Benchmark model inference."""
    
    print(f"\n  Warming up {description} ({warmup_iterations} iterations)...")
    for i in range(warmup_iterations):
        with torch.inference_mode():
            _ = model.predict_action(image, instruction, unnorm_key=unnorm_key)
        if i == 0:
            print(f"    First warmup complete")
    torch.cuda.synchronize()
    print(f"    Warmup complete")
    
    print(f"  Benchmarking {description} ({num_iterations} iterations)...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times_ms = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start_event.record()
        
        with torch.inference_mode():
            action = model.predict_action(image, instruction, unnorm_key=unnorm_key)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times_ms.append(elapsed_ms)
        
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{num_iterations}, last: {elapsed_ms:.1f}ms")
    
    mean_ms = np.mean(times_ms)
    std_ms = np.std(times_ms)
    min_ms = np.min(times_ms)
    max_ms = np.max(times_ms)
    hz = 1000.0 / mean_ms
    
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "hz": hz,
        "times_ms": times_ms,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("MiniVLA Optimization Benchmark")
    print("=" * 70)
    
    cfg = Config()
    
    # Load HF token
    hf_token_path = Path(cfg.hf_token_path)
    if hf_token_path.exists():
        hf_token = hf_token_path.read_text().strip()
    else:
        hf_token = os.environ.get("HF_TOKEN", None)
    
    if not hf_token:
        print("ERROR: No HuggingFace token found!")
        return
    
    # Create test inputs
    test_image = create_test_image(cfg.image_size)
    test_instruction = "pick up the red block"
    
    results = {}
    
    # ========================================================================
    # Test 1: Baseline (no optimization) 9Hz
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Baseline (no optimization)")
    print("=" * 70)
    
    model = load_minivla(cfg.checkpoint, hf_token)
    
    results["baseline"] = benchmark_model(
        model, test_image, test_instruction, cfg.unnorm_key,
        cfg.num_iterations, cfg.warmup_iterations, "Baseline"
    )
    
    print(f"\n  RESULT: {results['baseline']['mean_ms']:.2f} ± {results['baseline']['std_ms']:.2f} ms")
    print(f"          {results['baseline']['hz']:.2f} Hz")
    
    del model
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Test 2: torch.compile with default mode 15Hz nice
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: torch.compile (default mode, dynamic shapes)")
    print("=" * 70)
    
    model = load_minivla(cfg.checkpoint, hf_token)
    
    if apply_torch_compile_default(model):
        results["compile_default"] = benchmark_model(
            model, test_image, test_instruction, cfg.unnorm_key,
            cfg.num_iterations, cfg.warmup_iterations, "Compiled (default)"
        )
        print(f"\n  RESULT: {results['compile_default']['mean_ms']:.2f} ± {results['compile_default']['std_ms']:.2f} ms")
        print(f"          {results['compile_default']['hz']:.2f} Hz")
    
    del model
    torch.cuda.empty_cache()
    # ========================================================================
    # Test 2bis: torch.compile with default mode sdpa attn
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 2bis: torch.compile (default mode, dynamic shapes, sdpa attn)")
    print("=" * 70)
    
    model = load_minivla(cfg.checkpoint, hf_token)
    
    if apply_torch_compile_default_sdpa_attn(model):
        results["compile_default_sdpa_attn"] = benchmark_model(
            model, test_image, test_instruction, cfg.unnorm_key,
            cfg.num_iterations, cfg.warmup_iterations, "Compiled (default sdpa attn)"
        )
        print(f"\n  RESULT: {results['compile_default_sdpa_attn']['mean_ms']:.2f} ± {results['compile_default_sdpa_attn']['std_ms']:.2f} ms")
        print(f"          {results['compile_default_sdpa_attn']['hz']:.2f} Hz")
    
    del model
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Test 3: torch.compile reduce-overhead but disable CUDA graphs
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: torch.compile (reduce-overhead, CUDA graphs disabled)")
    print("=" * 70)
    
    model = load_minivla(cfg.checkpoint, hf_token)
    
    if apply_torch_compile_no_cudagraph(model):
        results["compile_no_cudagraph"] = benchmark_model(
            model, test_image, test_instruction, cfg.unnorm_key,
            cfg.num_iterations, cfg.warmup_iterations, "Compiled (no cudagraph)"
        )
        print(f"\n  RESULT: {results['compile_no_cudagraph']['mean_ms']:.2f} ± {results['compile_no_cudagraph']['std_ms']:.2f} ms")
        print(f"          {results['compile_no_cudagraph']['hz']:.2f} Hz")
    
    del model
    torch.cuda.empty_cache()
    
    # # ========================================================================
    # # Test 4: torch.compile on vision backbone only
    # # ========================================================================
    # print("\n" + "=" * 70)
    # print("TEST 4: torch.compile (vision backbone only)")
    # print("=" * 70)
    
    # model = load_minivla(cfg.checkpoint, hf_token)
    
    # if apply_torch_compile_vision(model):
    #     results["compile_vision"] = benchmark_model(
    #         model, test_image, test_instruction, cfg.unnorm_key,
    #         cfg.num_iterations, cfg.warmup_iterations, "Compiled (vision)"
    #     )
    #     print(f"\n  RESULT: {results['compile_vision']['mean_ms']:.2f} ± {results['compile_vision']['std_ms']:.2f} ms")
    #     print(f"          {results['compile_vision']['hz']:.2f} Hz")
    
    # del model
    # torch.cuda.empty_cache()
    
    # # ========================================================================
    # # Test 5: Combined - vision compile + LLM compile (default)
    # # ========================================================================
    # print("\n" + "=" * 70)
    # print("TEST 5: torch.compile (vision + LLM default)")
    # print("=" * 70)
    
    # model = load_minivla(cfg.checkpoint, hf_token)
    
    # vision_ok = apply_torch_compile_vision(model)
    # llm_ok = apply_torch_compile_default(model)
    
    # if vision_ok or llm_ok:
    #     results["compile_combined"] = benchmark_model(
    #         model, test_image, test_instruction, cfg.unnorm_key,
    #         cfg.num_iterations, cfg.warmup_iterations, "Compiled (combined)"
    #     )
    #     print(f"\n  RESULT: {results['compile_combined']['mean_ms']:.2f} ± {results['compile_combined']['std_ms']:.2f} ms")
    #     print(f"          {results['compile_combined']['hz']:.2f} Hz")
    
    # del model
    # torch.cuda.empty_cache()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline_hz = results["baseline"]["hz"]
    baseline_ms = results["baseline"]["mean_ms"]
    
    print(f"\n{'Method':<45} {'Time (ms)':<20} {'Hz':<10} {'Speedup':<10}")
    print("-" * 85)
    
    for name, res in results.items():
        speedup = res["hz"] / baseline_hz
        time_str = f"{res['mean_ms']:.2f} ± {res['std_ms']:.2f}"
        print(f"{name:<45} {time_str:<20} {res['hz']:<10.2f} {speedup:<10.2f}x")
    
    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]["hz"])
    best_hz = results[best_name]["hz"]
    best_speedup = best_hz / baseline_hz
    
    print(f"\nBEST: {best_name} at {best_hz:.2f} Hz ({best_speedup:.2f}x speedup)")
    print("=" * 70)


if __name__ == "__main__":
    main()
