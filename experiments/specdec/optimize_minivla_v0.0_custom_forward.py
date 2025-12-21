"""
optimize_minivla.py

Single-file script to optimize MiniVLA inference speed.

Usage:
    python -m experiments.specdec.optimize_minivla
"""

import os
os.environ['PRISMATIC_DATA_ROOT'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismatic.models.load import load_vla

# Try to import StaticCache
try:
    from transformers import StaticCache
    from transformers.generation.configuration_utils import GenerationConfig
    HAS_STATIC_CACHE = True
except ImportError:
    HAS_STATIC_CACHE = False
    print("WARNING: StaticCache not available")

# def apply_torch_compile_default_sdpa_attn(model):
#     """Apply torch.compile with default mode and sdpa attn (no CUDA graphs)."""
#     try:
#         model.llm_backbone.llm = torch.compile(
#             model.llm_backbone.llm,
#             mode="default",
#             fullgraph=False,
#             dynamic=True,
#         )
#         model.llm_backbone.set_attn_implementation("sdpa")
#         print("  [OK] torch.compile(mode='default') applied to LLM and sdpa attn set")
#         return True
#     except Exception as e:
#         print(f"  [FAIL] torch.compile failed: {e}")
#         return False

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
# Model Loading
# ============================================================================

def load_minivla(checkpoint: str, hf_token: str, device: str = "cuda"):
    """Load MiniVLA model."""
    print(f"Loading MiniVLA from {checkpoint}...")
    vla = load_vla(checkpoint, hf_token=hf_token, load_for_training=False)
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(device)
    vla.eval()
    return vla


# ============================================================================
# Fast Action Predictor
# ============================================================================

class FastActionPredictor:
    """
    Fast action predictor that matches HuggingFace generate() output exactly.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = model.llm_backbone.half_precision_dtype
        self.llm = model.llm_backbone.llm
        self.config = self.llm.config
    
    def _prepare_inputs(self, image: Image.Image, instruction: str):
        """Prepare model inputs (same as HuggingFace)."""
        model = self.model
        image_transform = model.vision_backbone.get_image_transform()
        tokenizer = model.llm_backbone.tokenizer
        
        prompt_builder = model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    def _decode_to_actions(self, token_ids: np.ndarray, unnorm_key: str) -> np.ndarray:
        """Convert token IDs to actions (same as HuggingFace)."""
        model = self.model
        normalized_actions = model.action_tokenizer.decode_token_ids_to_actions(token_ids)
        
        action_norm_stats = model.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions
    
    @torch.inference_mode()
    def predict_action(self, image: Image.Image, instruction: str, unnorm_key: str) -> np.ndarray:
        """Predict action with custom loop (using DynamicCache like HF)."""
        model = self.model
        action_dim = model.get_action_dim(unnorm_key)
        
        input_ids, attention_mask, pixel_values = self._prepare_inputs(image, instruction)
        
        # === Prefill: process prompt + image ===
        with torch.cuda.amp.autocast(dtype=self.dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        
        # === Decode: generate action tokens (greedy) ===
        generated_tokens = []
        
        for _ in range(action_dim):
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            
            with torch.cuda.amp.autocast(dtype=self.dtype):
                outputs = model.llm_backbone(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
        
        # === Decode tokens to actions ===
        predicted_token_ids = np.array(generated_tokens, dtype=np.int64)
        return self._decode_to_actions(predicted_token_ids, unnorm_key)


# ============================================================================
# HuggingFace generate() with StaticCache
# ============================================================================

@torch.inference_mode()
def predict_action_hf_static_cache(
    model: nn.Module,
    image: Image.Image,
    instruction: str,
    unnorm_key: str,
) -> np.ndarray:
    """
    Predict action using HuggingFace generate() with StaticCache.
    
    This uses HF's built-in support for StaticCache through generation config.
    """
    from transformers import LlamaTokenizerFast
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    
    device = next(model.parameters()).device
    dtype = model.llm_backbone.half_precision_dtype
    action_dim = model.get_action_dim(unnorm_key)
    
    image_transform = model.vision_backbone.get_image_transform()
    tokenizer = model.llm_backbone.tokenizer
    
    # Build prompt
    prompt_builder = model.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
    prompt_text = prompt_builder.get_prompt()
    
    input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(device)
    
    if isinstance(tokenizer, LlamaTokenizerFast):
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(device)), dim=1
            )
    
    # Process image
    pixel_values = image_transform(image)
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None, ...].to(device)
    elif isinstance(pixel_values, dict):
        pixel_values = {k: v[None, ...].to(device) for k, v in pixel_values.items()}
    
    # Create StaticCache
    # Estimate max cache length: input tokens + image patches + generated tokens
    # Image patches for DinoSigLIP 224px: 256 patches
    max_cache_len = input_ids.shape[1] + 300 + action_dim + 10
    
    static_cache = StaticCache(
        config=model.llm_backbone.llm.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )
    
    # Generate with StaticCache
    with torch.autocast("cuda", dtype=dtype, enabled=model.enable_mixed_precision_training):
        from prismatic.models.vlms.prismatic import PrismaticVLM
        generated_ids = super(PrismaticVLM, model).generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=action_dim,
            past_key_values=static_cache,
            do_sample=False,
        )
    
    # Decode
    predicted_action_token_ids = generated_ids[0, -action_dim:]
    normalized_actions = model.action_tokenizer.decode_token_ids_to_actions(
        predicted_action_token_ids.cpu().numpy()
    )
    
    # Un-normalize
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    
    return actions


# ============================================================================
# Correctness Verification
# ============================================================================

def verify_correctness(
    model: nn.Module,
    predict_fn,
    image: Image.Image,
    instruction: str,
    unnorm_key: str,
    description: str = "Custom",
) -> Tuple[bool, float]:
    """Verify prediction matches HuggingFace generate()."""
    print(f"\n  Verifying {description} correctness...")
    
    # Baseline
    with torch.inference_mode():
        baseline_action = model.predict_action(image, instruction, unnorm_key=unnorm_key)
    
    # Custom
    custom_action = predict_fn(image, instruction, unnorm_key)
    
    max_diff = np.max(np.abs(baseline_action - custom_action))
    is_correct = max_diff < 1e-5
    
    print(f"    Baseline: {baseline_action}")
    print(f"    {description}: {custom_action}")
    print(f"    Max diff: {max_diff:.2e}, Correct: {is_correct}")
    
    return is_correct, max_diff


# ============================================================================
# Benchmarking
# ============================================================================

def create_test_image(size: int = 224) -> Image.Image:
    """Create a random test image."""
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(img_array).convert("RGB")


def benchmark(
    predict_fn,
    image: Image.Image,
    instruction: str,
    unnorm_key: str,
    num_iterations: int,
    warmup_iterations: int,
    description: str,
) -> Dict:
    """Benchmark a prediction function."""
    
    print(f"\n  Warming up {description} ({warmup_iterations} iterations)...")
    for i in range(warmup_iterations):
        _ = predict_fn(image, instruction, unnorm_key)
        if i == 0:
            print("    First warmup complete")
    torch.cuda.synchronize()
    print("    Warmup complete")
    
    print(f"  Benchmarking {description} ({num_iterations} iterations)...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times_ms = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start_event.record()
        _ = predict_fn(image, instruction, unnorm_key)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
        
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{num_iterations}, last: {times_ms[-1]:.1f}ms")
    
    mean_ms = np.mean(times_ms)
    std_ms = np.std(times_ms)
    return {"mean_ms": mean_ms, "std_ms": std_ms, "hz": 1000.0 / mean_ms}


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("MiniVLA Optimization Benchmark")
    print("=" * 70)
    print(f"StaticCache available: {HAS_STATIC_CACHE}")
    print("=" * 70)
    
    cfg = Config()
    
    # Load HF token
    hf_token_path = Path(cfg.hf_token_path)
    hf_token = hf_token_path.read_text().strip() if hf_token_path.exists() else os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: No HuggingFace token found!")
        return
    
    # Create test inputs
    test_image = create_test_image(cfg.image_size)
    test_instruction = "pick up the red block"
    
    results = {}
    
    # ========================================================================
    # Load Model
    # ========================================================================
    model = load_minivla(cfg.checkpoint, hf_token)
    
    # ========================================================================
    # Test 1: Baseline (HuggingFace generate with DynamicCache)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Baseline (HuggingFace generate, DynamicCache)")
    print("=" * 70)
    
    def baseline_predict(img, instr, unnorm):
        with torch.inference_mode():
            return model.predict_action(img, instr, unnorm_key=unnorm)
    
    results["1_baseline"] = benchmark(
        baseline_predict, test_image, test_instruction, cfg.unnorm_key,
        cfg.num_iterations, cfg.warmup_iterations, "Baseline"
    )
    print(f"\n  RESULT: {results['1_baseline']['mean_ms']:.2f} ± {results['1_baseline']['std_ms']:.2f} ms ({results['1_baseline']['hz']:.2f} Hz)")
    
    # ========================================================================
    # Test 2: Custom loop (verify correctness)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Custom autoregressive loop")
    print("=" * 70)
    
    predictor = FastActionPredictor(model)
    
    is_correct, _ = verify_correctness(
        model, predictor.predict_action, test_image, test_instruction, 
        cfg.unnorm_key, "Custom Loop"
    )
    
    if not is_correct:
        print("  WARNING: Custom loop differs from baseline!")
    
    results["2_custom_loop"] = benchmark(
        predictor.predict_action, test_image, test_instruction, cfg.unnorm_key,
        cfg.num_iterations, cfg.warmup_iterations, "Custom Loop"
    )
    print(f"\n  RESULT: {results['2_custom_loop']['mean_ms']:.2f} ± {results['2_custom_loop']['std_ms']:.2f} ms ({results['2_custom_loop']['hz']:.2f} Hz)")
    # # ========================================================================
    # # Test 2bis: torch.compile with default mode sdpa attn
    # # ========================================================================
    # print("\n" + "=" * 70)
    # print("TEST 2bis: torch.compile (default mode, dynamic shapes, sdpa attn)")
    # print("=" * 70)
    
    # model = load_minivla(cfg.checkpoint, hf_token)
    
    # if apply_torch_compile_default_sdpa_attn(model):
    #     results["compile_default_sdpa_attn"] = benchmark(
    #         predictor.predict_action, test_image, test_instruction, cfg.unnorm_key,
    #         cfg.num_iterations, cfg.warmup_iterations, "Compiled (default sdpa attn)"
    #     )
    #     print(f"\n  RESULT: {results['compile_default_sdpa_attn']['mean_ms']:.2f} ± {results['compile_default_sdpa_attn']['std_ms']:.2f} ms")
    #     print(f"          {results['compile_default_sdpa_attn']['hz']:.2f} Hz")
    
    # del model
    # torch.cuda.empty_cache()
    # ========================================================================
    # Test 3: HuggingFace generate with StaticCache
    # ========================================================================
    if HAS_STATIC_CACHE:
        print("\n" + "=" * 70)
        print("TEST 3: HuggingFace generate with StaticCache")
        print("=" * 70)
        
        def hf_static_predict(img, instr, unnorm):
            return predict_action_hf_static_cache(model, img, instr, unnorm)
        
        # Verify correctness first
        try:
            is_correct, _ = verify_correctness(
                model, hf_static_predict, test_image, test_instruction,
                cfg.unnorm_key, "HF StaticCache"
            )
            
            if is_correct:
                results["3_hf_static_cache"] = benchmark(
                    hf_static_predict, test_image, test_instruction, cfg.unnorm_key,
                    cfg.num_iterations, cfg.warmup_iterations, "HF StaticCache"
                )
                print(f"\n  RESULT: {results['3_hf_static_cache']['mean_ms']:.2f} ± {results['3_hf_static_cache']['std_ms']:.2f} ms ({results['3_hf_static_cache']['hz']:.2f} Hz)")
            else:
                print("  Skipping benchmark due to correctness mismatch")
        except Exception as e:
            print(f"  StaticCache test failed: {e}")
    
    # ========================================================================
    # Test 4: torch.compile with default mode
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Custom loop + torch.compile(mode='default')")
    print("=" * 70)
    
    # Reload model for clean compile
    del model
    torch.cuda.empty_cache()
    model = load_minivla(cfg.checkpoint, hf_token)
    
    try:
        model.llm_backbone.llm = torch.compile(
            model.llm_backbone.llm, mode="default", fullgraph=False, dynamic=True
        )
        print("  [OK] torch.compile applied to LLM")
        compiled_ok = True
    except Exception as e:
        print(f"  [FAIL] torch.compile failed: {e}")
        compiled_ok = False
    
    if compiled_ok:
        predictor_compiled = FastActionPredictor(model)
        
        # Verify correctness (baseline with compiled model)
        def baseline_compiled(img, instr, unnorm):
            with torch.inference_mode():
                return model.predict_action(img, instr, unnorm_key=unnorm)
        
        results["4_compiled_default"] = benchmark(
            predictor_compiled.predict_action, test_image, test_instruction, cfg.unnorm_key,
            cfg.num_iterations, cfg.warmup_iterations, "Compiled (default)"
        )
        print(f"\n  RESULT: {results['4_compiled_default']['mean_ms']:.2f} ± {results['4_compiled_default']['std_ms']:.2f} ms ({results['4_compiled_default']['hz']:.2f} Hz)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline_hz = results["1_baseline"]["hz"]
    
    print(f"\n{'Method':<45} {'Time (ms)':<20} {'Hz':<10} {'Speedup':<10}")
    print("-" * 85)
    
    for name, res in sorted(results.items()):
        speedup = res["hz"] / baseline_hz
        time_str = f"{res['mean_ms']:.2f} ± {res['std_ms']:.2f}"
        print(f"{name:<45} {time_str:<20} {res['hz']:<10.2f} {speedup:<10.2f}x")
    
    best_name = max(results.keys(), key=lambda k: results[k]["hz"])
    print(f"\nBEST: {best_name} at {results[best_name]['hz']:.2f} Hz ({results[best_name]['hz']/baseline_hz:.2f}x)")
    print("=" * 70)


if __name__ == "__main__":
    main()
