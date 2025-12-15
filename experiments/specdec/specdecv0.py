# %%
"""
Benchmarks inference throughput on a single LIBERO task observation
"""

import os
os.environ['PRISMATIC_DATA_ROOT'] = ''
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import torch

import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# sys.path.insert(0, str(Path().resolve().parents[1]))
sys.path.append("../..")
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import get_processor, get_vla, get_prismatic_vla
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

assert torch.cuda.is_available(), "ERROR: CUDA not available!"

os.system("nvidia-smi")

# %% [markdown]
# ### config

# %%
@dataclass
class BenchmarkConfig:
    # fmt: off
    # target_checkpoint: Union[str, Path] = "/pub/scratch/aagouzoul/ovla/openvla-mini/ft_experiments_logs/openvla-7b+libero_90_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug+libero_90_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    target_checkpoint: Union[str, Path] = "/pub/scratch/aagouzoul/ovla/openvla-mini/ft_experiments_logs/openvla-7b+libero_90_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug+libero_90_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug_step-75000_l1-loss-0.0012_tokacc-0.955"
    draft_checkpoint: Union[str, Path] = "Stanford-ILIAD/minivla-libero90-prismatic"
    hf_token: str = Path("/pub/scratch/aagouzoul/ovla/openvla-mini/.hf_token")
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    
    # Speculative decoding parameters
    gamma: int = 7
    temperature: float = 0.0
    
    # Benchmark parameters
    task_suite_name: str = "libero_90"
    task_id: int = 0
    num_iterations: int = 1
    warmup_iterations: int = 5
    seed: int = 42
    
    # fmt: on
    
cfg = BenchmarkConfig()

print("=" * 80)
print("Speculative Decoding Throughput Benchmark")
print("=" * 80)
print(f"Target: {cfg.target_checkpoint}")
print(f"Draft: {cfg.draft_checkpoint}")
print(f"Gamma: {cfg.gamma}")
print(f"Iterations: {cfg.num_iterations}")
print("=" * 80)

set_seed_everywhere(cfg.seed)

# Create config objects for model loading
class TargetConfig:
    def __init__(self, c):
        self.pretrained_checkpoint = c.target_checkpoint
        self.load_in_8bit = c.load_in_8bit
        self.load_in_4bit = c.load_in_4bit
        self.hf_token = c.hf_token

class DraftConfig:
    def __init__(self, c):
        self.pretrained_checkpoint = c.draft_checkpoint
        self.model_family = "prismatic"
        self.hf_token = c.hf_token
        self.center_crop = c.center_crop


# %% [markdown]
# ### load models
# 

# %%
print("\n[1/4] Loading TARGET model (OpenVLA)...")
target_cfg = TargetConfig(cfg)
target_model = get_vla(target_cfg)
target_processor = get_processor(target_cfg)

print("\n[2/4] Loading DRAFT model (MiniVLA)...")
draft_cfg = DraftConfig(cfg)
draft_model = get_prismatic_vla(draft_cfg)

# Set unnorm key
unnorm_key_target = cfg.task_suite_name
if unnorm_key_target not in target_model.norm_stats:
    if f"{unnorm_key_target}_no_noops" in target_model.norm_stats:
        unnorm_key_target = f"{unnorm_key_target}_no_noops"
    elif f"{unnorm_key_target.replace('_no_noops', '')}" in target_model.norm_stats:
        unnorm_key_target = f"{unnorm_key_target}"
    else:
        unnorm_key_target = list(target_model.norm_stats.keys())[0]
        
unnorm_key_draft = cfg.task_suite_name
if unnorm_key_draft not in draft_model.norm_stats:
    if f"{unnorm_key_draft}_no_noops" in draft_model.norm_stats:
        unnorm_key_draft = f"{unnorm_key_draft}_no_noops"
    elif f"{unnorm_key_draft.replace('_no_noops', '')}" in draft_model.norm_stats:
        unnorm_key_draft = f"{unnorm_key_draft}"
    else:
        unnorm_key_draft = list(draft_model.norm_stats.keys())[0]

print(f"Using unnorm_keys: {unnorm_key_target} (target), {unnorm_key_draft} (draft)")

# %%
# Load LIBERO task and get observation
print(f"\n[4/4] Loading LIBERO task: {cfg.task_suite_name} (task {cfg.task_id})...")
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[cfg.task_suite_name]()
task = task_suite.get_task(cfg.task_id)
env, task_description = get_libero_env(task, "openvla", resolution=224)

initial_states = task_suite.get_task_init_states(cfg.task_id)
env.reset()
obs = env.set_init_state(initial_states[0])

# Prepare observation
img = get_libero_image(obs, 224)
observation = {
    "full_image": img,
    "state": np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ),
}

print(f"Task: {task_description}")
print(f"Image shape: {img.shape}")


print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"call params: get_vla_action(target_model..., target_processor..., cfg.target_checkpoint={cfg.target_checkpoint}, observation=..., task_description={task_description}, unnorm_key={unnorm_key_target}, center_crop={cfg.center_crop})")
print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"call params: get_prismatic_vla_action(draft_model..., processor=None, cfg.draft_checkpoint={cfg.draft_checkpoint}, observation..., task_description={task_description}, unnorm_key={unnorm_key_draft}, center_crop={cfg.center_crop})")

def run_target_inference():
    from experiments.robot.openvla_utils import get_vla_action
    return get_vla_action(
        target_model,
        target_processor,
        str(cfg.target_checkpoint),
        observation,
        task_description,
        unnorm_key_target,
        center_crop=cfg.center_crop,
    )

def run_draft_inference():
    from experiments.robot.openvla_utils import get_prismatic_vla_action    
    return get_prismatic_vla_action(
        draft_model,
        None,
        str(cfg.draft_checkpoint),
        observation,
        task_description,
        unnorm_key_draft,
        center_crop=cfg.center_crop,
    )

def timed_cuda(fn):
    """Time a function using CUDA events for accurate GPU timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000  # Convert to seconds

print(f"\nRunning warmup ({cfg.warmup_iterations} iterations each)...")

print("  Warming up TARGET...")
for _ in range(cfg.warmup_iterations):
    run_target_inference()
    torch.cuda.synchronize()

print("  Warming up DRAFT...")
for _ in range(cfg.warmup_iterations):
    run_draft_inference()
    torch.cuda.synchronize()
    
# Benchmark TARGET
print(f"\nBenchmarking TARGET ({cfg.num_iterations} iterations)...")
target_times = []
for i in range(cfg.num_iterations):
    result, dt = timed_cuda(run_target_inference)
    target_times.append(dt)
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{cfg.num_iterations}, last: {dt*1000:.1f}ms")

# Benchmark DRAFT
print(f"\nBenchmarking DRAFT ({cfg.num_iterations} iterations)...")
draft_times = []
for i in range(cfg.num_iterations):
    result, dt = timed_cuda(run_draft_inference)
    draft_times.append(dt)
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{cfg.num_iterations}, last: {dt*1000:.1f}ms")


# %%
# %load vla_speculative_decoding.py
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.cache_utils import DynamicCache

# ============================================================================
# Cache utilities for KV cache pruning
# ============================================================================
def prune_cache(
    cache: Union[Tuple[Tuple[torch.Tensor, torch.Tensor]], DynamicCache, None],
    num_tokens_to_discard: int,
) -> Union[Tuple[Tuple[torch.Tensor, torch.Tensor]], DynamicCache, None]:
    """Prune the KV cache by removing tokens from the end."""
    if cache is None or num_tokens_to_discard <= 0:
        return cache
    
    if isinstance(cache, DynamicCache):
        for layer in range(len(cache)):
            cache.key_cache[layer] = cache.key_cache[layer][:, :, :-num_tokens_to_discard, :]
            cache.value_cache[layer] = cache.value_cache[layer][:, :, :-num_tokens_to_discard, :]
        cache._seen_tokens -= num_tokens_to_discard
        return cache
    
    elif isinstance(cache, tuple):
        new_cache = []
        for layer_cache in cache:
            if layer_cache is None:
                new_cache.append(None)
                continue
            layer = []
            for tensor in layer_cache:
                new_tensor = tensor[:, :, :-num_tokens_to_discard, :]
                layer.append(new_tensor)
            new_cache.append(tuple(layer))
        return tuple(new_cache)
    
    else:
        raise ValueError(f"Unsupported cache type: {type(cache)}")

# ============================================================================
# Image preprocessing utilities
# ============================================================================
def apply_center_crop(im: np.ndarray, t_h: int, t_w: int) -> np.ndarray:
    """Center crop an image to target dimensions."""
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]

def prepare_image(full_image: Union[np.ndarray, List[np.ndarray]], center_crop: bool = False) -> Image.Image:
    """Convert numpy image to PIL Image with optional center crop."""
    if isinstance(full_image, list):
        full_image = full_image[0]
    
    image = Image.fromarray(full_image).convert("RGB")
    
    if center_crop:
        temp_image = np.array(image)
        crop_scale = 0.9
        sqrt_crop_scale = math.sqrt(crop_scale)
        temp_image_cropped = apply_center_crop(
            temp_image,
            t_h=int(sqrt_crop_scale * temp_image.shape[0]),
            t_w=int(sqrt_crop_scale * temp_image.shape[1]),
        )
        image = Image.fromarray(temp_image_cropped)
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
    
    return image

# ============================================================================
# Speculative decoding core implementation
# ============================================================================
@dataclass
class SpeculativeDecodingStats:
    """Statistics from speculative decoding run."""
    total_tokens_generated: int = 0
    total_draft_tokens_proposed: int = 0
    total_draft_tokens_accepted: int = 0
    total_target_forward_passes: int = 0
    total_draft_forward_passes: int = 0
    
    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens_proposed == 0:
            return 0.0
        return self.total_draft_tokens_accepted / self.total_draft_tokens_proposed
    
    @property
    def tokens_per_target_forward(self) -> float:
        if self.total_target_forward_passes == 0:
            return 0.0
        return self.total_tokens_generated / self.total_target_forward_passes

def max_fn(x: torch.Tensor) -> torch.Tensor:
    """Normalize max(0, x) to create a valid probability distribution."""
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    # Avoid division by zero
    return x_max / (x_max_sum + 1e-10)

class VLASpeculativeDecoder:
    """
    Speculative decoding for VLA models.
    
    Uses a draft model (MiniVLA) to propose action tokens and a target model 
    (OpenVLA) to verify them.
    
    IMPORTANT: For speculative decoding to work correctly, both models should
    share the same tokenizer/vocabulary. If they don't, action token remapping
    is attempted but this only works for action tokens (last 256 tokens of vocab).
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        target_processor=None,
        gamma: int = 4,  # Number of draft tokens to propose at once
        use_cache: bool = True,
        temperature: float = 0.0,  # 0 = greedy/argmax
        n_action_bins: int = 256,  # Number of action bins (tokens)
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: OpenVLA model (larger, slower, more accurate)
            draft_model: MiniVLA model (smaller, faster)
            target_processor: HuggingFace processor for target model
            gamma: Number of tokens to speculate at each step
            use_cache: Whether to use KV caching
            temperature: Sampling temperature (0 = greedy/argmax)
            n_action_bins: Number of action token bins (typically 256)
        """
        self.target = target_model
        self.draft = draft_model
        self.target_processor = target_processor
        self.gamma = gamma
        self.use_cache = use_cache
        self.temperature = temperature
        self.n_action_bins = n_action_bins
        
        # Get device
        self.device = next(target_model.parameters()).device
        
        # Stats tracking
        self.stats = SpeculativeDecodingStats()
        
        # Check vocabulary compatibility and setup token mapping
        self._setup_token_mapping()
    
    def _setup_token_mapping(self):
        """Setup token mapping between draft and target vocabularies."""
        # Get vocabulary sizes
        # Target model (OpenVLA/HF style) - use ACTUAL embedding dimension, not vocab_size attribute
        # The embedding may be padded to "multiple of" for efficiency
        if hasattr(self.target, 'language_model') and hasattr(self.target.language_model, 'model'):
            # Actual embedding dimension (includes padding)
            self.target_logit_dim = self.target.language_model.model.embed_tokens.weight.shape[0]
        elif hasattr(self.target, 'get_output_embeddings'):
            self.target_logit_dim = self.target.get_output_embeddings().weight.shape[0]
        else:
            self.target_logit_dim = self.target.config.vocab_size
        
        # Also get the "logical" vocab size (without padding) for action token calculation
        if hasattr(self.target, 'vocab_size'):
            self.target_vocab_size = self.target.vocab_size
        elif hasattr(self.target, 'config') and hasattr(self.target.config, 'vocab_size'):
            self.target_vocab_size = self.target.config.vocab_size
        else:
            self.target_vocab_size = self.target_logit_dim
        
        # Draft model (Prismatic style)
        if hasattr(self.draft, 'llm_backbone'):
            draft_tokenizer = self.draft.llm_backbone.tokenizer
            # Qwen2 uses len(tokenizer) for full vocab including added tokens
            self.draft_vocab_size = len(draft_tokenizer) if hasattr(draft_tokenizer, '__len__') else draft_tokenizer.vocab_size
            # Get actual logit dimension from draft model
            if hasattr(self.draft.llm_backbone, 'llm') and hasattr(self.draft.llm_backbone.llm, 'lm_head'):
                self.draft_logit_dim = self.draft.llm_backbone.llm.lm_head.weight.shape[0]
            else:
                self.draft_logit_dim = self.draft_vocab_size
        else:
            self.draft_vocab_size = self.draft.config.vocab_size
            self.draft_logit_dim = self.draft_vocab_size
        
        # Check if vocabularies match
        self.vocab_compatible = (self.target_logit_dim == self.draft_logit_dim)
        
        # Compute action token ranges
        # Action tokens are the LAST n_action_bins tokens before padding
        # For target: use vocab_size (not padded logit_dim)
        self.target_action_start = self.target_vocab_size - self.n_action_bins
        self.draft_action_start = self.draft_vocab_size - self.n_action_bins
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Target vocab_size: {self.target_vocab_size}, logit_dim: {self.target_logit_dim}, action tokens: [{self.target_action_start}, {self.target_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Draft vocab_size: {self.draft_vocab_size}, logit_dim: {self.draft_logit_dim}, action tokens: [{self.draft_action_start}, {self.draft_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Vocabularies compatible: {self.vocab_compatible}")
        
        if not self.vocab_compatible:
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"WARNING: Vocabulary mismatch! Token remapping will be used for action tokens only.")
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"This may affect acceptance rates. Consider using models with matching tokenizers.")
    
    def _draft_token_to_target(self, draft_token_id: int) -> int:
        """Map a draft token ID to target vocabulary."""
        if self.vocab_compatible:
            return draft_token_id
        
        # Check if this is an action token (from end of draft vocabulary)
        if draft_token_id >= self.draft_action_start:
            # Map to corresponding action token in target vocabulary
            action_bin = draft_token_id - self.draft_action_start
            target_token = self.target_action_start + action_bin
            return target_token
        else:
            # Non-action token - this shouldn't happen during action generation
            # Return as-is but clamp to valid range
            return min(draft_token_id, self.target_vocab_size - 1)
    
    def _target_token_to_draft(self, target_token_id: int) -> int:
        """Map a target token ID to draft vocabulary."""
        if self.vocab_compatible:
            return target_token_id
        
        # Check if this is an action token
        if target_token_id >= self.target_action_start:
            action_bin = target_token_id - self.target_action_start
            draft_token = self.draft_action_start + action_bin
            return draft_token
        else:
            return min(target_token_id, self.draft_vocab_size - 1)
    
    def _remap_logits_draft_to_target(self, draft_logits: torch.Tensor, target_logit_dim: int = None) -> torch.Tensor:
        """
        Remap draft logits to target vocabulary space.
        Only remaps action tokens; non-action tokens get -inf.
        
        Args:
            draft_logits: Logits from draft model
            target_logit_dim: Actual dimension of target logits (may differ from vocab_size due to padding)
        """
        if self.vocab_compatible:
            return draft_logits
        
        # Use provided dimension or fall back to stored logit_dim
        if target_logit_dim is None:
            target_logit_dim = self.target_logit_dim
        
        # Create target-sized logits filled with -inf (use actual logit dimension, not vocab_size)
        target_logits = torch.full(
            (draft_logits.shape[0], target_logit_dim),
            float('-inf'),
            device=draft_logits.device,
            dtype=draft_logits.dtype
        )
        
        # Copy action token logits from draft to corresponding target positions
        # Draft action tokens: [draft_action_start, draft_vocab_size)
        # Target action tokens: [target_action_start, target_vocab_size)
        draft_action_logits = draft_logits[:, self.draft_action_start:self.draft_vocab_size]
        target_logits[:, self.target_action_start:self.target_vocab_size] = draft_action_logits
        
        return target_logits
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = SpeculativeDecodingStats()
    
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
    
    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        if self.temperature <= 0:
            # For greedy, use a very low temperature to approximate argmax
            return F.softmax(logits / 0.01, dim=-1)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _prepare_target_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the target (OpenVLA) model."""
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.target_processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        return inputs
    
    def _prepare_draft_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the draft (MiniVLA) model."""
        # Get prompt using draft model's prompt builder
        prompt_builder = self.draft.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        # Tokenize
        tokenizer = self.draft.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # Handle special token for Llama tokenizer
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], device=self.device)),
                    dim=1
                )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Process image
        image_transform = self.draft.vision_backbone.get_image_transform()
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    @torch.inference_mode()
    def predict_action_speculative(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key_target: str,
    ) -> Tuple[np.ndarray, SpeculativeDecodingStats]:
        """
        Generate action using speculative decoding.
        
        Args:
            image: PIL Image observation
            instruction: Task instruction string
            unnorm_key_target: Key for action un-normalization statistics of the target model
            
        Returns:
            Tuple of (unnormalized action array, decoding statistics)
        """
        # Reset per-call stats
        call_stats = SpeculativeDecodingStats()
        
        action_dim = self.target.get_action_dim(unnorm_key_target)
        
        
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target vocab size: {self.target.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target embedding size: {self.target.language_model.model.embed_tokens.weight.shape[0]}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Draft tokenizer vocab size: {self.draft.llm_backbone.tokenizer.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Action dim: {action_dim}")
        
        # Prepare inputs for both models
        target_inputs = self._prepare_target_inputs(image, instruction)
        draft_input_ids, draft_attention_mask, draft_pixel_values = self._prepare_draft_inputs(image, instruction)
        
        # Cast draft inputs to appropriate dtype
        autocast_dtype = self.draft.llm_backbone.half_precision_dtype
        
        # Initialize caches
        target_cache = None
        draft_cache = None
        
        generated_token_ids = []
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # === Initial forward pass to get first token and cache ===
            
            # Target model initial forward
            target_out = self.target(
                **target_inputs,
                past_key_values=None,
                use_cache=self.use_cache,
            )
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # Draft model initial forward
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_out = self.draft(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    pixel_values=draft_pixel_values,
                    past_key_values=None,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # Sample first token from target (in target vocab space)
            first_token_target = self._sample_token(target_logits)
            target_token_id = int(first_token_target.item())
            generated_token_ids.append(target_token_id)  # Store in target vocab space
            call_stats.total_tokens_generated += 1
            
            # Update target cache with target token
            target_step = self.target(
                input_ids=first_token_target,
                past_key_values=target_cache,
                use_cache=self.use_cache,
            )
            target_cache = target_step.past_key_values
            target_logits = target_step.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # Map token to draft vocab space for draft model
            first_token_draft_id = self._target_token_to_draft(target_token_id)
            first_token_draft = torch.tensor([[first_token_draft_id]], device=self.device)
            
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_step = self.draft(
                    input_ids=first_token_draft,
                    past_key_values=draft_cache,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_step.past_key_values
            draft_logits = draft_step.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # === Main speculative decoding loop ===
            while len(generated_token_ids) < action_dim:
                # Determine how many tokens to speculate
                gamma = min(self.gamma, action_dim - len(generated_token_ids))
                
                # Generate gamma draft tokens
                draft_tokens = []
                draft_probs_list = []
                
                current_draft_cache = draft_cache
                current_draft_logits = draft_logits
                
                for _ in range(gamma):
                    draft_probs = self._get_probs(current_draft_logits)
                    draft_token = self._sample_token(current_draft_logits)
                    
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(draft_probs)
                    
                    # Advance draft model
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=draft_token.to(self.device),
                            past_key_values=current_draft_cache,
                            use_cache=self.use_cache,
                        )
                    current_draft_cache = draft_step.past_key_values
                    current_draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                
                call_stats.total_draft_tokens_proposed += gamma
                
                # Map draft tokens to target vocab space for verification
                draft_token_ids_target = []
                for dt in draft_tokens:
                    draft_id = dt.item()
                    target_id = self._draft_token_to_target(draft_id)
                    draft_token_ids_target.append(target_id)
                
                # Verify with target model - run all gamma tokens through
                target_cache_for_verify = target_cache
                target_logits_list = []
                
                for i in range(gamma):
                    target_token_input = torch.tensor([[draft_token_ids_target[i]]], device=self.device)
                    target_step = self.target(
                        input_ids=target_token_input,
                        past_key_values=target_cache_for_verify,
                        use_cache=self.use_cache,
                    )
                    target_cache_for_verify = target_step.past_key_values
                    target_logits_list.append(target_step.logits[:, -1:, :])
                
                call_stats.total_target_forward_passes += gamma
                
                # Stack target logits
                target_logits_batch = torch.cat(target_logits_list, dim=1)  # [1, gamma, actual_vocab_dim]
                target_probs_batch = self._get_probs(target_logits_batch)
                
                # Get actual target logit dimension from the output
                actual_target_logit_dim = target_logits_batch.shape[-1]
                
                # Remap draft probs to target vocab space for comparison
                # Use actual target logit dimension to ensure tensor size match
                draft_probs_remapped = [self._remap_logits_draft_to_target(dp, actual_target_logit_dim) for dp in draft_probs_list]
                draft_probs_remapped = [self._get_probs(dp) for dp in draft_probs_remapped]
                
                # Rejection sampling loop
                n_accepted = 0
                for i in range(gamma):
                    draft_token_id_draft = draft_tokens[i].item()  # In draft vocab
                    draft_token_id_target = draft_token_ids_target[i]  # Mapped to target vocab
                    
                    draft_prob_remapped = draft_probs_remapped[i]
                    target_prob = target_probs_batch[:, i, :]
                    
                    # Get probability of the token under both models (in target vocab space)
                    p_target = target_prob[0, draft_token_id_target].item()
                    p_draft = draft_prob_remapped[0, draft_token_id_target].item()
                    
                    # Rejection sampling
                    if p_draft > 0:
                        acceptance_prob = min(1.0, p_target / p_draft)
                    else:
                        acceptance_prob = 1.0 if p_target > 0 else 0.0
                    
                    if torch.rand(1).item() < acceptance_prob:
                        # Accept this token (store in target vocab space)
                        print(f"\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Accepted token: {draft_token_id_target} since p_target={p_target:.4f} > p_draft={p_draft:.4f}")
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        if len(generated_token_ids) >= action_dim:
                            print(f"\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Generated {len(generated_token_ids)} tokens, breaking")
                            break
                    else:
                        # Reject - sample from adjusted distribution
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        if adjusted_probs.sum() > 0:
                            corrected_token = torch.multinomial(adjusted_probs, num_samples=1)
                        else:
                            corrected_token = self._sample_token(target_prob.unsqueeze(0))
                        
                        print(f"\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Rejected token: {draft_token_id_target} since p_target={p_target:.4f} < p_draft={p_draft:.4f}, corrected token: {corrected_token.item()}")
                        print(f"\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Absolute id distance between rejected and corrected token: {abs(draft_token_id_target - corrected_token.item())}")
                        
                        # Store corrected token (already in target vocab space)
                        generated_token_ids.append(int(corrected_token.item()))
                        call_stats.total_tokens_generated += 1
                        n_accepted = i  # Number of accepted tokens (before this rejection)
                        break
                                    
                # Update caches after acceptance/rejection
                if n_accepted == gamma and len(generated_token_ids) < action_dim:
                    # All accepted - need to sample one more from target
                    target_cache = target_cache_for_verify
                    target_logits = target_logits_list[-1].squeeze(1)
                    
                    # Sample additional token from target (in target vocab space)
                    bonus_token_target = self._sample_token(target_logits)
                    bonus_token_id_target = int(bonus_token_target.item())
                    generated_token_ids.append(bonus_token_id_target)
                    call_stats.total_tokens_generated += 1
                    
                    # Update target cache
                    target_step = self.target(
                        input_ids=bonus_token_target,
                        past_key_values=target_cache,
                        use_cache=self.use_cache,
                    )
                    target_cache = target_step.past_key_values
                    target_logits = target_step.logits[:, -1, :]
                    call_stats.total_target_forward_passes += 1
                    
                    # Map bonus token to draft vocab and update draft cache
                    bonus_token_id_draft = self._target_token_to_draft(bonus_token_id_target)
                    bonus_token_draft = torch.tensor([[bonus_token_id_draft]], device=self.device)
                    
                    draft_cache = current_draft_cache
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=bonus_token_draft,
                            past_key_values=draft_cache,
                            use_cache=self.use_cache,
                        )
                    draft_cache = draft_step.past_key_values
                    draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                    
                else:
                    # Some tokens rejected - prune caches
                    tokens_to_discard = gamma - n_accepted
                    if tokens_to_discard > 0 and self.use_cache:
                        # We need to prune and resync
                        # Use the cache state after the accepted tokens
                        target_cache = target_cache_for_verify
                        if tokens_to_discard > 0:
                            target_cache = prune_cache(target_cache, tokens_to_discard)
                        
                        # Rebuild draft cache
                        draft_cache = prune_cache(current_draft_cache, gamma - n_accepted)
                    
                    # Get logits for next round
                    if len(generated_token_ids) < action_dim:
                        # Last token is in target vocab space
                        last_token_id_target = generated_token_ids[-1]
                        last_token_target = torch.tensor([[last_token_id_target]], device=self.device)
                        
                        target_step = self.target(
                            input_ids=last_token_target,
                            past_key_values=target_cache,
                            use_cache=self.use_cache,
                        )
                        target_cache = target_step.past_key_values
                        target_logits = target_step.logits[:, -1, :]
                        call_stats.total_target_forward_passes += 1
                        
                        # Map to draft vocab for draft model
                        last_token_id_draft = self._target_token_to_draft(last_token_id_target)
                        last_token_draft = torch.tensor([[last_token_id_draft]], device=self.device)
                        
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                            draft_step = self.draft(
                                input_ids=last_token_draft,
                                past_key_values=draft_cache,
                                use_cache=self.use_cache,
                            )
                        draft_cache = draft_step.past_key_values
                        draft_logits = draft_step.logits[:, -1, :]
                        call_stats.total_draft_forward_passes += 1
            
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Call stats: {call_stats}")
            
        # Decode tokens to actions
        predicted_action_token_ids = np.array(generated_token_ids[:action_dim], dtype=np.int64)
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Predicted action token ids: {predicted_action_token_ids}")
        
        # Use target model's decoding (vocab_size - token_id approach)
        vocab_size = self.target.vocab_size
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.target.bin_centers.shape[0] - 1)
        normalized_actions = self.target.bin_centers[discretized_actions]
        
        # Un-normalize actions
        action_norm_stats = self.target.get_action_stats(unnorm_key_target)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # Update global stats
        self.stats.total_tokens_generated += call_stats.total_tokens_generated
        self.stats.total_draft_tokens_proposed += call_stats.total_draft_tokens_proposed
        self.stats.total_draft_tokens_accepted += call_stats.total_draft_tokens_accepted
        self.stats.total_target_forward_passes += call_stats.total_target_forward_passes
        self.stats.total_draft_forward_passes += call_stats.total_draft_forward_passes
        
        return actions, call_stats


# ============================================================================
# Simplified speculative action prediction (standalone function)
# ============================================================================

@torch.inference_mode()
def speculative_predict_action(
    target_vla,
    draft_vla,
    target_processor,
    observation: Dict,
    instruction: str,
    unnorm_key_target: str,
    center_crop: bool = False,
    gamma: int = 4,
    temperature: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Speculative decoding for VLA action prediction.
    
    Args:
        target_vla: OpenVLA model (loaded via get_vla)
        draft_vla: MiniVLA model (loaded via get_prismatic_vla)
        target_processor: HuggingFace processor for target
        observation: Dict with 'full_image' key
        instruction: Task instruction string
        unnorm_key_target: Key for un-normalization of the target model
        center_crop: Whether to center crop the image
        gamma: Number of tokens to speculate
        temperature: Sampling temperature
        
    Returns:
        Tuple of (action array, acceptance rate)
    """
    # Prepare image
    image = prepare_image(observation["full_image"], center_crop=center_crop)
    
    # Create decoder
    decoder = VLASpeculativeDecoder(
        target_model=target_vla,
        draft_model=draft_vla,
        target_processor=target_processor,
        gamma=gamma,
        use_cache=True,
        temperature=temperature,
    )
    
    # Run speculative decoding
    action, stats = decoder.predict_action_speculative(image, instruction, unnorm_key_target)
    
    return action, stats.acceptance_rate



# %%
class VLASpeculativeDecoderD:
    """
    Speculative decoding for VLA models.
    
    Uses a draft model (MiniVLA) to propose action tokens and a target model 
    (OpenVLA) to verify them.
    
    IMPORTANT: For speculative decoding to work correctly, both models should
    share the same tokenizer/vocabulary. If they don't, action token remapping
    is attempted but this only works for action tokens (last 256 tokens of vocab).
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        target_processor=None,
        gamma: int = 4,  # Number of draft tokens to propose at once
        use_cache: bool = True,
        temperature: float = 0.0,  # 0 = greedy/argmax
        n_action_bins: int = 256,  # Number of action bins (tokens)
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: OpenVLA model (larger, slower, more accurate)
            draft_model: MiniVLA model (smaller, faster)
            target_processor: HuggingFace processor for target model
            gamma: Number of tokens to speculate at each step
            use_cache: Whether to use KV caching
            temperature: Sampling temperature (0 = greedy/argmax)
            n_action_bins: Number of action token bins (typically 256)
        """
        self.target = target_model
        self.draft = draft_model
        self.target_processor = target_processor
        self.gamma = gamma
        self.use_cache = use_cache
        self.temperature = temperature
        self.n_action_bins = n_action_bins
        
        # Get device
        self.device = next(target_model.parameters()).device
        
        # Stats tracking
        self.stats = SpeculativeDecodingStats()
        
        # Check vocabulary compatibility and setup token mapping
        self._setup_token_mapping()
    
    def _setup_token_mapping(self):
        """Setup token mapping between draft and target vocabularies."""
        # Get vocabulary sizes
        # Target model (OpenVLA/HF style) - use ACTUAL embedding dimension, not vocab_size attribute
        # The embedding may be padded to "multiple of" for efficiency
        if hasattr(self.target, 'language_model') and hasattr(self.target.language_model, 'model'):
            # Actual embedding dimension (includes padding)
            self.target_logit_dim = self.target.language_model.model.embed_tokens.weight.shape[0]
        elif hasattr(self.target, 'get_output_embeddings'):
            self.target_logit_dim = self.target.get_output_embeddings().weight.shape[0]
        else:
            self.target_logit_dim = self.target.config.vocab_size
        
        # Also get the "logical" vocab size (without padding) for action token calculation
        if hasattr(self.target, 'vocab_size'):
            self.target_vocab_size = self.target.vocab_size
        elif hasattr(self.target, 'config') and hasattr(self.target.config, 'vocab_size'):
            self.target_vocab_size = self.target.config.vocab_size
        else:
            self.target_vocab_size = self.target_logit_dim
        
        # Draft model (Prismatic style)
        if hasattr(self.draft, 'llm_backbone'):
            draft_tokenizer = self.draft.llm_backbone.tokenizer
            # Qwen2 uses len(tokenizer) for full vocab including added tokens
            self.draft_vocab_size = len(draft_tokenizer) if hasattr(draft_tokenizer, '__len__') else draft_tokenizer.vocab_size
            # Get actual logit dimension from draft model
            if hasattr(self.draft.llm_backbone, 'llm') and hasattr(self.draft.llm_backbone.llm, 'lm_head'):
                self.draft_logit_dim = self.draft.llm_backbone.llm.lm_head.weight.shape[0]
            else:
                self.draft_logit_dim = self.draft_vocab_size
        else:
            self.draft_vocab_size = self.draft.config.vocab_size
            self.draft_logit_dim = self.draft_vocab_size
        
        # Check if vocabularies match
        self.vocab_compatible = (self.target_logit_dim == self.draft_logit_dim)
        
        # Compute action token ranges
        # Action tokens are the LAST n_action_bins tokens before padding
        # For target: use vocab_size (not padded logit_dim)
        self.target_action_start = self.target_vocab_size - self.n_action_bins
        self.draft_action_start = self.draft_vocab_size - self.n_action_bins
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Target vocab_size: {self.target_vocab_size}, logit_dim: {self.target_logit_dim}, action tokens: [{self.target_action_start}, {self.target_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Draft vocab_size: {self.draft_vocab_size}, logit_dim: {self.draft_logit_dim}, action tokens: [{self.draft_action_start}, {self.draft_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Vocabularies compatible: {self.vocab_compatible}")
        
        if not self.vocab_compatible:
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"WARNING: Vocabulary mismatch! Token remapping will be used for action tokens only.")
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"This may affect acceptance rates. Consider using models with matching tokenizers.")
        
        # DEBUG: Verify mapping with example tokens
        print("\033[38;2;100;200;255m[DEBUG] Token mapping verification examples:\033[0m")
        for bin_idx in [0, 127, 255]:  # First, middle, last action bins
            draft_tok = self.draft_action_start + bin_idx
            target_tok = self._draft_token_to_target(draft_tok)
            draft_bin = self._get_action_bin_from_draft_token(draft_tok)
            target_bin = self._get_action_bin_from_target_token(target_tok)
            print(f"  Bin {bin_idx}: draft_token={draft_tok} → target_token={target_tok} | draft_bin={draft_bin}, target_bin={target_bin} | {'✓' if draft_bin == target_bin else '✗'}")
    
    def _draft_token_to_target(self, draft_token_id: int) -> int:
        """Map a draft token ID to target vocabulary."""
        if self.vocab_compatible:
            return draft_token_id
        
        # Check if this is an action token (from end of draft vocabulary)
        if draft_token_id >= self.draft_action_start:
            # Map to corresponding action token in target vocabulary
            action_bin = draft_token_id - self.draft_action_start
            target_token = self.target_action_start + action_bin
            return target_token
        else:
            # Non-action token - this shouldn't happen during action generation
            # Return as-is but clamp to valid range
            return min(draft_token_id, self.target_vocab_size - 1)
    
    def _target_token_to_draft(self, target_token_id: int) -> int:
        """Map a target token ID to draft vocabulary."""
        if self.vocab_compatible:
            return target_token_id
        
        # Check if this is an action token
        if target_token_id >= self.target_action_start:
            action_bin = target_token_id - self.target_action_start
            draft_token = self.draft_action_start + action_bin
            return draft_token
        else:
            return min(target_token_id, self.draft_vocab_size - 1)
    
    def _remap_logits_draft_to_target(self, draft_logits: torch.Tensor, target_logit_dim: int = None) -> torch.Tensor:
        """
        Remap draft logits to target vocabulary space.
        Only remaps action tokens; non-action tokens get -inf.
        
        Args:
            draft_logits: Logits from draft model
            target_logit_dim: Actual dimension of target logits (may differ from vocab_size due to padding)
        """
        if self.vocab_compatible:
            return draft_logits
        
        # Use provided dimension or fall back to stored logit_dim
        if target_logit_dim is None:
            target_logit_dim = self.target_logit_dim
        
        # Create target-sized logits filled with -inf (use actual logit dimension, not vocab_size)
        target_logits = torch.full(
            (draft_logits.shape[0], target_logit_dim),
            float('-inf'),
            device=draft_logits.device,
            dtype=draft_logits.dtype
        )
        
        # Copy action token logits from draft to corresponding target positions
        # Draft action tokens: [draft_action_start, draft_vocab_size)
        # Target action tokens: [target_action_start, target_vocab_size)
        draft_action_logits = draft_logits[:, self.draft_action_start:self.draft_vocab_size]
        target_logits[:, self.target_action_start:self.target_vocab_size] = draft_action_logits
        
        return target_logits
    
    # =========================================================================
    # DEBUG: Token mapping verification methods
    # =========================================================================
    
    def _get_action_bin_from_draft_token(self, draft_token_id: int) -> int:
        """Get action bin index from draft token ID."""
        if draft_token_id >= self.draft_action_start:
            return draft_token_id - self.draft_action_start
        return -1  # Not an action token
    
    def _get_action_bin_from_target_token(self, target_token_id: int) -> int:
        """Get action bin index from target token ID."""
        if target_token_id >= self.target_action_start:
            return target_token_id - self.target_action_start
        return -1  # Not an action token
    
    def _get_continuous_action_from_bin(self, bin_idx: int) -> float:
        """Convert action bin index to continuous action value using target's bin centers."""
        if 0 <= bin_idx < len(self.target.bin_centers):
            return self.target.bin_centers[bin_idx]
        return float('nan')
    
    def _debug_token_mapping(self, draft_token_id: int, target_token_id: int, prefix: str = ""):
        """Debug print showing token mapping verification."""
        draft_bin = self._get_action_bin_from_draft_token(draft_token_id)
        target_bin = self._get_action_bin_from_target_token(target_token_id)
        
        draft_action = self._get_continuous_action_from_bin(draft_bin)
        target_action = self._get_continuous_action_from_bin(target_bin)
        
        match_status = "✓ MATCH" if draft_bin == target_bin else "✗ MISMATCH"
        
        print(f"\033[38;2;100;200;255m[DEBUG TOKEN MAP] {prefix}\033[0m")
        print(f"  Draft token:  {draft_token_id} → bin {draft_bin} → action {draft_action:.4f}")
        print(f"  Target token: {target_token_id} → bin {target_bin} → action {target_action:.4f}")
        print(f"  Bins match: {match_status}")
        
        return draft_bin == target_bin
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = SpeculativeDecodingStats()
    
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
    
    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        if self.temperature <= 0:
            # For greedy, use a very low temperature to approximate argmax
            return F.softmax(logits / 0.01, dim=-1)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _prepare_target_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the target (OpenVLA) model."""
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.target_processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        return inputs
    
    def _prepare_draft_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the draft (MiniVLA) model."""
        # Get prompt using draft model's prompt builder
        prompt_builder = self.draft.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        # Tokenize
        tokenizer = self.draft.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # Handle special token for Llama tokenizer
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], device=self.device)),
                    dim=1
                )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Process image
        image_transform = self.draft.vision_backbone.get_image_transform()
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    @torch.inference_mode()
    def predict_action_speculative(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key_target: str,
    ) -> Tuple[np.ndarray, SpeculativeDecodingStats]:
        """
        Generate action using speculative decoding.
        
        Args:
            image: PIL Image observation
            instruction: Task instruction string
            unnorm_key_target: Key for action un-normalization statistics of the target model
            
        Returns:
            Tuple of (unnormalized action array, decoding statistics)
        """
        # Reset per-call stats
        call_stats = SpeculativeDecodingStats()
        
        action_dim = self.target.get_action_dim(unnorm_key_target)
        
        
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target vocab size: {self.target.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target embedding size: {self.target.language_model.model.embed_tokens.weight.shape[0]}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Draft tokenizer vocab size: {self.draft.llm_backbone.tokenizer.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Action dim: {action_dim}")
        
        # Prepare inputs for both models
        target_inputs = self._prepare_target_inputs(image, instruction)
        draft_input_ids, draft_attention_mask, draft_pixel_values = self._prepare_draft_inputs(image, instruction)
        
        # Cast draft inputs to appropriate dtype
        autocast_dtype = self.draft.llm_backbone.half_precision_dtype
        
        # Initialize caches
        target_cache = None
        draft_cache = None
        
        generated_token_ids = []
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # === Initial forward pass to get first token and cache ===
            
            # Target model initial forward
            target_out = self.target(
                **target_inputs,
                past_key_values=None,
                use_cache=self.use_cache,
            )
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # Draft model initial forward
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_out = self.draft(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    pixel_values=draft_pixel_values,
                    past_key_values=None,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # Sample first token from target (in target vocab space)
            first_token_target = self._sample_token(target_logits)
            target_token_id = int(first_token_target.item())
            generated_token_ids.append(target_token_id)  # Store in target vocab space
            call_stats.total_tokens_generated += 1
            
            # DEBUG: Show first token info
            first_target_bin = self._get_action_bin_from_target_token(target_token_id)
            first_target_action = self._get_continuous_action_from_bin(first_target_bin)
            print(f"\033[38;2;100;200;255m[DEBUG] First token sampled from target:\033[0m")
            print(f"  target_token={target_token_id} → bin={first_target_bin} → action={first_target_action:.4f}")
            
            # Update target cache with target token
            target_step = self.target(
                input_ids=first_token_target,
                past_key_values=target_cache,
                use_cache=self.use_cache,
            )
            target_cache = target_step.past_key_values
            target_logits = target_step.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # Map token to draft vocab space for draft model
            first_token_draft_id = self._target_token_to_draft(target_token_id)
            first_token_draft = torch.tensor([[first_token_draft_id]], device=self.device)
            
            # DEBUG: Verify reverse mapping
            first_draft_bin = self._get_action_bin_from_draft_token(first_token_draft_id)
            print(f"  Mapped to draft: draft_token={first_token_draft_id} → bin={first_draft_bin} {'✓' if first_target_bin == first_draft_bin else '✗ MISMATCH!'}")
            
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_step = self.draft(
                    input_ids=first_token_draft,
                    past_key_values=draft_cache,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_step.past_key_values
            draft_logits = draft_step.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # === Main speculative decoding loop ===
            while len(generated_token_ids) < action_dim:
                # Determine how many tokens to speculate
                gamma = min(self.gamma, action_dim - len(generated_token_ids))
                
                # Generate gamma draft tokens
                draft_tokens = []
                draft_probs_list = []
                
                current_draft_cache = draft_cache
                current_draft_logits = draft_logits
                
                for _ in range(gamma):
                    draft_probs = self._get_probs(current_draft_logits)
                    draft_token = self._sample_token(current_draft_logits)
                    
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(draft_probs)
                    
                    # Advance draft model
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=draft_token.to(self.device),
                            past_key_values=current_draft_cache,
                            use_cache=self.use_cache,
                        )
                    current_draft_cache = draft_step.past_key_values
                    current_draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                
                call_stats.total_draft_tokens_proposed += gamma
                
                # Map draft tokens to target vocab space for verification
                draft_token_ids_target = []
                print(f"\033[38;2;100;200;255m[DEBUG] Mapping {gamma} draft tokens to target vocab:\033[0m")
                for idx, dt in enumerate(draft_tokens):
                    draft_id = dt.item()
                    target_id = self._draft_token_to_target(draft_id)
                    draft_token_ids_target.append(target_id)
                    
                    # Verify mapping preserves action bin
                    draft_bin = self._get_action_bin_from_draft_token(draft_id)
                    target_bin = self._get_action_bin_from_target_token(target_id)
                    draft_action = self._get_continuous_action_from_bin(draft_bin)
                    target_action = self._get_continuous_action_from_bin(target_bin)
                    match = "✓" if draft_bin == target_bin else "✗ MISMATCH!"
                    print(f"  [{idx}] draft_tok={draft_id} → target_tok={target_id} | bin: {draft_bin}→{target_bin} | action: {draft_action:.4f}→{target_action:.4f} {match}")
                
                # Verify with target model - run all gamma tokens through
                target_cache_for_verify = target_cache
                target_logits_list = []
                
                for i in range(gamma):
                    target_token_input = torch.tensor([[draft_token_ids_target[i]]], device=self.device)
                    target_step = self.target(
                        input_ids=target_token_input,
                        past_key_values=target_cache_for_verify,
                        use_cache=self.use_cache,
                    )
                    target_cache_for_verify = target_step.past_key_values
                    target_logits_list.append(target_step.logits[:, -1:, :])
                
                call_stats.total_target_forward_passes += gamma
                
                # Stack target logits
                target_logits_batch = torch.cat(target_logits_list, dim=1)  # [1, gamma, actual_vocab_dim]
                target_probs_batch = self._get_probs(target_logits_batch)
                
                # Get actual target logit dimension from the output
                actual_target_logit_dim = target_logits_batch.shape[-1]
                
                # Remap draft probs to target vocab space for comparison
                # Use actual target logit dimension to ensure tensor size match
                draft_probs_remapped = [self._remap_logits_draft_to_target(dp, actual_target_logit_dim) for dp in draft_probs_list]
                draft_probs_remapped = [self._get_probs(dp) for dp in draft_probs_remapped]
                
                # Rejection sampling loop
                n_accepted = 0
                for i in range(gamma):
                    draft_token_id_draft = draft_tokens[i].item()  # In draft vocab
                    draft_token_id_target = draft_token_ids_target[i]  # Mapped to target vocab
                    
                    draft_prob_remapped = draft_probs_remapped[i]
                    target_prob = target_probs_batch[:, i, :]
                    
                    # Get probability of the token under both models (in target vocab space)
                    p_target = target_prob[0, draft_token_id_target].item()
                    p_draft = draft_prob_remapped[0, draft_token_id_target].item()
                    
                    # Rejection sampling
                    if p_draft > 0:
                        acceptance_prob = min(1.0, p_target / p_draft)
                    else:
                        acceptance_prob = 1.0 if p_target > 0 else 0.0
                    
                    if torch.rand(1).item() < acceptance_prob:
                        # Accept this token (store in target vocab space)
                        accepted_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                        accepted_action = self._get_continuous_action_from_bin(accepted_bin)
                        print(f"\033[38;2;0;255;0m[ACCEPT]\033[0m token[{i}]: target_tok={draft_token_id_target} → bin={accepted_bin} → action={accepted_action:.4f} | p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        if len(generated_token_ids) >= action_dim:
                            print(f"\033[38;2;255;165;0m[SRP] -> \033[0m Generated {len(generated_token_ids)}/{action_dim} tokens, done.")
                            break
                    else:
                        # Reject - sample from adjusted distribution
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        if adjusted_probs.sum() > 0:
                            corrected_token = torch.multinomial(adjusted_probs, num_samples=1)
                        else:
                            corrected_token = self._sample_token(target_prob.unsqueeze(0))
                        
                        corrected_token_id = int(corrected_token.item())
                        rejected_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                        corrected_bin = self._get_action_bin_from_target_token(corrected_token_id)
                        rejected_action = self._get_continuous_action_from_bin(rejected_bin)
                        corrected_action = self._get_continuous_action_from_bin(corrected_bin)
                        bin_diff = abs(rejected_bin - corrected_bin) if rejected_bin >= 0 and corrected_bin >= 0 else -1
                        
                        print(f"\033[38;2;255;100;100m[REJECT]\033[0m token[{i}]: p_target={p_target:.4f} < p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        print(f"  Rejected:  target_tok={draft_token_id_target} → bin={rejected_bin} → action={rejected_action:.4f}")
                        print(f"  Corrected: target_tok={corrected_token_id} → bin={corrected_bin} → action={corrected_action:.4f}")
                        print(f"  Bin difference: {bin_diff} | Action difference: {abs(rejected_action - corrected_action):.4f}")
                        
                        # Store corrected token (already in target vocab space)
                        generated_token_ids.append(corrected_token_id)
                        call_stats.total_tokens_generated += 1
                        n_accepted = i  # Number of accepted tokens (before this rejection)
                        break
                                    
                # Update caches after acceptance/rejection
                if n_accepted == gamma and len(generated_token_ids) < action_dim:
                    # All accepted - need to sample one more from target
                    print(f"\033[38;2;0;255;0m[ALL ACCEPTED]\033[0m All {gamma} draft tokens accepted! Sampling bonus token from target...")
                    target_cache = target_cache_for_verify
                    target_logits = target_logits_list[-1].squeeze(1)
                    
                    # Sample additional token from target (in target vocab space)
                    bonus_token_target = self._sample_token(target_logits)
                    bonus_token_id_target = int(bonus_token_target.item())
                    bonus_bin = self._get_action_bin_from_target_token(bonus_token_id_target)
                    bonus_action = self._get_continuous_action_from_bin(bonus_bin)
                    print(f"  Bonus token: target_tok={bonus_token_id_target} → bin={bonus_bin} → action={bonus_action:.4f}")
                    generated_token_ids.append(bonus_token_id_target)
                    call_stats.total_tokens_generated += 1
                    
                    # Update target cache
                    target_step = self.target(
                        input_ids=bonus_token_target,
                        past_key_values=target_cache,
                        use_cache=self.use_cache,
                    )
                    target_cache = target_step.past_key_values
                    target_logits = target_step.logits[:, -1, :]
                    call_stats.total_target_forward_passes += 1
                    
                    # Map bonus token to draft vocab and update draft cache
                    bonus_token_id_draft = self._target_token_to_draft(bonus_token_id_target)
                    bonus_draft_bin = self._get_action_bin_from_draft_token(bonus_token_id_draft)
                    print(f"  Mapped to draft: draft_tok={bonus_token_id_draft} → bin={bonus_draft_bin} {'✓' if bonus_bin == bonus_draft_bin else '✗ MISMATCH!'}")
                    bonus_token_draft = torch.tensor([[bonus_token_id_draft]], device=self.device)
                    
                    draft_cache = current_draft_cache
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=bonus_token_draft,
                            past_key_values=draft_cache,
                            use_cache=self.use_cache,
                        )
                    draft_cache = draft_step.past_key_values
                    draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                    
                else:
                    # Some tokens rejected - prune caches
                    tokens_to_discard = gamma - n_accepted
                    if tokens_to_discard > 0 and self.use_cache:
                        # We need to prune and resync
                        # Use the cache state after the accepted tokens
                        target_cache = target_cache_for_verify
                        if tokens_to_discard > 0:
                            target_cache = prune_cache(target_cache, tokens_to_discard)
                        
                        # Rebuild draft cache
                        draft_cache = prune_cache(current_draft_cache, gamma - n_accepted)
                    
                    # Get logits for next round
                    if len(generated_token_ids) < action_dim:
                        # Last token is in target vocab space
                        last_token_id_target = generated_token_ids[-1]
                        last_token_target = torch.tensor([[last_token_id_target]], device=self.device)
                        
                        target_step = self.target(
                            input_ids=last_token_target,
                            past_key_values=target_cache,
                            use_cache=self.use_cache,
                        )
                        target_cache = target_step.past_key_values
                        target_logits = target_step.logits[:, -1, :]
                        call_stats.total_target_forward_passes += 1
                        
                        # Map to draft vocab for draft model
                        last_token_id_draft = self._target_token_to_draft(last_token_id_target)
                        last_token_draft = torch.tensor([[last_token_id_draft]], device=self.device)
                        
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                            draft_step = self.draft(
                                input_ids=last_token_draft,
                                past_key_values=draft_cache,
                                use_cache=self.use_cache,
                            )
                        draft_cache = draft_step.past_key_values
                        draft_logits = draft_step.logits[:, -1, :]
                        call_stats.total_draft_forward_passes += 1
            
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Call stats: {call_stats}")
            
        # Decode tokens to actions
        predicted_action_token_ids = np.array(generated_token_ids[:action_dim], dtype=np.int64)
        
        # DEBUG: Show all generated tokens with their action values
        print(f"\033[38;2;100;200;255m[DEBUG] Final generated tokens summary:\033[0m")
        for dim_idx, tok_id in enumerate(predicted_action_token_ids):
            bin_idx = self._get_action_bin_from_target_token(int(tok_id))
            action_val = self._get_continuous_action_from_bin(bin_idx)
            print(f"  dim[{dim_idx}]: target_tok={tok_id} → bin={bin_idx} → normalized_action={action_val:.4f}")
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Predicted action token ids: {predicted_action_token_ids}")
        
        # Use target model's decoding (vocab_size - token_id approach)
        vocab_size = self.target.vocab_size
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.target.bin_centers.shape[0] - 1)
        normalized_actions = self.target.bin_centers[discretized_actions]
        
        # Un-normalize actions
        action_norm_stats = self.target.get_action_stats(unnorm_key_target)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # Update global stats
        self.stats.total_tokens_generated += call_stats.total_tokens_generated
        self.stats.total_draft_tokens_proposed += call_stats.total_draft_tokens_proposed
        self.stats.total_draft_tokens_accepted += call_stats.total_draft_tokens_accepted
        self.stats.total_target_forward_passes += call_stats.total_target_forward_passes
        self.stats.total_draft_forward_passes += call_stats.total_draft_forward_passes
        
        return actions, call_stats

# %%
class VLASpeculativeDecoderDD:
    """
    Speculative decoding for VLA models.
    
    Uses a draft model (MiniVLA) to propose action tokens and a target model 
    (OpenVLA) to verify them.
    
    IMPORTANT: For speculative decoding to work correctly, both models should
    share the same tokenizer/vocabulary. If they don't, action token remapping
    is attempted but this only works for action tokens (last 256 tokens of vocab).
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        target_processor=None,
        gamma: int = 4,  # Number of draft tokens to propose at once
        use_cache: bool = True,
        temperature: float = 0.0,  # 0 = greedy/argmax
        n_action_bins: int = 256,  # Number of action bins (tokens)
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: OpenVLA model (larger, slower, more accurate)
            draft_model: MiniVLA model (smaller, faster)
            target_processor: HuggingFace processor for target model
            gamma: Number of tokens to speculate at each step
            use_cache: Whether to use KV caching
            temperature: Sampling temperature (0 = greedy/argmax)
            n_action_bins: Number of action token bins (typically 256)
        """
        self.target = target_model
        self.draft = draft_model
        self.target_processor = target_processor
        self.gamma = gamma
        self.use_cache = use_cache
        self.temperature = temperature
        self.n_action_bins = n_action_bins
        
        # Get device
        self.device = next(target_model.parameters()).device
        
        # Stats tracking
        self.stats = SpeculativeDecodingStats()
        
        # Check vocabulary compatibility and setup token mapping
        self._setup_token_mapping()
    
    def _setup_token_mapping(self):
        """Setup token mapping between draft and target vocabularies."""
        # Get vocabulary sizes
        # Target model (OpenVLA/HF style) - use ACTUAL embedding dimension, not vocab_size attribute
        # The embedding may be padded to "multiple of" for efficiency
        if hasattr(self.target, 'language_model') and hasattr(self.target.language_model, 'model'):
            # Actual embedding dimension (includes padding)
            self.target_logit_dim = self.target.language_model.model.embed_tokens.weight.shape[0]
        elif hasattr(self.target, 'get_output_embeddings'):
            self.target_logit_dim = self.target.get_output_embeddings().weight.shape[0]
        else:
            self.target_logit_dim = self.target.config.vocab_size
        
        # Also get the "logical" vocab size (without padding) for action token calculation
        if hasattr(self.target, 'vocab_size'):
            self.target_vocab_size = self.target.vocab_size
        elif hasattr(self.target, 'config') and hasattr(self.target.config, 'vocab_size'):
            self.target_vocab_size = self.target.config.vocab_size
        else:
            self.target_vocab_size = self.target_logit_dim
        
        # Draft model (Prismatic style)
        if hasattr(self.draft, 'llm_backbone'):
            draft_tokenizer = self.draft.llm_backbone.tokenizer
            # Qwen2 uses len(tokenizer) for full vocab including added tokens
            self.draft_vocab_size = len(draft_tokenizer) if hasattr(draft_tokenizer, '__len__') else draft_tokenizer.vocab_size
            # Get actual logit dimension from draft model
            if hasattr(self.draft.llm_backbone, 'llm') and hasattr(self.draft.llm_backbone.llm, 'lm_head'):
                self.draft_logit_dim = self.draft.llm_backbone.llm.lm_head.weight.shape[0]
            else:
                self.draft_logit_dim = self.draft_vocab_size
        else:
            self.draft_vocab_size = self.draft.config.vocab_size
            self.draft_logit_dim = self.draft_vocab_size
        
        # Check if vocabularies match
        self.vocab_compatible = (self.target_logit_dim == self.draft_logit_dim)
        
        # Compute action token ranges
        # Action tokens are the LAST n_action_bins tokens before padding
        # For target: use vocab_size (not padded logit_dim)
        self.target_action_start = self.target_vocab_size - self.n_action_bins
        self.draft_action_start = self.draft_vocab_size - self.n_action_bins
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Target vocab_size: {self.target_vocab_size}, logit_dim: {self.target_logit_dim}, action tokens: [{self.target_action_start}, {self.target_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Draft vocab_size: {self.draft_vocab_size}, logit_dim: {self.draft_logit_dim}, action tokens: [{self.draft_action_start}, {self.draft_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Vocabularies compatible: {self.vocab_compatible}")
        
        if not self.vocab_compatible:
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"WARNING: Vocabulary mismatch! Token remapping will be used for action tokens only.")
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"This may affect acceptance rates. Consider using models with matching tokenizers.")
        
        # DEBUG: Verify mapping with example tokens
        print("\033[38;2;100;200;255m[DEBUG] Token mapping verification examples:\033[0m")
        for bin_idx in [0, 127, 255]:  # First, middle, last action bins
            draft_tok = self.draft_action_start + bin_idx
            target_tok = self._draft_token_to_target(draft_tok)
            draft_bin = self._get_action_bin_from_draft_token(draft_tok)
            target_bin = self._get_action_bin_from_target_token(target_tok)
            print(f"  Bin {bin_idx}: draft_token={draft_tok} → target_token={target_tok} | draft_bin={draft_bin}, target_bin={target_bin} | {'✓' if draft_bin == target_bin else '✗'}")
    
    def _draft_token_to_target(self, draft_token_id: int) -> int:
        """Map a draft token ID to target vocabulary."""
        if self.vocab_compatible:
            return draft_token_id
        
        # Check if this is an action token (from end of draft vocabulary)
        if draft_token_id >= self.draft_action_start:
            # Map to corresponding action token in target vocabulary
            action_bin = draft_token_id - self.draft_action_start
            target_token = self.target_action_start + action_bin
            return target_token
        else:
            # Non-action token - this shouldn't happen during action generation
            # Return as-is but clamp to valid range
            return min(draft_token_id, self.target_vocab_size - 1)
    
    def _target_token_to_draft(self, target_token_id: int) -> int:
        """Map a target token ID to draft vocabulary."""
        if self.vocab_compatible:
            return target_token_id
        
        # Check if this is an action token
        if target_token_id >= self.target_action_start:
            action_bin = target_token_id - self.target_action_start
            draft_token = self.draft_action_start + action_bin
            return draft_token
        else:
            return min(target_token_id, self.draft_vocab_size - 1)
    
    def _remap_logits_draft_to_target(self, draft_logits: torch.Tensor, target_logit_dim: int = None) -> torch.Tensor:
        """
        Remap draft logits to target vocabulary space.
        Only remaps action tokens; non-action tokens get -inf.
        
        Args:
            draft_logits: Logits from draft model
            target_logit_dim: Actual dimension of target logits (may differ from vocab_size due to padding)
        """
        if self.vocab_compatible:
            return draft_logits
        
        # Use provided dimension or fall back to stored logit_dim
        if target_logit_dim is None:
            target_logit_dim = self.target_logit_dim
        
        # Create target-sized logits filled with -inf (use actual logit dimension, not vocab_size)
        target_logits = torch.full(
            (draft_logits.shape[0], target_logit_dim),
            float('-inf'),
            device=draft_logits.device,
            dtype=draft_logits.dtype
        )
        
        # Copy action token logits from draft to corresponding target positions
        # Draft action tokens: [draft_action_start, draft_vocab_size)
        # Target action tokens: [target_action_start, target_vocab_size)
        draft_action_logits = draft_logits[:, self.draft_action_start:self.draft_vocab_size]
        target_logits[:, self.target_action_start:self.target_vocab_size] = draft_action_logits
        
        return target_logits
    
    # =========================================================================
    # DEBUG: Token mapping verification methods
    # =========================================================================
    
    def _get_action_bin_from_draft_token(self, draft_token_id: int) -> int:
        """Get action bin index from draft token ID."""
        if draft_token_id >= self.draft_action_start:
            return draft_token_id - self.draft_action_start
        return -1  # Not an action token
    
    def _get_action_bin_from_target_token(self, target_token_id: int) -> int:
        """Get action bin index from target token ID."""
        if target_token_id >= self.target_action_start:
            return target_token_id - self.target_action_start
        return -1  # Not an action token
    
    def _get_continuous_action_from_bin(self, bin_idx: int) -> float:
        """Convert action bin index to continuous action value using target's bin centers."""
        if 0 <= bin_idx < len(self.target.bin_centers):
            return self.target.bin_centers[bin_idx]
        return float('nan')
    
    def _debug_token_mapping(self, draft_token_id: int, target_token_id: int, prefix: str = ""):
        """Debug print showing token mapping verification."""
        draft_bin = self._get_action_bin_from_draft_token(draft_token_id)
        target_bin = self._get_action_bin_from_target_token(target_token_id)
        
        draft_action = self._get_continuous_action_from_bin(draft_bin)
        target_action = self._get_continuous_action_from_bin(target_bin)
        
        match_status = "✓ MATCH" if draft_bin == target_bin else "✗ MISMATCH"
        
        print(f"\033[38;2;100;200;255m[DEBUG TOKEN MAP] {prefix}\033[0m")
        print(f"  Draft token:  {draft_token_id} → bin {draft_bin} → action {draft_action:.4f}")
        print(f"  Target token: {target_token_id} → bin {target_bin} → action {target_action:.4f}")
        print(f"  Bins match: {match_status}")
        
        return draft_bin == target_bin
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = SpeculativeDecodingStats()
    
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
    
    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        if self.temperature <= 0:
            # For greedy, use a very low temperature to approximate argmax
            return F.softmax(logits / 0.01, dim=-1)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _prepare_target_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the target (OpenVLA) model."""
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.target_processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        # IMPORTANT: Add the special empty token (29871) to input_ids if not present https://archive.is/20240927102623/https://medium.com/@manyi.yim/in-depth-understanding-of-llama-tokenizer-d91777025dab#selection-1151.5-1151.109
        # This is what OpenVLA's predict_action does - the action tokens come AFTER this token
            #if not torch.all(input_ids[:, -1] == 29871):
            # input_ids = torch.cat((input_ids, torch.Tensor([29871]).long()), dim=1)
            # generated_ids = self.generate(input_ids, max_new_tokens=action_dim, ...)
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"], torch.tensor([[29871]], device=self.device)),
                dim=1
            )
            # Also extend attention mask
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    (inputs["attention_mask"], torch.ones((1, 1), device=self.device, dtype=inputs["attention_mask"].dtype)),
                    dim=1
                )
        
        return inputs
    
    def _prepare_draft_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the draft (MiniVLA) model."""
        # Get prompt using draft model's prompt builder
        prompt_builder = self.draft.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        # Tokenize
        tokenizer = self.draft.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # Handle special token for Llama tokenizer
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], device=self.device)),
                    dim=1
                )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Process image
        image_transform = self.draft.vision_backbone.get_image_transform()
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    @torch.inference_mode()
    def predict_action_speculative(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key_target: str,
    ) -> Tuple[np.ndarray, SpeculativeDecodingStats]:
        """
        Generate action using speculative decoding.
        
        Args:
            image: PIL Image observation
            instruction: Task instruction string
            unnorm_key_target: Key for action un-normalization statistics of the target model
            
        Returns:
            Tuple of (unnormalized action array, decoding statistics)
        """
        # Reset per-call stats
        call_stats = SpeculativeDecodingStats()
        
        action_dim = self.target.get_action_dim(unnorm_key_target)
        
        
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target vocab size: {self.target.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target embedding size: {self.target.language_model.model.embed_tokens.weight.shape[0]}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Draft tokenizer vocab size: {self.draft.llm_backbone.tokenizer.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Action dim: {action_dim}")
        
        # Prepare inputs for both models
        target_inputs = self._prepare_target_inputs(image, instruction)
        draft_input_ids, draft_attention_mask, draft_pixel_values = self._prepare_draft_inputs(image, instruction)
        
        # Cast draft inputs to appropriate dtype
        autocast_dtype = self.draft.llm_backbone.half_precision_dtype
        
        # Initialize caches
        target_cache = None
        draft_cache = None
        
        generated_token_ids = []
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # === Initial forward pass to get KV cache and initial logits ===
            # NOTE: After adding token 29871 to input, the model should output action tokens directly
            
            # Target model initial forward (prefill)
            target_out = self.target(
                **target_inputs,
                past_key_values=None,
                use_cache=self.use_cache,
            )
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # DEBUG: Check what the target model wants to output first
            top_target_token = torch.argmax(target_logits, dim=-1).item()
            top_target_bin = self._get_action_bin_from_target_token(top_target_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, target top token:\033[0m")
            print(f"  top_token={top_target_token} → bin={top_target_bin} → is_action_token={top_target_bin >= 0}")
            
            # Draft model initial forward (prefill)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_out = self.draft(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    pixel_values=draft_pixel_values,
                    past_key_values=None,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # DEBUG: Check what the draft model wants to output first
            top_draft_token = torch.argmax(draft_logits, dim=-1).item()
            top_draft_bin = self._get_action_bin_from_draft_token(top_draft_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, draft top token:\033[0m")
            print(f"  top_token={top_draft_token} → bin={top_draft_bin} → is_action_token={top_draft_bin >= 0}")
            
            # === Main speculative decoding loop ===
            # We start directly with drafting - no need to sample a first token separately
            while len(generated_token_ids) < action_dim:
                # Determine how many tokens to speculate
                gamma = min(self.gamma, action_dim - len(generated_token_ids))
                
                # Generate gamma draft tokens
                draft_tokens = []
                draft_probs_list = []
                
                current_draft_cache = draft_cache
                current_draft_logits = draft_logits
                
                for _ in range(gamma):
                    draft_probs = self._get_probs(current_draft_logits)
                    draft_token = self._sample_token(current_draft_logits)
                    
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(draft_probs)
                    
                    # Advance draft model
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=draft_token.to(self.device),
                            past_key_values=current_draft_cache,
                            use_cache=self.use_cache,
                        )
                    current_draft_cache = draft_step.past_key_values
                    current_draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                
                call_stats.total_draft_tokens_proposed += gamma
                
                # Map draft tokens to target vocab space for verification
                draft_token_ids_target = []
                print(f"\033[38;2;100;200;255m[DEBUG] Mapping {gamma} draft tokens to target vocab:\033[0m")
                for idx, dt in enumerate(draft_tokens):
                    draft_id = dt.item()
                    target_id = self._draft_token_to_target(draft_id)
                    draft_token_ids_target.append(target_id)
                    
                    # Verify mapping preserves action bin
                    draft_bin = self._get_action_bin_from_draft_token(draft_id)
                    target_bin = self._get_action_bin_from_target_token(target_id)
                    draft_action = self._get_continuous_action_from_bin(draft_bin)
                    target_action = self._get_continuous_action_from_bin(target_bin)
                    match = "✓" if draft_bin == target_bin else "✗ MISMATCH!"
                    print(f"  [{idx}] draft_tok={draft_id} → target_tok={target_id} | bin: {draft_bin}→{target_bin} | action: {draft_action:.4f}→{target_action:.4f} {match}")
                
                # Verify with target model - run all gamma tokens through
                target_cache_for_verify = target_cache
                target_logits_list = []
                
                for i in range(gamma):
                    target_token_input = torch.tensor([[draft_token_ids_target[i]]], device=self.device)
                    target_step = self.target(
                        input_ids=target_token_input,
                        past_key_values=target_cache_for_verify,
                        use_cache=self.use_cache,
                    )
                    target_cache_for_verify = target_step.past_key_values
                    target_logits_list.append(target_step.logits[:, -1:, :])
                
                call_stats.total_target_forward_passes += gamma
                
                # Stack target logits
                target_logits_batch = torch.cat(target_logits_list, dim=1)  # [1, gamma, actual_vocab_dim]
                target_probs_batch = self._get_probs(target_logits_batch)
                                
                # Get actual target logit dimension from the output
                actual_target_logit_dim = target_logits_batch.shape[-1]
                
                # Remap draft probs to target vocab space for comparison
                # Use actual target logit dimension to ensure tensor size match
                draft_probs_remapped = [self._remap_logits_draft_to_target(dp, actual_target_logit_dim) for dp in draft_probs_list]
                draft_probs_remapped = [self._get_probs(dp) for dp in draft_probs_remapped]
                
                print(f"\033[38;2;100;200;255m[SRP] -> \033[0m", f"[DEBUG] Target probs batch: {target_probs_batch}")
                print(f"\033[38;2;100;200;255m[SRP] -> \033[0m", f"[DEBUG] Draft probs list: {draft_probs_list}")

                
                # Rejection sampling loop
                n_accepted = 0
                for i in range(gamma):
                    draft_token_id_draft = draft_tokens[i].item()  # In draft vocab
                    draft_token_id_target = draft_token_ids_target[i]  # Mapped to target vocab
                    
                    draft_prob_remapped = draft_probs_remapped[i]
                    target_prob = target_probs_batch[:, i, :]
                    
                    # Get probability of the token under both models (in target vocab space)
                    p_target = target_prob[0, draft_token_id_target].item()
                    p_draft = draft_prob_remapped[0, draft_token_id_target].item()
                    
                    # Rejection sampling
                    if p_draft > 0:
                        acceptance_prob = min(1.0, p_target / p_draft)
                    else:
                        acceptance_prob = 1.0 if p_target > 0 else 0.0
                    
                    if torch.rand(1).item() < acceptance_prob:
                        # Accept this token (store in target vocab space)
                        accepted_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                        accepted_action = self._get_continuous_action_from_bin(accepted_bin)
                        print(f"\033[38;2;0;255;0m[ACCEPT]\033[0m token[{i}]: target_tok={draft_token_id_target} → bin={accepted_bin} → action={accepted_action:.4f} | p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        if len(generated_token_ids) >= action_dim:
                            print(f"\033[38;2;255;165;0m[SRP] -> \033[0m Generated {len(generated_token_ids)}/{action_dim} tokens, done.")
                            break
                    else:
                        # Reject - sample from adjusted distribution
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        if adjusted_probs.sum() > 0:
                            corrected_token = torch.multinomial(adjusted_probs, num_samples=1)
                        else:
                            corrected_token = self._sample_token(target_prob.unsqueeze(0))
                        
                        corrected_token_id = int(corrected_token.item())
                        rejected_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                        corrected_bin = self._get_action_bin_from_target_token(corrected_token_id)
                        rejected_action = self._get_continuous_action_from_bin(rejected_bin)
                        corrected_action = self._get_continuous_action_from_bin(corrected_bin)
                        bin_diff = abs(rejected_bin - corrected_bin) if rejected_bin >= 0 and corrected_bin >= 0 else -1
                        
                        print(f"\033[38;2;255;100;100m[REJECT]\033[0m token[{i}]: p_target={p_target:.4f} < p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        print(f"  Rejected:  target_tok={draft_token_id_target} → bin={rejected_bin} → action={rejected_action:.4f}")
                        print(f"  Corrected: target_tok={corrected_token_id} → bin={corrected_bin} → action={corrected_action:.4f}")
                        print(f"  Bin difference: {bin_diff} | Action difference: {abs(rejected_action - corrected_action):.4f}")
                        
                        # Store corrected token (already in target vocab space)
                        generated_token_ids.append(corrected_token_id)
                        call_stats.total_tokens_generated += 1
                        n_accepted = i  # Number of accepted tokens (before this rejection)
                        break
                                    
                # Update caches after acceptance/rejection
                if n_accepted == gamma and len(generated_token_ids) < action_dim:
                    # All accepted - need to sample one more from target
                    print(f"\033[38;2;0;255;0m[ALL ACCEPTED]\033[0m All {gamma} draft tokens accepted! Sampling bonus token from target...")
                    target_cache = target_cache_for_verify
                    target_logits = target_logits_list[-1].squeeze(1)
                    
                    # Sample additional token from target (in target vocab space)
                    bonus_token_target = self._sample_token(target_logits)
                    bonus_token_id_target = int(bonus_token_target.item())
                    bonus_bin = self._get_action_bin_from_target_token(bonus_token_id_target)
                    bonus_action = self._get_continuous_action_from_bin(bonus_bin)
                    print(f"  Bonus token: target_tok={bonus_token_id_target} → bin={bonus_bin} → action={bonus_action:.4f}")
                    generated_token_ids.append(bonus_token_id_target)
                    call_stats.total_tokens_generated += 1
                    
                    # Update target cache
                    target_step = self.target(
                        input_ids=bonus_token_target,
                        past_key_values=target_cache,
                        use_cache=self.use_cache,
                    )
                    target_cache = target_step.past_key_values
                    target_logits = target_step.logits[:, -1, :]
                    call_stats.total_target_forward_passes += 1
                    
                    # Map bonus token to draft vocab and update draft cache
                    bonus_token_id_draft = self._target_token_to_draft(bonus_token_id_target)
                    bonus_draft_bin = self._get_action_bin_from_draft_token(bonus_token_id_draft)
                    print(f"  Mapped to draft: draft_tok={bonus_token_id_draft} → bin={bonus_draft_bin} {'✓' if bonus_bin == bonus_draft_bin else '✗ MISMATCH!'}")
                    bonus_token_draft = torch.tensor([[bonus_token_id_draft]], device=self.device)
                    
                    draft_cache = current_draft_cache
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=bonus_token_draft,
                            past_key_values=draft_cache,
                            use_cache=self.use_cache,
                        )
                    draft_cache = draft_step.past_key_values
                    draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                    
                else:
                    # Some tokens rejected - prune caches
                    tokens_to_discard = gamma - n_accepted
                    if tokens_to_discard > 0 and self.use_cache:
                        # We need to prune and resync
                        # Use the cache state after the accepted tokens
                        target_cache = target_cache_for_verify
                        if tokens_to_discard > 0:
                            target_cache = prune_cache(target_cache, tokens_to_discard)
                        
                        # Rebuild draft cache
                        draft_cache = prune_cache(current_draft_cache, gamma - n_accepted)
                    
                    # Get logits for next round
                    if len(generated_token_ids) < action_dim:
                        # Last token is in target vocab space
                        last_token_id_target = generated_token_ids[-1]
                        last_token_target = torch.tensor([[last_token_id_target]], device=self.device)
                        
                        target_step = self.target(
                            input_ids=last_token_target,
                            past_key_values=target_cache,
                            use_cache=self.use_cache,
                        )
                        target_cache = target_step.past_key_values
                        target_logits = target_step.logits[:, -1, :]
                        call_stats.total_target_forward_passes += 1
                        
                        # Map to draft vocab for draft model
                        last_token_id_draft = self._target_token_to_draft(last_token_id_target)
                        last_token_draft = torch.tensor([[last_token_id_draft]], device=self.device)
                        
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                            draft_step = self.draft(
                                input_ids=last_token_draft,
                                past_key_values=draft_cache,
                                use_cache=self.use_cache,
                            )
                        draft_cache = draft_step.past_key_values
                        draft_logits = draft_step.logits[:, -1, :]
                        call_stats.total_draft_forward_passes += 1
            
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Call stats: {call_stats}")
            
        # Decode tokens to actions
        predicted_action_token_ids = np.array(generated_token_ids[:action_dim], dtype=np.int64)
        
        # DEBUG: Show all generated tokens with their action values
        print(f"\033[38;2;100;200;255m[DEBUG] Final generated tokens summary:\033[0m")
        for dim_idx, tok_id in enumerate(predicted_action_token_ids):
            bin_idx = self._get_action_bin_from_target_token(int(tok_id))
            action_val = self._get_continuous_action_from_bin(bin_idx)
            print(f"  dim[{dim_idx}]: target_tok={tok_id} → bin={bin_idx} → normalized_action={action_val:.4f}")
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Predicted action token ids: {predicted_action_token_ids}")
        
        # Use target model's decoding (vocab_size - token_id approach)
        vocab_size = self.target.vocab_size
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.target.bin_centers.shape[0] - 1)
        normalized_actions = self.target.bin_centers[discretized_actions]
        
        # Un-normalize actions
        action_norm_stats = self.target.get_action_stats(unnorm_key_target)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # Update global stats
        self.stats.total_tokens_generated += call_stats.total_tokens_generated
        self.stats.total_draft_tokens_proposed += call_stats.total_draft_tokens_proposed
        self.stats.total_draft_tokens_accepted += call_stats.total_draft_tokens_accepted
        self.stats.total_target_forward_passes += call_stats.total_target_forward_passes
        self.stats.total_draft_forward_passes += call_stats.total_draft_forward_passes
        
        return actions, call_stats

# %%
class VLASpeculativeDecoderDDD:
    """
    Speculative decoding for VLA models.
    
    Uses a draft model (MiniVLA) to propose action tokens and a target model 
    (OpenVLA) to verify them.
    
    IMPORTANT: For speculative decoding to work correctly, both models should
    share the same tokenizer/vocabulary. If they don't, action token remapping
    is attempted but this only works for action tokens (last 256 tokens of vocab).
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        target_processor=None,
        gamma: int = 4,  # Number of draft tokens to propose at once
        use_cache: bool = True,
        temperature: float = 0.0,  # 0 = greedy/argmax
        n_action_bins: int = 256,  # Number of action bins (tokens)
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: OpenVLA model (larger, slower, more accurate)
            draft_model: MiniVLA model (smaller, faster)
            target_processor: HuggingFace processor for target model
            gamma: Number of tokens to speculate at each step
            use_cache: Whether to use KV caching
            temperature: Sampling temperature (0 = greedy/argmax)
            n_action_bins: Number of action token bins (typically 256)
        """
        self.target = target_model
        self.draft = draft_model
        self.target_processor = target_processor
        self.gamma = gamma
        self.use_cache = use_cache
        self.temperature = temperature
        self.n_action_bins = n_action_bins
        
        # Get device
        self.device = next(target_model.parameters()).device
        
        # Stats tracking
        self.stats = SpeculativeDecodingStats()
        
        # Check vocabulary compatibility and setup token mapping
        self._setup_token_mapping()
    
    def _setup_token_mapping(self):
        """Setup token mapping between draft and target vocabularies."""
        # Get vocabulary sizes
        # Target model (OpenVLA/HF style) - use ACTUAL embedding dimension, not vocab_size attribute
        # The embedding may be padded to "multiple of" for efficiency
        if hasattr(self.target, 'language_model') and hasattr(self.target.language_model, 'model'):
            # Actual embedding dimension (includes padding)
            self.target_logit_dim = self.target.language_model.model.embed_tokens.weight.shape[0]
        elif hasattr(self.target, 'get_output_embeddings'):
            self.target_logit_dim = self.target.get_output_embeddings().weight.shape[0]
        else:
            self.target_logit_dim = self.target.config.vocab_size
        
        # Also get the "logical" vocab size (without padding) for action token calculation
        if hasattr(self.target, 'vocab_size'):
            self.target_vocab_size = self.target.vocab_size
        elif hasattr(self.target, 'config') and hasattr(self.target.config, 'vocab_size'):
            self.target_vocab_size = self.target.config.vocab_size
        else:
            self.target_vocab_size = self.target_logit_dim
        
        # Draft model (Prismatic style)
        if hasattr(self.draft, 'llm_backbone'):
            draft_tokenizer = self.draft.llm_backbone.tokenizer
            # Qwen2 uses len(tokenizer) for full vocab including added tokens
            self.draft_vocab_size = len(draft_tokenizer) if hasattr(draft_tokenizer, '__len__') else draft_tokenizer.vocab_size
            # Get actual logit dimension from draft model
            if hasattr(self.draft.llm_backbone, 'llm') and hasattr(self.draft.llm_backbone.llm, 'lm_head'):
                self.draft_logit_dim = self.draft.llm_backbone.llm.lm_head.weight.shape[0]
            else:
                self.draft_logit_dim = self.draft_vocab_size
        else:
            self.draft_vocab_size = self.draft.config.vocab_size
            self.draft_logit_dim = self.draft_vocab_size
        
        # Check if vocabularies match
        self.vocab_compatible = (self.target_logit_dim == self.draft_logit_dim)
        
        # Compute action token ranges
        # Action tokens are the LAST n_action_bins tokens before padding
        # For target: use vocab_size (not padded logit_dim)
        self.target_action_start = self.target_vocab_size - self.n_action_bins
        self.draft_action_start = self.draft_vocab_size - self.n_action_bins
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Target vocab_size: {self.target_vocab_size}, logit_dim: {self.target_logit_dim}, action tokens: [{self.target_action_start}, {self.target_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Draft vocab_size: {self.draft_vocab_size}, logit_dim: {self.draft_logit_dim}, action tokens: [{self.draft_action_start}, {self.draft_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Vocabularies compatible: {self.vocab_compatible}")
        
        if not self.vocab_compatible:
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"WARNING: Vocabulary mismatch! Token remapping will be used for action tokens only.")
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"This may affect acceptance rates. Consider using models with matching tokenizers.")
        
        # DEBUG: Verify mapping with example tokens
        print("\033[38;2;100;200;255m[DEBUG] Token mapping verification examples:\033[0m")
        for bin_idx in [0, 127, 255]:  # First, middle, last action bins
            draft_tok = self.draft_action_start + bin_idx
            target_tok = self._draft_token_to_target(draft_tok)
            draft_bin = self._get_action_bin_from_draft_token(draft_tok)
            target_bin = self._get_action_bin_from_target_token(target_tok)
            print(f"  Bin {bin_idx}: draft_token={draft_tok} → target_token={target_tok} | draft_bin={draft_bin}, target_bin={target_bin} | {'✓' if draft_bin == target_bin else '✗'}")
    
    def _draft_token_to_target(self, draft_token_id: int) -> int:
        """Map a draft token ID to target vocabulary."""
        if self.vocab_compatible:
            return draft_token_id
        
        # Check if this is an action token (from end of draft vocabulary)
        if draft_token_id >= self.draft_action_start:
            # Map to corresponding action token in target vocabulary
            action_bin = draft_token_id - self.draft_action_start
            target_token = self.target_action_start + action_bin
            return target_token
        else:
            # Non-action token - this shouldn't happen during action generation
            # Return as-is but clamp to valid range
            return min(draft_token_id, self.target_vocab_size - 1)
    
    def _target_token_to_draft(self, target_token_id: int) -> int:
        """Map a target token ID to draft vocabulary."""
        if self.vocab_compatible:
            return target_token_id
        
        # Check if this is an action token
        if target_token_id >= self.target_action_start:
            action_bin = target_token_id - self.target_action_start
            draft_token = self.draft_action_start + action_bin
            return draft_token
        else:
            return min(target_token_id, self.draft_vocab_size - 1)
    
    def _remap_logits_draft_to_target(self, draft_logits: torch.Tensor, target_logit_dim: int = None) -> torch.Tensor:
        """
        Remap draft logits to target vocabulary space.
        Only remaps action tokens; non-action tokens get -inf.
        
        Args:
            draft_logits: Logits from draft model
            target_logit_dim: Actual dimension of target logits (may differ from vocab_size due to padding)
        """
        if self.vocab_compatible:
            return draft_logits
        
        # Use provided dimension or fall back to stored logit_dim
        if target_logit_dim is None:
            target_logit_dim = self.target_logit_dim
        
        # Create target-sized logits filled with -inf (use actual logit dimension, not vocab_size)
        target_logits = torch.full(
            (draft_logits.shape[0], target_logit_dim),
            float('-inf'),
            device=draft_logits.device,
            dtype=draft_logits.dtype
        )
        
        # Copy action token logits from draft to corresponding target positions
        # Draft action tokens: [draft_action_start, draft_vocab_size)
        # Target action tokens: [target_action_start, target_vocab_size)
        draft_action_logits = draft_logits[:, self.draft_action_start:self.draft_vocab_size]
        target_logits[:, self.target_action_start:self.target_vocab_size] = draft_action_logits
        
        return target_logits
    
    # =========================================================================
    # DEBUG: Token mapping verification methods
    # =========================================================================
    
    def _get_action_bin_from_draft_token(self, draft_token_id: int) -> int:
        """Get action bin index from draft token ID."""
        if draft_token_id >= self.draft_action_start:
            return draft_token_id - self.draft_action_start
        return -1  # Not an action token
    
    def _get_action_bin_from_target_token(self, target_token_id: int) -> int:
        """Get action bin index from target token ID."""
        if target_token_id >= self.target_action_start:
            return target_token_id - self.target_action_start
        return -1  # Not an action token
    
    def _get_continuous_action_from_bin(self, bin_idx: int) -> float:
        """Convert action bin index to continuous action value using target's bin centers."""
        if 0 <= bin_idx < len(self.target.bin_centers):
            return self.target.bin_centers[bin_idx]
        return float('nan')
    
    def _debug_token_mapping(self, draft_token_id: int, target_token_id: int, prefix: str = ""):
        """Debug print showing token mapping verification."""
        draft_bin = self._get_action_bin_from_draft_token(draft_token_id)
        target_bin = self._get_action_bin_from_target_token(target_token_id)
        
        draft_action = self._get_continuous_action_from_bin(draft_bin)
        target_action = self._get_continuous_action_from_bin(target_bin)
        
        match_status = "✓ MATCH" if draft_bin == target_bin else "✗ MISMATCH"
        
        print(f"\033[38;2;100;200;255m[DEBUG TOKEN MAP] {prefix}\033[0m")
        print(f"  Draft token:  {draft_token_id} → bin {draft_bin} → action {draft_action:.4f}")
        print(f"  Target token: {target_token_id} → bin {target_bin} → action {target_action:.4f}")
        print(f"  Bins match: {match_status}")
        
        return draft_bin == target_bin
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = SpeculativeDecodingStats()
    
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
    
    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        if self.temperature <= 0:
            # For greedy, use a very low temperature to approximate argmax
            return F.softmax(logits / 0.01, dim=-1)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _prepare_target_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the target (OpenVLA) model."""
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.target_processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        # IMPORTANT: Add the special empty token (29871) to input_ids if not present
        # This is what OpenVLA's predict_action does - the action tokens come AFTER this token
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"], torch.tensor([[29871]], device=self.device)),
                dim=1
            )
            # Also extend attention mask
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    (inputs["attention_mask"], torch.ones((1, 1), device=self.device, dtype=inputs["attention_mask"].dtype)),
                    dim=1
                )
        
        return inputs
    
    def _prepare_draft_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the draft (MiniVLA) model."""
        # Get prompt using draft model's prompt builder
        prompt_builder = self.draft.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        # Tokenize
        tokenizer = self.draft.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # Handle special token for Llama tokenizer
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], device=self.device)),
                    dim=1
                )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Process image
        image_transform = self.draft.vision_backbone.get_image_transform()
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    @torch.inference_mode()
    def predict_action_speculative(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key_target: str,
    ) -> Tuple[np.ndarray, SpeculativeDecodingStats]:
        """
        Generate action using speculative decoding.
        
        Args:
            image: PIL Image observation
            instruction: Task instruction string
            unnorm_key_target: Key for action un-normalization statistics of the target model
            
        Returns:
            Tuple of (unnormalized action array, decoding statistics)
        """
        # Reset per-call stats
        call_stats = SpeculativeDecodingStats()
        
        action_dim = self.target.get_action_dim(unnorm_key_target)
        
        
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target vocab size: {self.target.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target embedding size: {self.target.language_model.model.embed_tokens.weight.shape[0]}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Draft tokenizer vocab size: {self.draft.llm_backbone.tokenizer.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Action dim: {action_dim}")
        
        # Prepare inputs for both models
        target_inputs = self._prepare_target_inputs(image, instruction)
        draft_input_ids, draft_attention_mask, draft_pixel_values = self._prepare_draft_inputs(image, instruction)
        
        # Cast draft inputs to appropriate dtype
        autocast_dtype = self.draft.llm_backbone.half_precision_dtype
        
        # Initialize caches
        target_cache = None
        draft_cache = None
        
        generated_token_ids = []
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # === Initial forward pass to get KV cache and initial logits ===
            # NOTE: After adding token 29871 to input, the model should output action tokens directly
            
            # Target model initial forward (prefill)
            target_out = self.target(
                **target_inputs,
                past_key_values=None,
                use_cache=self.use_cache,
            )
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # DEBUG: Check what the target model wants to output first
            top_target_token = torch.argmax(target_logits, dim=-1).item()
            top_target_bin = self._get_action_bin_from_target_token(top_target_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, target top token:\033[0m")
            print(f"  top_token={top_target_token} → bin={top_target_bin} → is_action_token={top_target_bin >= 0}")
            
            # Draft model initial forward (prefill)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_out = self.draft(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    pixel_values=draft_pixel_values,
                    past_key_values=None,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # DEBUG: Check what the draft model wants to output first
            top_draft_token = torch.argmax(draft_logits, dim=-1).item()
            top_draft_bin = self._get_action_bin_from_draft_token(top_draft_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, draft top token:\033[0m")
            print(f"  top_token={top_draft_token} → bin={top_draft_bin} → is_action_token={top_draft_bin >= 0}")
            
            # === Main speculative decoding loop ===
            # We start directly with drafting - no need to sample a first token separately
            while len(generated_token_ids) < action_dim:
                # Determine how many tokens to speculate
                gamma = min(self.gamma, action_dim - len(generated_token_ids))
                
                # Generate gamma draft tokens
                draft_tokens = []
                draft_probs_list = []
                
                current_draft_cache = draft_cache
                current_draft_logits = draft_logits
                
                for _ in range(gamma):
                    draft_probs = self._get_probs(current_draft_logits)
                    draft_token = self._sample_token(current_draft_logits)
                    
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(draft_probs)
                    
                    # Advance draft model
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=draft_token.to(self.device),
                            past_key_values=current_draft_cache,
                            use_cache=self.use_cache,
                        )
                    current_draft_cache = draft_step.past_key_values
                    current_draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                
                call_stats.total_draft_tokens_proposed += gamma
                
                # Map draft tokens to target vocab space for verification
                draft_token_ids_target = []
                print(f"\033[38;2;100;200;255m[DEBUG] Mapping {gamma} draft tokens to target vocab:\033[0m")
                for idx, dt in enumerate(draft_tokens):
                    draft_id = dt.item()
                    target_id = self._draft_token_to_target(draft_id)
                    draft_token_ids_target.append(target_id)
                    
                    # Verify mapping preserves action bin
                    draft_bin = self._get_action_bin_from_draft_token(draft_id)
                    target_bin = self._get_action_bin_from_target_token(target_id)
                    draft_action = self._get_continuous_action_from_bin(draft_bin)
                    target_action = self._get_continuous_action_from_bin(target_bin)
                    match = "✓" if draft_bin == target_bin else "✗ MISMATCH!"
                    print(f"  [{idx}] draft_tok={draft_id} → target_tok={target_id} | bin: {draft_bin}→{target_bin} | action: {draft_action:.4f}→{target_action:.4f} {match}")
                
                # Verify with target model
                # IMPORTANT: We need to collect logits BEFORE feeding each token
                # - target_logits (from prefill or last step) is used to evaluate draft_token[0]
                # - logits after feeding token[0] is used to evaluate draft_token[1]
                # etc.
                
                target_cache_for_verify = target_cache
                # Start with current target_logits for evaluating first draft token
                target_logits_for_verification = [target_logits.unsqueeze(1)]  # [1, 1, vocab]
                last_target_logits = None  # Will hold logits after feeding last token
                
                for i in range(gamma):
                    target_token_input = torch.tensor([[draft_token_ids_target[i]]], device=self.device)
                    target_step = self.target(
                        input_ids=target_token_input,
                        past_key_values=target_cache_for_verify,
                        use_cache=self.use_cache,
                    )
                    target_cache_for_verify = target_step.past_key_values
                    last_target_logits = target_step.logits[:, -1, :]  # Always save last logits
                    # Store logits for evaluating the NEXT token (if there is one)
                    if i < gamma - 1:
                        target_logits_for_verification.append(target_step.logits[:, -1:, :])
                
                call_stats.total_target_forward_passes += gamma
                
                # Stack target logits - now target_logits_for_verification[i] evaluates draft_token[i]
                target_logits_batch = torch.cat(target_logits_for_verification, dim=1)  # [1, gamma, actual_vocab_dim]
                target_probs_batch = self._get_probs(target_logits_batch)
                
                # DEBUG: Show which token the target actually wants at each position
                print(f"\033[38;2;100;200;255m[DEBUG] Target's preferred tokens at each position:\033[0m")
                for i in range(gamma):
                    top_tok = torch.argmax(target_logits_batch[0, i, :]).item()
                    top_bin = self._get_action_bin_from_target_token(top_tok)
                    draft_tok = draft_token_ids_target[i]
                    draft_bin = self._get_action_bin_from_target_token(draft_tok)
                    p_top = target_probs_batch[0, i, top_tok].item()
                    p_draft = target_probs_batch[0, i, draft_tok].item()
                    print(f"  pos[{i}]: target_wants={top_tok}(bin={top_bin}, p={p_top:.4f}) | draft_proposed={draft_tok}(bin={draft_bin}, p={p_draft:.4f})")
                
                # Get actual target logit dimension from the output
                actual_target_logit_dim = target_logits_batch.shape[-1]
                
                # Remap draft probs to target vocab space for comparison
                # Use actual target logit dimension to ensure tensor size match
                draft_probs_remapped = [self._remap_logits_draft_to_target(dp, actual_target_logit_dim) for dp in draft_probs_list]
                draft_probs_remapped = [self._get_probs(dp) for dp in draft_probs_remapped]
                
                # Rejection sampling loop
                n_accepted = 0
                for i in range(gamma):
                    draft_token_id_draft = draft_tokens[i].item()  # In draft vocab
                    draft_token_id_target = draft_token_ids_target[i]  # Mapped to target vocab
                    
                    draft_prob_remapped = draft_probs_remapped[i]
                    target_prob = target_probs_batch[:, i, :]
                    
                    # Get probability of the token under both models (in target vocab space)
                    p_target = target_prob[0, draft_token_id_target].item()
                    p_draft = draft_prob_remapped[0, draft_token_id_target].item()
                    
                    # Rejection sampling
                    if p_draft > 0:
                        acceptance_prob = min(1.0, p_target / p_draft)
                    else:
                        acceptance_prob = 1.0 if p_target > 0 else 0.0
                    
                    if torch.rand(1).item() < acceptance_prob:
                        # Accept this token (store in target vocab space)
                        accepted_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                        accepted_action = self._get_continuous_action_from_bin(accepted_bin)
                        print(f"\033[38;2;0;255;0m[ACCEPT]\033[0m token[{i}]: target_tok={draft_token_id_target} → bin={accepted_bin} → action={accepted_action:.4f} | p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        if len(generated_token_ids) >= action_dim:
                            print(f"\033[38;2;255;165;0m[SRP] -> \033[0m Generated {len(generated_token_ids)}/{action_dim} tokens, done.")
                            break
                    else:
                        # Reject - sample from adjusted distribution
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        if adjusted_probs.sum() > 0:
                            corrected_token = torch.multinomial(adjusted_probs, num_samples=1)
                        else:
                            corrected_token = self._sample_token(target_prob.unsqueeze(0))
                        
                        corrected_token_id = int(corrected_token.item())
                        rejected_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                        corrected_bin = self._get_action_bin_from_target_token(corrected_token_id)
                        rejected_action = self._get_continuous_action_from_bin(rejected_bin)
                        corrected_action = self._get_continuous_action_from_bin(corrected_bin)
                        bin_diff = abs(rejected_bin - corrected_bin) if rejected_bin >= 0 and corrected_bin >= 0 else -1
                        
                        print(f"\033[38;2;255;100;100m[REJECT]\033[0m token[{i}]: p_target={p_target:.4f} < p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        print(f"  Rejected:  target_tok={draft_token_id_target} → bin={rejected_bin} → action={rejected_action:.4f}")
                        print(f"  Corrected: target_tok={corrected_token_id} → bin={corrected_bin} → action={corrected_action:.4f}")
                        print(f"  Bin difference: {bin_diff} | Action difference: {abs(rejected_action - corrected_action):.4f}")
                        
                        # Store corrected token (already in target vocab space)
                        generated_token_ids.append(corrected_token_id)
                        call_stats.total_tokens_generated += 1
                        n_accepted = i  # Number of accepted tokens (before this rejection)
                        break
                                    
                # Update caches after acceptance/rejection
                if n_accepted == gamma and len(generated_token_ids) < action_dim:
                    # All accepted - need to sample one more from target
                    print(f"\033[38;2;0;255;0m[ALL ACCEPTED]\033[0m All {gamma} draft tokens accepted! Sampling bonus token from target...")
                    target_cache = target_cache_for_verify
                    target_logits = last_target_logits  # Logits after feeding all gamma tokens
                    
                    # Sample additional token from target (in target vocab space)
                    bonus_token_target = self._sample_token(target_logits)
                    bonus_token_id_target = int(bonus_token_target.item())
                    bonus_bin = self._get_action_bin_from_target_token(bonus_token_id_target)
                    bonus_action = self._get_continuous_action_from_bin(bonus_bin)
                    print(f"  Bonus token: target_tok={bonus_token_id_target} → bin={bonus_bin} → action={bonus_action:.4f}")
                    generated_token_ids.append(bonus_token_id_target)
                    call_stats.total_tokens_generated += 1
                    
                    # Update target cache
                    target_step = self.target(
                        input_ids=bonus_token_target,
                        past_key_values=target_cache,
                        use_cache=self.use_cache,
                    )
                    target_cache = target_step.past_key_values
                    target_logits = target_step.logits[:, -1, :]
                    call_stats.total_target_forward_passes += 1
                    
                    # Map bonus token to draft vocab and update draft cache
                    bonus_token_id_draft = self._target_token_to_draft(bonus_token_id_target)
                    bonus_draft_bin = self._get_action_bin_from_draft_token(bonus_token_id_draft)
                    print(f"  Mapped to draft: draft_tok={bonus_token_id_draft} → bin={bonus_draft_bin} {'✓' if bonus_bin == bonus_draft_bin else '✗ MISMATCH!'}")
                    bonus_token_draft = torch.tensor([[bonus_token_id_draft]], device=self.device)
                    
                    draft_cache = current_draft_cache
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=bonus_token_draft,
                            past_key_values=draft_cache,
                            use_cache=self.use_cache,
                        )
                    draft_cache = draft_step.past_key_values
                    draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                    
                else:
                    # Some tokens rejected - prune caches
                    tokens_to_discard = gamma - n_accepted
                    if tokens_to_discard > 0 and self.use_cache:
                        # We need to prune and resync
                        # Use the cache state after the accepted tokens
                        target_cache = target_cache_for_verify
                        if tokens_to_discard > 0:
                            target_cache = prune_cache(target_cache, tokens_to_discard)
                        
                        # Rebuild draft cache
                        draft_cache = prune_cache(current_draft_cache, gamma - n_accepted)
                    
                    # Get logits for next round
                    if len(generated_token_ids) < action_dim:
                        # Last token is in target vocab space
                        last_token_id_target = generated_token_ids[-1]
                        last_token_target = torch.tensor([[last_token_id_target]], device=self.device)
                        
                        target_step = self.target(
                            input_ids=last_token_target,
                            past_key_values=target_cache,
                            use_cache=self.use_cache,
                        )
                        target_cache = target_step.past_key_values
                        target_logits = target_step.logits[:, -1, :]
                        call_stats.total_target_forward_passes += 1
                        
                        # Map to draft vocab for draft model
                        last_token_id_draft = self._target_token_to_draft(last_token_id_target)
                        last_token_draft = torch.tensor([[last_token_id_draft]], device=self.device)
                        
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                            draft_step = self.draft(
                                input_ids=last_token_draft,
                                past_key_values=draft_cache,
                                use_cache=self.use_cache,
                            )
                        draft_cache = draft_step.past_key_values
                        draft_logits = draft_step.logits[:, -1, :]
                        call_stats.total_draft_forward_passes += 1
            
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Call stats: {call_stats}")
            
        # Decode tokens to actions
        predicted_action_token_ids = np.array(generated_token_ids[:action_dim], dtype=np.int64)
        
        # DEBUG: Show all generated tokens with their action values
        print(f"\033[38;2;100;200;255m[DEBUG] Final generated tokens summary:\033[0m")
        for dim_idx, tok_id in enumerate(predicted_action_token_ids):
            bin_idx = self._get_action_bin_from_target_token(int(tok_id))
            action_val = self._get_continuous_action_from_bin(bin_idx)
            print(f"  dim[{dim_idx}]: target_tok={tok_id} → bin={bin_idx} → normalized_action={action_val:.4f}")
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Predicted action token ids: {predicted_action_token_ids}")
        
        # Use target model's decoding (vocab_size - token_id approach)
        vocab_size = self.target.vocab_size
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.target.bin_centers.shape[0] - 1)
        normalized_actions = self.target.bin_centers[discretized_actions]
        
        # Un-normalize actions
        action_norm_stats = self.target.get_action_stats(unnorm_key_target)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # Update global stats
        self.stats.total_tokens_generated += call_stats.total_tokens_generated
        self.stats.total_draft_tokens_proposed += call_stats.total_draft_tokens_proposed
        self.stats.total_draft_tokens_accepted += call_stats.total_draft_tokens_accepted
        self.stats.total_target_forward_passes += call_stats.total_target_forward_passes
        self.stats.total_draft_forward_passes += call_stats.total_draft_forward_passes
        
        return actions, call_stats

# %%
class VLASpeculativeDecoderDDDR:
    """
    Speculative decoding for VLA models.
    
    Uses a draft model (MiniVLA) to propose action tokens and a target model 
    (OpenVLA) to verify them.
    
    IMPORTANT: For speculative decoding to work correctly, both models should
    share the same tokenizer/vocabulary. If they don't, action token remapping
    is attempted but this only works for action tokens (last 256 tokens of vocab).
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        target_processor=None,
        gamma: int = 4,  # Number of draft tokens to propose at once
        use_cache: bool = True,
        temperature: float = 0.0,  # 0 = greedy/argmax
        n_action_bins: int = 256,  # Number of action bins (tokens)
        relaxed_acceptance_r: int = 0,  # Relaxed acceptance radius (0 = standard spec dec)
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: OpenVLA model (larger, slower, more accurate)
            draft_model: MiniVLA model (smaller, faster)
            target_processor: HuggingFace processor for target model
            gamma: Number of tokens to speculate at each step
            use_cache: Whether to use KV caching
            temperature: Sampling temperature (0 = greedy/argmax)
            n_action_bins: Number of action token bins (typically 256)
            relaxed_acceptance_r: Relaxed acceptance radius. If the draft token's bin
                is within r bins of the target's preferred bin, accept it.
                Set to 0 for standard speculative decoding behavior.
        """
        self.target = target_model
        self.draft = draft_model
        self.target_processor = target_processor
        self.gamma = gamma
        self.use_cache = use_cache
        self.temperature = temperature
        self.n_action_bins = n_action_bins
        self.relaxed_acceptance_r = relaxed_acceptance_r
        
        # Get device
        self.device = next(target_model.parameters()).device
        
        # Stats tracking
        self.stats = SpeculativeDecodingStats()
        
        # Check vocabulary compatibility and setup token mapping
        self._setup_token_mapping()
    
    def _setup_token_mapping(self):
        """Setup token mapping between draft and target vocabularies."""
        # Get vocabulary sizes
        # Target model (OpenVLA/HF style) - use ACTUAL embedding dimension, not vocab_size attribute
        # The embedding may be padded to "multiple of" for efficiency
        if hasattr(self.target, 'language_model') and hasattr(self.target.language_model, 'model'):
            # Actual embedding dimension (includes padding)
            self.target_logit_dim = self.target.language_model.model.embed_tokens.weight.shape[0]
        elif hasattr(self.target, 'get_output_embeddings'):
            self.target_logit_dim = self.target.get_output_embeddings().weight.shape[0]
        else:
            self.target_logit_dim = self.target.config.vocab_size
        
        # Also get the "logical" vocab size (without padding) for action token calculation
        if hasattr(self.target, 'vocab_size'):
            self.target_vocab_size = self.target.vocab_size
        elif hasattr(self.target, 'config') and hasattr(self.target.config, 'vocab_size'):
            self.target_vocab_size = self.target.config.vocab_size
        else:
            self.target_vocab_size = self.target_logit_dim
        
        # Draft model (Prismatic style)
        if hasattr(self.draft, 'llm_backbone'):
            draft_tokenizer = self.draft.llm_backbone.tokenizer
            # Qwen2 uses len(tokenizer) for full vocab including added tokens
            self.draft_vocab_size = len(draft_tokenizer) if hasattr(draft_tokenizer, '__len__') else draft_tokenizer.vocab_size
            # Get actual logit dimension from draft model
            if hasattr(self.draft.llm_backbone, 'llm') and hasattr(self.draft.llm_backbone.llm, 'lm_head'):
                self.draft_logit_dim = self.draft.llm_backbone.llm.lm_head.weight.shape[0]
            else:
                self.draft_logit_dim = self.draft_vocab_size
        else:
            self.draft_vocab_size = self.draft.config.vocab_size
            self.draft_logit_dim = self.draft_vocab_size
        
        # Check if vocabularies match
        self.vocab_compatible = (self.target_logit_dim == self.draft_logit_dim)
        
        # Compute action token ranges
        # Action tokens are the LAST n_action_bins tokens before padding
        # For target: use vocab_size (not padded logit_dim)
        self.target_action_start = self.target_vocab_size - self.n_action_bins
        self.draft_action_start = self.draft_vocab_size - self.n_action_bins
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Target vocab_size: {self.target_vocab_size}, logit_dim: {self.target_logit_dim}, action tokens: [{self.target_action_start}, {self.target_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Draft vocab_size: {self.draft_vocab_size}, logit_dim: {self.draft_logit_dim}, action tokens: [{self.draft_action_start}, {self.draft_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Vocabularies compatible: {self.vocab_compatible}")
        
        if not self.vocab_compatible:
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"WARNING: Vocabulary mismatch! Token remapping will be used for action tokens only.")
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"This may affect acceptance rates. Consider using models with matching tokenizers.")
        
        # DEBUG: Verify mapping with example tokens
        print("\033[38;2;100;200;255m[DEBUG] Token mapping verification examples:\033[0m")
        for bin_idx in [0, 127, 255]:  # First, middle, last action bins
            draft_tok = self.draft_action_start + bin_idx
            target_tok = self._draft_token_to_target(draft_tok)
            draft_bin = self._get_action_bin_from_draft_token(draft_tok)
            target_bin = self._get_action_bin_from_target_token(target_tok)
            print(f"  Bin {bin_idx}: draft_token={draft_tok} → target_token={target_tok} | draft_bin={draft_bin}, target_bin={target_bin} | {'✓' if draft_bin == target_bin else '✗'}")
    
    def _draft_token_to_target(self, draft_token_id: int) -> int:
        """Map a draft token ID to target vocabulary."""
        if self.vocab_compatible:
            return draft_token_id
        
        # Check if this is an action token (from end of draft vocabulary)
        if draft_token_id >= self.draft_action_start:
            # Map to corresponding action token in target vocabulary
            action_bin = draft_token_id - self.draft_action_start
            target_token = self.target_action_start + action_bin
            return target_token
        else:
            # Non-action token - this shouldn't happen during action generation
            # Return as-is but clamp to valid range
            return min(draft_token_id, self.target_vocab_size - 1)
    
    def _target_token_to_draft(self, target_token_id: int) -> int:
        """Map a target token ID to draft vocabulary."""
        if self.vocab_compatible:
            return target_token_id
        
        # Check if this is an action token
        if target_token_id >= self.target_action_start:
            action_bin = target_token_id - self.target_action_start
            draft_token = self.draft_action_start + action_bin
            return draft_token
        else:
            return min(target_token_id, self.draft_vocab_size - 1)
    
    def _remap_logits_draft_to_target(self, draft_logits: torch.Tensor, target_logit_dim: int = None) -> torch.Tensor:
        """
        Remap draft logits to target vocabulary space.
        Only remaps action tokens; non-action tokens get -inf.
        
        Args:
            draft_logits: Logits from draft model
            target_logit_dim: Actual dimension of target logits (may differ from vocab_size due to padding)
        """
        if self.vocab_compatible:
            return draft_logits
        
        # Use provided dimension or fall back to stored logit_dim
        if target_logit_dim is None:
            target_logit_dim = self.target_logit_dim
        
        # Create target-sized logits filled with -inf (use actual logit dimension, not vocab_size)
        target_logits = torch.full(
            (draft_logits.shape[0], target_logit_dim),
            float('-inf'),
            device=draft_logits.device,
            dtype=draft_logits.dtype
        )
        
        # Copy action token logits from draft to corresponding target positions
        # Draft action tokens: [draft_action_start, draft_vocab_size)
        # Target action tokens: [target_action_start, target_vocab_size)
        draft_action_logits = draft_logits[:, self.draft_action_start:self.draft_vocab_size]
        target_logits[:, self.target_action_start:self.target_vocab_size] = draft_action_logits
        
        return target_logits
    
    # =========================================================================
    # DEBUG: Token mapping verification methods
    # =========================================================================
    
    def _get_action_bin_from_draft_token(self, draft_token_id: int) -> int:
        """Get action bin index from draft token ID."""
        if draft_token_id >= self.draft_action_start:
            return draft_token_id - self.draft_action_start
        return -1  # Not an action token
    
    def _get_action_bin_from_target_token(self, target_token_id: int) -> int:
        """Get action bin index from target token ID."""
        if target_token_id >= self.target_action_start:
            return target_token_id - self.target_action_start
        return -1  # Not an action token
    
    def _get_continuous_action_from_bin(self, bin_idx: int) -> float:
        """Convert action bin index to continuous action value using target's bin centers."""
        if 0 <= bin_idx < len(self.target.bin_centers):
            return self.target.bin_centers[bin_idx]
        return float('nan')
    
    def _debug_token_mapping(self, draft_token_id: int, target_token_id: int, prefix: str = ""):
        """Debug print showing token mapping verification."""
        draft_bin = self._get_action_bin_from_draft_token(draft_token_id)
        target_bin = self._get_action_bin_from_target_token(target_token_id)
        
        draft_action = self._get_continuous_action_from_bin(draft_bin)
        target_action = self._get_continuous_action_from_bin(target_bin)
        
        match_status = "✓ MATCH" if draft_bin == target_bin else "✗ MISMATCH"
        
        print(f"\033[38;2;100;200;255m[DEBUG TOKEN MAP] {prefix}\033[0m")
        print(f"  Draft token:  {draft_token_id} → bin {draft_bin} → action {draft_action:.4f}")
        print(f"  Target token: {target_token_id} → bin {target_bin} → action {target_action:.4f}")
        print(f"  Bins match: {match_status}")
        
        return draft_bin == target_bin
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = SpeculativeDecodingStats()
    
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
    
    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        if self.temperature <= 0:
            # For greedy, use a very low temperature to approximate argmax
            return F.softmax(logits / 0.01, dim=-1)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _prepare_target_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the target (OpenVLA) model."""
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.target_processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        # IMPORTANT: Add the special empty token (29871) to input_ids if not present
        # This is what OpenVLA's predict_action does - the action tokens come AFTER this token
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"], torch.tensor([[29871]], device=self.device)),
                dim=1
            )
            # Also extend attention mask
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    (inputs["attention_mask"], torch.ones((1, 1), device=self.device, dtype=inputs["attention_mask"].dtype)),
                    dim=1
                )
        
        return inputs
    
    def _prepare_draft_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the draft (MiniVLA) model."""
        # Get prompt using draft model's prompt builder
        prompt_builder = self.draft.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        # Tokenize
        tokenizer = self.draft.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # Handle special token for Llama tokenizer
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], device=self.device)),
                    dim=1
                )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Process image
        image_transform = self.draft.vision_backbone.get_image_transform()
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    @torch.inference_mode()
    def predict_action_speculative(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key_target: str,
    ) -> Tuple[np.ndarray, SpeculativeDecodingStats]:
        """
        Generate action using speculative decoding.
        
        Args:
            image: PIL Image observation
            instruction: Task instruction string
            unnorm_key_target: Key for action un-normalization statistics of the target model
            
        Returns:
            Tuple of (unnormalized action array, decoding statistics)
        """
        # Reset per-call stats
        call_stats = SpeculativeDecodingStats()
        
        action_dim = self.target.get_action_dim(unnorm_key_target)
        
        
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target vocab size: {self.target.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target embedding size: {self.target.language_model.model.embed_tokens.weight.shape[0]}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Draft tokenizer vocab size: {self.draft.llm_backbone.tokenizer.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Action dim: {action_dim}")
        
        # Prepare inputs for both models
        target_inputs = self._prepare_target_inputs(image, instruction)
        draft_input_ids, draft_attention_mask, draft_pixel_values = self._prepare_draft_inputs(image, instruction)
        
        # Cast draft inputs to appropriate dtype
        autocast_dtype = self.draft.llm_backbone.half_precision_dtype
        
        # Initialize caches
        target_cache = None
        draft_cache = None
        
        generated_token_ids = []
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # === Initial forward pass to get KV cache and initial logits ===
            # NOTE: After adding token 29871 to input, the model should output action tokens directly
            
            # Target model initial forward (prefill)
            target_out = self.target(
                **target_inputs,
                past_key_values=None,
                use_cache=self.use_cache,
            )
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # DEBUG: Check what the target model wants to output first
            top_target_token = torch.argmax(target_logits, dim=-1).item()
            top_target_bin = self._get_action_bin_from_target_token(top_target_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, target top token:\033[0m")
            print(f"  top_token={top_target_token} → bin={top_target_bin} → is_action_token={top_target_bin >= 0}")
            
            # Draft model initial forward (prefill)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_out = self.draft(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    pixel_values=draft_pixel_values,
                    past_key_values=None,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # DEBUG: Check what the draft model wants to output first
            top_draft_token = torch.argmax(draft_logits, dim=-1).item()
            top_draft_bin = self._get_action_bin_from_draft_token(top_draft_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, draft top token:\033[0m")
            print(f"  top_token={top_draft_token} → bin={top_draft_bin} → is_action_token={top_draft_bin >= 0}")
            
            # === Main speculative decoding loop ===
            # We start directly with drafting - no need to sample a first token separately
            while len(generated_token_ids) < action_dim:
                # Determine how many tokens to speculate
                gamma = min(self.gamma, action_dim - len(generated_token_ids))
                
                # Generate gamma draft tokens
                draft_tokens = []
                draft_probs_list = []
                
                current_draft_cache = draft_cache
                current_draft_logits = draft_logits
                
                for _ in range(gamma):
                    draft_probs = self._get_probs(current_draft_logits)
                    draft_token = self._sample_token(current_draft_logits)
                    
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(draft_probs)
                    
                    # Advance draft model
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=draft_token.to(self.device),
                            past_key_values=current_draft_cache,
                            use_cache=self.use_cache,
                        )
                    current_draft_cache = draft_step.past_key_values
                    current_draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                
                call_stats.total_draft_tokens_proposed += gamma
                
                # Map draft tokens to target vocab space for verification
                draft_token_ids_target = []
                print(f"\033[38;2;100;200;255m[DEBUG] Mapping {gamma} draft tokens to target vocab:\033[0m")
                for idx, dt in enumerate(draft_tokens):
                    draft_id = dt.item()
                    target_id = self._draft_token_to_target(draft_id)
                    draft_token_ids_target.append(target_id)
                    
                    # Verify mapping preserves action bin
                    draft_bin = self._get_action_bin_from_draft_token(draft_id)
                    target_bin = self._get_action_bin_from_target_token(target_id)
                    draft_action = self._get_continuous_action_from_bin(draft_bin)
                    target_action = self._get_continuous_action_from_bin(target_bin)
                    match = "✓" if draft_bin == target_bin else "✗ MISMATCH!"
                    print(f"  [{idx}] draft_tok={draft_id} → target_tok={target_id} | bin: {draft_bin}→{target_bin} | action: {draft_action:.4f}→{target_action:.4f} {match}")
                
                # Verify with target model
                # IMPORTANT: We need to collect logits BEFORE feeding each token
                # - target_logits (from prefill or last step) is used to evaluate draft_token[0]
                # - logits after feeding token[0] is used to evaluate draft_token[1]
                # etc.
                
                target_cache_for_verify = target_cache
                # Start with current target_logits for evaluating first draft token
                target_logits_for_verification = [target_logits.unsqueeze(1)]  # [1, 1, vocab]
                last_target_logits = None  # Will hold logits after feeding last token
                
                for i in range(gamma):
                    target_token_input = torch.tensor([[draft_token_ids_target[i]]], device=self.device)
                    target_step = self.target(
                        input_ids=target_token_input,
                        past_key_values=target_cache_for_verify,
                        use_cache=self.use_cache,
                    )
                    target_cache_for_verify = target_step.past_key_values
                    last_target_logits = target_step.logits[:, -1, :]  # Always save last logits
                    # Store logits for evaluating the NEXT token (if there is one)
                    if i < gamma - 1:
                        target_logits_for_verification.append(target_step.logits[:, -1:, :])
                
                call_stats.total_target_forward_passes += gamma
                
                # Stack target logits - now target_logits_for_verification[i] evaluates draft_token[i]
                target_logits_batch = torch.cat(target_logits_for_verification, dim=1)  # [1, gamma, actual_vocab_dim]
                target_probs_batch = self._get_probs(target_logits_batch)
                
                # DEBUG: Show which token the target actually wants at each position
                print(f"\033[38;2;100;200;255m[DEBUG] Target's preferred tokens at each position:\033[0m")
                for i in range(gamma):
                    top_tok = torch.argmax(target_logits_batch[0, i, :]).item()
                    top_bin = self._get_action_bin_from_target_token(top_tok)
                    draft_tok = draft_token_ids_target[i]
                    draft_bin = self._get_action_bin_from_target_token(draft_tok)
                    p_top = target_probs_batch[0, i, top_tok].item()
                    p_draft = target_probs_batch[0, i, draft_tok].item()
                    print(f"  pos[{i}]: target_wants={top_tok}(bin={top_bin}, p={p_top:.4f}) | draft_proposed={draft_tok}(bin={draft_bin}, p={p_draft:.4f})")
                
                # Get actual target logit dimension from the output
                actual_target_logit_dim = target_logits_batch.shape[-1]
                
                # Remap draft probs to target vocab space for comparison
                # Use actual target logit dimension to ensure tensor size match
                draft_probs_remapped = [self._remap_logits_draft_to_target(dp, actual_target_logit_dim) for dp in draft_probs_list]
                draft_probs_remapped = [self._get_probs(dp) for dp in draft_probs_remapped]
                
                # Rejection sampling loop
                n_accepted = 0
                for i in range(gamma):
                    draft_token_id_draft = draft_tokens[i].item()  # In draft vocab
                    draft_token_id_target = draft_token_ids_target[i]  # Mapped to target vocab
                    
                    draft_prob_remapped = draft_probs_remapped[i]
                    target_prob = target_probs_batch[:, i, :]
                    
                    # Get probability of the token under both models (in target vocab space)
                    p_target = target_prob[0, draft_token_id_target].item()
                    p_draft = draft_prob_remapped[0, draft_token_id_target].item()
                    
                    # Get the target's preferred token and compute bin distances
                    target_preferred_token = torch.argmax(target_prob, dim=-1).item()
                    draft_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                    target_bin = self._get_action_bin_from_target_token(target_preferred_token)
                    
                    # Compute bin distance (for relaxed acceptance)
                    if draft_bin >= 0 and target_bin >= 0:
                        bin_distance = abs(draft_bin - target_bin)
                    else:
                        bin_distance = float('inf')  # Non-action tokens don't benefit from relaxed acceptance
                    
                    # Relaxed acceptance: accept if within r bins of target's preference
                    relaxed_accept = (self.relaxed_acceptance_r > 0 and bin_distance <= self.relaxed_acceptance_r)
                    
                    # Standard rejection sampling
                    if p_draft > 0:
                        acceptance_prob = min(1.0, p_target / p_draft)
                    else:
                        acceptance_prob = 1.0 if p_target > 0 else 0.0
                    
                    standard_accept = (torch.rand(1).item() < acceptance_prob)
                    
                    # Accept if either relaxed acceptance or standard acceptance passes
                    if relaxed_accept or standard_accept:
                        # Accept this token (store in target vocab space)
                        accepted_action = self._get_continuous_action_from_bin(draft_bin)
                        accept_reason = "RELAXED" if relaxed_accept and not standard_accept else "STANDARD"
                        print(f"\033[38;2;0;255;0m[ACCEPT-{accept_reason}]\033[0m token[{i}]: target_tok={draft_token_id_target} → bin={draft_bin} → action={accepted_action:.4f}")
                        print(f"  target_preferred: bin={target_bin} | bin_distance={bin_distance} | r={self.relaxed_acceptance_r}")
                        print(f"  p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        if len(generated_token_ids) >= action_dim:
                            print(f"\033[38;2;255;165;0m[SRP] -> \033[0m Generated {len(generated_token_ids)}/{action_dim} tokens, done.")
                            break
                    else:
                        # Reject - sample from adjusted distribution
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        if adjusted_probs.sum() > 0:
                            corrected_token = torch.multinomial(adjusted_probs, num_samples=1)
                        else:
                            corrected_token = self._sample_token(target_prob.unsqueeze(0))
                        
                        corrected_token_id = int(corrected_token.item())
                        corrected_bin = self._get_action_bin_from_target_token(corrected_token_id)
                        rejected_action = self._get_continuous_action_from_bin(draft_bin)
                        corrected_action = self._get_continuous_action_from_bin(corrected_bin)
                        bin_diff = abs(draft_bin - corrected_bin) if draft_bin >= 0 and corrected_bin >= 0 else -1
                        
                        print(f"\033[38;2;255;100;100m[REJECT]\033[0m token[{i}]: bin_distance={bin_distance} > r={self.relaxed_acceptance_r}")
                        print(f"  Draft proposed: target_tok={draft_token_id_target} → bin={draft_bin} → action={rejected_action:.4f}")
                        print(f"  Target prefers: target_tok={target_preferred_token} → bin={target_bin}")
                        print(f"  Corrected to:   target_tok={corrected_token_id} → bin={corrected_bin} → action={corrected_action:.4f}")
                        print(f"  p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        
                        # Store corrected token (already in target vocab space)
                        generated_token_ids.append(corrected_token_id)
                        call_stats.total_tokens_generated += 1
                        n_accepted = i  # Number of accepted tokens (before this rejection)
                        break
                                    
                # Update caches after acceptance/rejection
                if n_accepted == gamma and len(generated_token_ids) < action_dim:
                    # All accepted - need to sample one more from target
                    print(f"\033[38;2;0;255;0m[ALL ACCEPTED]\033[0m All {gamma} draft tokens accepted! Sampling bonus token from target...")
                    target_cache = target_cache_for_verify
                    target_logits = last_target_logits  # Logits after feeding all gamma tokens
                    
                    # Sample additional token from target (in target vocab space)
                    bonus_token_target = self._sample_token(target_logits)
                    bonus_token_id_target = int(bonus_token_target.item())
                    bonus_bin = self._get_action_bin_from_target_token(bonus_token_id_target)
                    bonus_action = self._get_continuous_action_from_bin(bonus_bin)
                    print(f"  Bonus token: target_tok={bonus_token_id_target} → bin={bonus_bin} → action={bonus_action:.4f}")
                    generated_token_ids.append(bonus_token_id_target)
                    call_stats.total_tokens_generated += 1
                    
                    # Update target cache
                    target_step = self.target(
                        input_ids=bonus_token_target,
                        past_key_values=target_cache,
                        use_cache=self.use_cache,
                    )
                    target_cache = target_step.past_key_values
                    target_logits = target_step.logits[:, -1, :]
                    call_stats.total_target_forward_passes += 1
                    
                    # Map bonus token to draft vocab and update draft cache
                    bonus_token_id_draft = self._target_token_to_draft(bonus_token_id_target)
                    bonus_draft_bin = self._get_action_bin_from_draft_token(bonus_token_id_draft)
                    print(f"  Mapped to draft: draft_tok={bonus_token_id_draft} → bin={bonus_draft_bin} {'✓' if bonus_bin == bonus_draft_bin else '✗ MISMATCH!'}")
                    bonus_token_draft = torch.tensor([[bonus_token_id_draft]], device=self.device)
                    
                    draft_cache = current_draft_cache
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=bonus_token_draft,
                            past_key_values=draft_cache,
                            use_cache=self.use_cache,
                        )
                    draft_cache = draft_step.past_key_values
                    draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                    
                else:
                    # Some tokens rejected - prune caches
                    tokens_to_discard = gamma - n_accepted
                    if tokens_to_discard > 0 and self.use_cache:
                        # We need to prune and resync
                        # Use the cache state after the accepted tokens
                        target_cache = target_cache_for_verify
                        if tokens_to_discard > 0:
                            target_cache = prune_cache(target_cache, tokens_to_discard)
                        
                        # Rebuild draft cache
                        draft_cache = prune_cache(current_draft_cache, gamma - n_accepted)
                    
                    # Get logits for next round
                    if len(generated_token_ids) < action_dim:
                        # Last token is in target vocab space
                        last_token_id_target = generated_token_ids[-1]
                        last_token_target = torch.tensor([[last_token_id_target]], device=self.device)
                        
                        target_step = self.target(
                            input_ids=last_token_target,
                            past_key_values=target_cache,
                            use_cache=self.use_cache,
                        )
                        target_cache = target_step.past_key_values
                        target_logits = target_step.logits[:, -1, :]
                        call_stats.total_target_forward_passes += 1
                        
                        # Map to draft vocab for draft model
                        last_token_id_draft = self._target_token_to_draft(last_token_id_target)
                        last_token_draft = torch.tensor([[last_token_id_draft]], device=self.device)
                        
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                            draft_step = self.draft(
                                input_ids=last_token_draft,
                                past_key_values=draft_cache,
                                use_cache=self.use_cache,
                            )
                        draft_cache = draft_step.past_key_values
                        draft_logits = draft_step.logits[:, -1, :]
                        call_stats.total_draft_forward_passes += 1
            
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Call stats: {call_stats}")
            
        # Decode tokens to actions
        predicted_action_token_ids = np.array(generated_token_ids[:action_dim], dtype=np.int64)
        
        # DEBUG: Show all generated tokens with their action values
        print(f"\033[38;2;100;200;255m[DEBUG] Final generated tokens summary:\033[0m")
        for dim_idx, tok_id in enumerate(predicted_action_token_ids):
            bin_idx = self._get_action_bin_from_target_token(int(tok_id))
            action_val = self._get_continuous_action_from_bin(bin_idx)
            print(f"  dim[{dim_idx}]: target_tok={tok_id} → bin={bin_idx} → normalized_action={action_val:.4f}")
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Predicted action token ids: {predicted_action_token_ids}")
        
        # Use target model's decoding (vocab_size - token_id approach)
        vocab_size = self.target.vocab_size
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.target.bin_centers.shape[0] - 1)
        normalized_actions = self.target.bin_centers[discretized_actions]
        
        # Un-normalize actions
        action_norm_stats = self.target.get_action_stats(unnorm_key_target)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # Update global stats
        self.stats.total_tokens_generated += call_stats.total_tokens_generated
        self.stats.total_draft_tokens_proposed += call_stats.total_draft_tokens_proposed
        self.stats.total_draft_tokens_accepted += call_stats.total_draft_tokens_accepted
        self.stats.total_target_forward_passes += call_stats.total_target_forward_passes
        self.stats.total_draft_forward_passes += call_stats.total_draft_forward_passes
        
        return actions, call_stats

# %%
class VLASpeculativeDecoderDDDRKVB:
    """
    Speculative decoding for VLA models.
    
    Uses a draft model (MiniVLA) to propose action tokens and a target model 
    (OpenVLA) to verify them.
    
    IMPORTANT: For speculative decoding to work correctly, both models should
    share the same tokenizer/vocabulary. If they don't, action token remapping
    is attempted but this only works for action tokens (last 256 tokens of vocab).
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        target_processor=None,
        gamma: int = 4,  # Number of draft tokens to propose at once
        use_cache: bool = True,
        temperature: float = 0.0,  # 0 = greedy/argmax
        n_action_bins: int = 256,  # Number of action bins (tokens)
        relaxed_acceptance_r: int = 0,  # Relaxed acceptance radius (0 = standard spec dec)
        use_batched_verification: bool = False,  # Use single forward pass for verification (no cache)
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: OpenVLA model (larger, slower, more accurate)
            draft_model: MiniVLA model (smaller, faster)
            target_processor: HuggingFace processor for target model
            gamma: Number of tokens to speculate at each step
            use_cache: Whether to use KV caching
            temperature: Sampling temperature (0 = greedy/argmax)
            n_action_bins: Number of action token bins (typically 256)
            relaxed_acceptance_r: Relaxed acceptance radius. If the draft token's bin
                is within r bins of the target's preferred bin, accept it.
                Set to 0 for standard speculative decoding behavior.
            use_batched_verification: If True, run verification in a single forward pass
                without KV cache. This re-processes the image but requires only 1 forward
                pass for all gamma tokens.
        """
        self.target = target_model
        self.draft = draft_model
        self.target_processor = target_processor
        self.gamma = gamma
        self.use_cache = use_cache
        self.temperature = temperature
        self.n_action_bins = n_action_bins
        self.relaxed_acceptance_r = relaxed_acceptance_r
        self.use_batched_verification = use_batched_verification
        
        # Get device
        self.device = next(target_model.parameters()).device
        
        # Stats tracking
        self.stats = SpeculativeDecodingStats()
        
        # Check vocabulary compatibility and setup token mapping
        self._setup_token_mapping()
    
    def _setup_token_mapping(self):
        """Setup token mapping between draft and target vocabularies."""
        # Get vocabulary sizes
        # Target model (OpenVLA/HF style) - use ACTUAL embedding dimension, not vocab_size attribute
        # The embedding may be padded to "multiple of" for efficiency
        if hasattr(self.target, 'language_model') and hasattr(self.target.language_model, 'model'):
            # Actual embedding dimension (includes padding)
            self.target_logit_dim = self.target.language_model.model.embed_tokens.weight.shape[0]
        elif hasattr(self.target, 'get_output_embeddings'):
            self.target_logit_dim = self.target.get_output_embeddings().weight.shape[0]
        else:
            self.target_logit_dim = self.target.config.vocab_size
        
        # Also get the "logical" vocab size (without padding) for action token calculation
        if hasattr(self.target, 'vocab_size'):
            self.target_vocab_size = self.target.vocab_size
        elif hasattr(self.target, 'config') and hasattr(self.target.config, 'vocab_size'):
            self.target_vocab_size = self.target.config.vocab_size
        else:
            self.target_vocab_size = self.target_logit_dim
        
        # Draft model (Prismatic style)
        if hasattr(self.draft, 'llm_backbone'):
            draft_tokenizer = self.draft.llm_backbone.tokenizer
            # Qwen2 uses len(tokenizer) for full vocab including added tokens
            self.draft_vocab_size = len(draft_tokenizer) if hasattr(draft_tokenizer, '__len__') else draft_tokenizer.vocab_size
            # Get actual logit dimension from draft model
            if hasattr(self.draft.llm_backbone, 'llm') and hasattr(self.draft.llm_backbone.llm, 'lm_head'):
                self.draft_logit_dim = self.draft.llm_backbone.llm.lm_head.weight.shape[0]
            else:
                self.draft_logit_dim = self.draft_vocab_size
        else:
            self.draft_vocab_size = self.draft.config.vocab_size
            self.draft_logit_dim = self.draft_vocab_size
        
        # Check if vocabularies match
        self.vocab_compatible = (self.target_logit_dim == self.draft_logit_dim)
        
        # Compute action token ranges
        # Action tokens are the LAST n_action_bins tokens before padding
        # For target: use vocab_size (not padded logit_dim)
        self.target_action_start = self.target_vocab_size - self.n_action_bins
        self.draft_action_start = self.draft_vocab_size - self.n_action_bins
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Target vocab_size: {self.target_vocab_size}, logit_dim: {self.target_logit_dim}, action tokens: [{self.target_action_start}, {self.target_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Draft vocab_size: {self.draft_vocab_size}, logit_dim: {self.draft_logit_dim}, action tokens: [{self.draft_action_start}, {self.draft_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Vocabularies compatible: {self.vocab_compatible}")
        
        if not self.vocab_compatible:
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"WARNING: Vocabulary mismatch! Token remapping will be used for action tokens only.")
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"This may affect acceptance rates. Consider using models with matching tokenizers.")
        
        # DEBUG: Verify mapping with example tokens
        print("\033[38;2;100;200;255m[DEBUG] Token mapping verification examples:\033[0m")
        for bin_idx in [0, 127, 255]:  # First, middle, last action bins
            draft_tok = self.draft_action_start + bin_idx
            target_tok = self._draft_token_to_target(draft_tok)
            draft_bin = self._get_action_bin_from_draft_token(draft_tok)
            target_bin = self._get_action_bin_from_target_token(target_tok)
            print(f"  Bin {bin_idx}: draft_token={draft_tok} → target_token={target_tok} | draft_bin={draft_bin}, target_bin={target_bin} | {'✓' if draft_bin == target_bin else '✗'}")
    
    def _draft_token_to_target(self, draft_token_id: int) -> int:
        """Map a draft token ID to target vocabulary."""
        if self.vocab_compatible:
            return draft_token_id
        
        # Check if this is an action token (from end of draft vocabulary)
        if draft_token_id >= self.draft_action_start:
            # Map to corresponding action token in target vocabulary
            action_bin = draft_token_id - self.draft_action_start
            target_token = self.target_action_start + action_bin
            return target_token
        else:
            # Non-action token - this shouldn't happen during action generation
            # Return as-is but clamp to valid range
            return min(draft_token_id, self.target_vocab_size - 1)
    
    def _target_token_to_draft(self, target_token_id: int) -> int:
        """Map a target token ID to draft vocabulary."""
        if self.vocab_compatible:
            return target_token_id
        
        # Check if this is an action token
        if target_token_id >= self.target_action_start:
            action_bin = target_token_id - self.target_action_start
            draft_token = self.draft_action_start + action_bin
            return draft_token
        else:
            return min(target_token_id, self.draft_vocab_size - 1)
    
    def _remap_logits_draft_to_target(self, draft_logits: torch.Tensor, target_logit_dim: int = None) -> torch.Tensor:
        """
        Remap draft logits to target vocabulary space.
        Only remaps action tokens; non-action tokens get -inf.
        
        Args:
            draft_logits: Logits from draft model
            target_logit_dim: Actual dimension of target logits (may differ from vocab_size due to padding)
        """
        if self.vocab_compatible:
            return draft_logits
        
        # Use provided dimension or fall back to stored logit_dim
        if target_logit_dim is None:
            target_logit_dim = self.target_logit_dim
        
        # Create target-sized logits filled with -inf (use actual logit dimension, not vocab_size)
        target_logits = torch.full(
            (draft_logits.shape[0], target_logit_dim),
            float('-inf'),
            device=draft_logits.device,
            dtype=draft_logits.dtype
        )
        
        # Copy action token logits from draft to corresponding target positions
        # Draft action tokens: [draft_action_start, draft_vocab_size)
        # Target action tokens: [target_action_start, target_vocab_size)
        draft_action_logits = draft_logits[:, self.draft_action_start:self.draft_vocab_size]
        target_logits[:, self.target_action_start:self.target_vocab_size] = draft_action_logits
        
        return target_logits
    
    # =========================================================================
    # DEBUG: Token mapping verification methods
    # =========================================================================
    
    def _get_action_bin_from_draft_token(self, draft_token_id: int) -> int:
        """Get action bin index from draft token ID."""
        if draft_token_id >= self.draft_action_start:
            return draft_token_id - self.draft_action_start
        return -1  # Not an action token
    
    def _get_action_bin_from_target_token(self, target_token_id: int) -> int:
        """Get action bin index from target token ID."""
        if target_token_id >= self.target_action_start:
            return target_token_id - self.target_action_start
        return -1  # Not an action token
    
    def _get_continuous_action_from_bin(self, bin_idx: int) -> float:
        """Convert action bin index to continuous action value using target's bin centers."""
        if 0 <= bin_idx < len(self.target.bin_centers):
            return self.target.bin_centers[bin_idx]
        return float('nan')
    
    def _debug_token_mapping(self, draft_token_id: int, target_token_id: int, prefix: str = ""):
        """Debug print showing token mapping verification."""
        draft_bin = self._get_action_bin_from_draft_token(draft_token_id)
        target_bin = self._get_action_bin_from_target_token(target_token_id)
        
        draft_action = self._get_continuous_action_from_bin(draft_bin)
        target_action = self._get_continuous_action_from_bin(target_bin)
        
        match_status = "✓ MATCH" if draft_bin == target_bin else "✗ MISMATCH"
        
        print(f"\033[38;2;100;200;255m[DEBUG TOKEN MAP] {prefix}\033[0m")
        print(f"  Draft token:  {draft_token_id} → bin {draft_bin} → action {draft_action:.4f}")
        print(f"  Target token: {target_token_id} → bin {target_bin} → action {target_action:.4f}")
        print(f"  Bins match: {match_status}")
        
        return draft_bin == target_bin
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = SpeculativeDecodingStats()
    
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
    
    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        if self.temperature <= 0:
            # For greedy, use a very low temperature to approximate argmax
            return F.softmax(logits / 0.01, dim=-1)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _prepare_target_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the target (OpenVLA) model."""
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.target_processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        # IMPORTANT: Add the special empty token (29871) to input_ids if not present
        # This is what OpenVLA's predict_action does - the action tokens come AFTER this token
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"], torch.tensor([[29871]], device=self.device)),
                dim=1
            )
            # Also extend attention mask
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    (inputs["attention_mask"], torch.ones((1, 1), device=self.device, dtype=inputs["attention_mask"].dtype)),
                    dim=1
                )
        
        return inputs
    
    def _prepare_draft_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the draft (MiniVLA) model."""
        # Get prompt using draft model's prompt builder
        prompt_builder = self.draft.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        # Tokenize
        tokenizer = self.draft.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # Handle special token for Llama tokenizer
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], device=self.device)),
                    dim=1
                )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Process image
        image_transform = self.draft.vision_backbone.get_image_transform()
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    @torch.inference_mode()
    def predict_action_speculative(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key_target: str,
    ) -> Tuple[np.ndarray, SpeculativeDecodingStats]:
        """
        Generate action using speculative decoding.
        
        Args:
            image: PIL Image observation
            instruction: Task instruction string
            unnorm_key_target: Key for action un-normalization statistics of the target model
            
        Returns:
            Tuple of (unnormalized action array, decoding statistics)
        """
        # Reset per-call stats
        call_stats = SpeculativeDecodingStats()
        
        action_dim = self.target.get_action_dim(unnorm_key_target)
        
        
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target vocab size: {self.target.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target embedding size: {self.target.language_model.model.embed_tokens.weight.shape[0]}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Draft tokenizer vocab size: {self.draft.llm_backbone.tokenizer.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Action dim: {action_dim}")
        
        # Prepare inputs for both models
        target_inputs = self._prepare_target_inputs(image, instruction)
        draft_input_ids, draft_attention_mask, draft_pixel_values = self._prepare_draft_inputs(image, instruction)
        
        # Cast draft inputs to appropriate dtype
        autocast_dtype = self.draft.llm_backbone.half_precision_dtype
        
        # Initialize caches
        target_cache = None
        draft_cache = None
        
        generated_token_ids = []
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # === Initial forward pass to get KV cache and initial logits ===
            # NOTE: After adding token 29871 to input, the model should output action tokens directly
            
            # Target model initial forward (prefill)
            target_out = self.target(
                **target_inputs,
                past_key_values=None,
                use_cache=self.use_cache,
            )
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # DEBUG: Check what the target model wants to output first
            top_target_token = torch.argmax(target_logits, dim=-1).item()
            top_target_bin = self._get_action_bin_from_target_token(top_target_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, target top token:\033[0m")
            print(f"  top_token={top_target_token} → bin={top_target_bin} → is_action_token={top_target_bin >= 0}")
            
            # Draft model initial forward (prefill)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_out = self.draft(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    pixel_values=draft_pixel_values,
                    past_key_values=None,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # DEBUG: Check what the draft model wants to output first
            top_draft_token = torch.argmax(draft_logits, dim=-1).item()
            top_draft_bin = self._get_action_bin_from_draft_token(top_draft_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, draft top token:\033[0m")
            print(f"  top_token={top_draft_token} → bin={top_draft_bin} → is_action_token={top_draft_bin >= 0}")
            
            # === Main speculative decoding loop ===
            # We start directly with drafting - no need to sample a first token separately
            while len(generated_token_ids) < action_dim:
                # Determine how many tokens to speculate
                gamma = min(self.gamma, action_dim - len(generated_token_ids))
                
                # Generate gamma draft tokens
                draft_tokens = []
                draft_probs_list = []
                
                current_draft_cache = draft_cache
                current_draft_logits = draft_logits
                
                for _ in range(gamma):
                    draft_probs = self._get_probs(current_draft_logits)
                    draft_token = self._sample_token(current_draft_logits)
                    
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(draft_probs)
                    
                    # Advance draft model
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=draft_token.to(self.device),
                            past_key_values=current_draft_cache,
                            use_cache=self.use_cache,
                        )
                    current_draft_cache = draft_step.past_key_values
                    current_draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                
                call_stats.total_draft_tokens_proposed += gamma
                
                # Map draft tokens to target vocab space for verification
                draft_token_ids_in_target_vocab = []
                print(f"\033[38;2;100;200;255m[DEBUG] Mapping {gamma} draft tokens to target vocab:\033[0m")
                for idx, dt in enumerate(draft_tokens):
                    draft_id = dt.item()
                    target_id = self._draft_token_to_target(draft_id)
                    draft_token_ids_in_target_vocab.append(target_id)
                    
                    # Verify mapping preserves action bin
                    draft_bin = self._get_action_bin_from_draft_token(draft_id)
                    target_bin = self._get_action_bin_from_target_token(target_id)
                    draft_action = self._get_continuous_action_from_bin(draft_bin)
                    target_action = self._get_continuous_action_from_bin(target_bin)
                    match = "✓" if draft_bin == target_bin else "✗ MISMATCH!"
                    print(f"  [{idx}] draft_tok={draft_id} → target_tok={target_id} | bin: {draft_bin}→{target_bin} | action: {draft_action:.4f}→{target_action:.4f} {match}")
                
                # Verify with target model
                if self.use_batched_verification:
                    # BATCHED VERIFICATION: Single forward pass without cache
                    # Re-processes the image but only 1 forward pass for all tokens
                    
                    # Build full input: original prompt + all draft tokens
                    # We need to re-run the full forward pass with image
                    draft_tokens_tensor = torch.tensor([draft_token_ids_in_target_vocab], device=self.device)  # [1, gamma]
                    
                    # Concatenate original input_ids with draft tokens
                    full_input_ids = torch.cat([
                        target_inputs["input_ids"],
                        draft_tokens_tensor
                    ], dim=1)
                    
                    # Run full forward pass (no cache, with image)
                    target_verify_out = self.target(
                        input_ids=full_input_ids,
                        attention_mask=torch.ones_like(full_input_ids),
                        pixel_values=target_inputs.get("pixel_values"),
                        past_key_values=None,  # No cache!
                        use_cache=False,
                    )
                    call_stats.total_target_forward_passes += 1
                    
                    # print(f"\033[38;2;100;200;255m[SRP] -- Target verify out:\033[0m {target_verify_out}, shape: {target_verify_out.logits.shape}")
                    
                    # Extract logits for draft token positions
                    # Original input has length L, draft tokens are at positions L to L+gamma-1
                    # Logits at position i predict token at position i+1
                    # So logits at L-1 predict first draft token, logits at L predict second, etc.
                    original_len = target_inputs["input_ids"].shape[1]
                    
                    # target_verify_out.logits has shape [1, L + gamma, vocab]
                    # We need logits at positions [L-1, L, L+1, ..., L+gamma-2] to evaluate draft tokens [0, 1, ..., gamma-1]
                    target_logits_batch = target_verify_out.logits[:, original_len-1:original_len-1+gamma, :]  # [1, gamma, vocab]
                    target_probs_batch = self._get_probs(target_logits_batch)
                    
                    # Last logits for bonus token (at position L+gamma-1)
                    last_target_logits = target_verify_out.logits[:, -1, :]
                    
                    # No cache to update in batched mode
                    target_cache_for_verify = None
                    
                    print(f"\033[38;2;100;200;255m[DEBUG] BATCHED verification: 1 forward pass for {gamma} tokens\033[0m")
                    
                else:
                    # SEQUENTIAL VERIFICATION: One forward pass per token with KV cache
                    # NOTE: OpenVLA/Prismatic models only support single-token inference with KV cache
                    
                    target_cache_for_verify = target_cache
                    # Start with current target_logits for evaluating first draft token
                    target_logits_for_verification = [target_logits.unsqueeze(1)]  # [1, 1, vocab]
                    last_target_logits = None
                    
                    for i in range(gamma):
                        target_token_input = torch.tensor([[draft_token_ids_in_target_vocab[i]]], device=self.device)
                        target_step = self.target(
                            input_ids=target_token_input,
                            past_key_values=target_cache_for_verify,
                            use_cache=self.use_cache,
                        )
                        target_cache_for_verify = target_step.past_key_values
                        last_target_logits = target_step.logits[:, -1, :]
                        # Store logits for evaluating the NEXT token (positions 1 to gamma-1)
                        if i < gamma - 1:
                            target_logits_for_verification.append(target_step.logits[:, -1:, :])
                        call_stats.total_target_forward_passes += 1
                    
                    # Stack target logits - target_logits_for_verification[i] evaluates draft_token[i]
                    target_logits_batch = torch.cat(target_logits_for_verification, dim=1)  # [1, gamma, vocab]
                    target_probs_batch = self._get_probs(target_logits_batch)
                
                # DEBUG: Show which token the target actually wants at each position
                print(f"\033[38;2;100;200;255m[DEBUG] Target's preferred tokens at each position:\033[0m")
                for i in range(gamma):
                    top_tok = torch.argmax(target_logits_batch[0, i, :]).item()
                    top_bin = self._get_action_bin_from_target_token(top_tok)
                    draft_tok = draft_token_ids_in_target_vocab[i]
                    draft_bin = self._get_action_bin_from_target_token(draft_tok)
                    p_top = target_probs_batch[0, i, top_tok].item()
                    p_draft = target_probs_batch[0, i, draft_tok].item()
                    print(f"  pos[{i}]: target_wants={top_tok}(bin={top_bin}, p={p_top:.4f}) | draft_proposed={draft_tok}(bin={draft_bin}, p={p_draft:.4f})")
                
                # Get actual target logit dimension from the output
                actual_target_logit_dim = target_logits_batch.shape[-1]
                
                # Remap draft probs to target vocab space for comparison
                # Use actual target logit dimension to ensure tensor size match
                draft_probs_remapped = [self._remap_logits_draft_to_target(dp, actual_target_logit_dim) for dp in draft_probs_list]
                draft_probs_remapped = [self._get_probs(dp) for dp in draft_probs_remapped]
                
                # Rejection sampling loop
                n_accepted = 0
                for i in range(gamma):
                    draft_token_id_draft = draft_tokens[i].item()  # In draft vocab
                    draft_token_id_target = draft_token_ids_in_target_vocab[i]  # Mapped to target vocab
                    
                    draft_prob_remapped = draft_probs_remapped[i]
                    target_prob = target_probs_batch[:, i, :]
                    
                    # Get probability of the token under both models (in target vocab space)
                    p_target = target_prob[0, draft_token_id_target].item()
                    p_draft = draft_prob_remapped[0, draft_token_id_target].item()
                    
                    # Get the target's preferred token and compute bin distances
                    target_preferred_token = torch.argmax(target_prob, dim=-1).item()
                    draft_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                    target_bin = self._get_action_bin_from_target_token(target_preferred_token)
                    
                    # Compute bin distance (for relaxed acceptance)
                    if draft_bin >= 0 and target_bin >= 0:
                        bin_distance = abs(draft_bin - target_bin)
                    else:
                        bin_distance = float('inf')  # Non-action tokens don't benefit from relaxed acceptance
                    
                    # Relaxed acceptance: accept if within r bins of target's preference
                    relaxed_accept = (self.relaxed_acceptance_r > 0 and bin_distance <= self.relaxed_acceptance_r)
                    
                    # Standard rejection sampling
                    if p_draft > 0:
                        acceptance_prob = min(1.0, p_target / p_draft)
                    else:
                        acceptance_prob = 1.0 if p_target > 0 else 0.0
                    
                    standard_accept = (torch.rand(1).item() < acceptance_prob)
                    
                    # Accept if either relaxed acceptance or standard acceptance passes
                    if relaxed_accept or standard_accept:
                        # Accept this token (store in target vocab space)
                        accepted_action = self._get_continuous_action_from_bin(draft_bin)
                        accept_reason = "RELAXED" if relaxed_accept and not standard_accept else "STANDARD"
                        print(f"\033[38;2;0;255;0m[ACCEPT-{accept_reason}]\033[0m token[{i}]: target_tok={draft_token_id_target} → bin={draft_bin} → action={accepted_action:.4f}")
                        print(f"  target_preferred: bin={target_bin} | bin_distance={bin_distance} | r={self.relaxed_acceptance_r}")
                        print(f"  p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        if len(generated_token_ids) >= action_dim:
                            print(f"\033[38;2;255;165;0m[SRP] -> \033[0m Generated {len(generated_token_ids)}/{action_dim} tokens, done.")
                            break
                    else:
                        # Reject - sample from adjusted distribution
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        if adjusted_probs.sum() > 0:
                            corrected_token = torch.multinomial(adjusted_probs, num_samples=1)
                        else:
                            corrected_token = self._sample_token(target_prob.unsqueeze(0))
                        
                        corrected_token_id = int(corrected_token.item())
                        corrected_bin = self._get_action_bin_from_target_token(corrected_token_id)
                        rejected_action = self._get_continuous_action_from_bin(draft_bin)
                        corrected_action = self._get_continuous_action_from_bin(corrected_bin)
                        bin_diff = abs(draft_bin - corrected_bin) if draft_bin >= 0 and corrected_bin >= 0 else -1
                        
                        print(f"\033[38;2;255;100;100m[REJECT]\033[0m token[{i}]: bin_distance={bin_distance} > r={self.relaxed_acceptance_r}")
                        print(f"  Draft proposed: target_tok={draft_token_id_target} → bin={draft_bin} → action={rejected_action:.4f}")
                        print(f"  Target prefers: target_tok={target_preferred_token} → bin={target_bin}")
                        print(f"  Corrected to:   target_tok={corrected_token_id} → bin={corrected_bin} → action={corrected_action:.4f}")
                        print(f"  p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        
                        # Store corrected token (already in target vocab space)
                        generated_token_ids.append(corrected_token_id)
                        call_stats.total_tokens_generated += 1
                        n_accepted = i  # Number of accepted tokens (before this rejection)
                        break
                                    
                # Update caches after acceptance/rejection
                if n_accepted == gamma and len(generated_token_ids) < action_dim:
                    # All accepted - need to sample one more from target
                    print(f"\033[38;2;0;255;0m[ALL ACCEPTED]\033[0m All {gamma} draft tokens accepted! Sampling bonus token from target...")
                    if not self.use_batched_verification:
                        target_cache = target_cache_for_verify
                    target_logits = last_target_logits  # Logits after feeding all gamma tokens
                    
                    # Sample additional token from target (in target vocab space)
                    bonus_token_target = self._sample_token(target_logits)
                    bonus_token_id_target = int(bonus_token_target.item())
                    bonus_bin = self._get_action_bin_from_target_token(bonus_token_id_target)
                    bonus_action = self._get_continuous_action_from_bin(bonus_bin)
                    print(f"  Bonus token: target_tok={bonus_token_id_target} → bin={bonus_bin} → action={bonus_action:.4f}")
                    generated_token_ids.append(bonus_token_id_target)
                    call_stats.total_tokens_generated += 1
                    
                    # if self.use_batched_verification:
                    #     # In batched mode, rebuild context for next round
                    #     # Actually, if all tokens were accepted and we generated a bonus token,
                    #     # we need to rebuild context with all tokens for the next speculation round
                    #     if len(generated_token_ids) < action_dim:
                    #         generated_tokens_tensor = torch.tensor([generated_token_ids], device=self.device)
                    #         full_input_ids = torch.cat([
                    #             target_inputs["input_ids"],
                    #             generated_tokens_tensor
                    #         ], dim=1)
                            
                    #         target_step = self.target(
                    #             input_ids=full_input_ids,
                    #             attention_mask=torch.ones_like(full_input_ids),
                    #             pixel_values=target_inputs.get("pixel_values"),
                    #             past_key_values=None,
                    #             use_cache=False,
                    #         )
                    #         target_logits = target_step.logits[:, -1, :]
                    #         call_stats.total_target_forward_passes += 1
                            
                    #         # Same for draft
                    #         generated_tokens_draft = [self._target_token_to_draft(t) for t in generated_token_ids]
                    #         generated_tokens_draft_tensor = torch.tensor([generated_tokens_draft], device=self.device)
                    #         with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                    #             draft_full_ids = torch.cat([draft_input_ids, generated_tokens_draft_tensor], dim=1)
                    #             draft_step = self.draft(
                    #                 input_ids=draft_full_ids,
                    #                 attention_mask=torch.ones_like(draft_full_ids),
                    #                 pixel_values=draft_pixel_values,
                    #                 past_key_values=None,
                    #                 use_cache=False,
                    #             )
                    #         draft_logits = draft_step.logits[:, -1, :]
                    #         call_stats.total_draft_forward_passes += 1
                    # else:
                    if not self.use_batched_verification:
                        # Update target cache
                        target_step = self.target(
                            input_ids=bonus_token_target,
                            past_key_values=target_cache,
                            use_cache=self.use_cache,
                        )
                        target_cache = target_step.past_key_values
                        target_logits = target_step.logits[:, -1, :]
                        call_stats.total_target_forward_passes += 1
                        
                        # Map bonus token to draft vocab and update draft cache
                        bonus_token_id_draft = self._target_token_to_draft(bonus_token_id_target)
                        bonus_draft_bin = self._get_action_bin_from_draft_token(bonus_token_id_draft)
                        print(f"  Mapped to draft: draft_tok={bonus_token_id_draft} → bin={bonus_draft_bin} {'✓' if bonus_bin == bonus_draft_bin else '✗ MISMATCH!'}")
                        bonus_token_draft = torch.tensor([[bonus_token_id_draft]], device=self.device)
                        
                        draft_cache = current_draft_cache
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                            draft_step = self.draft(
                                input_ids=bonus_token_draft,
                                past_key_values=draft_cache,
                                use_cache=self.use_cache,
                            )
                        draft_cache = draft_step.past_key_values
                        draft_logits = draft_step.logits[:, -1, :]
                        call_stats.total_draft_forward_passes += 1
                    
                else:
                    # Some tokens rejected
                    if self.use_batched_verification:
                        # In batched mode, we don't have caches
                        # For next round, we'll rebuild everything from scratch
                        # We just need to get the logits for the next draft round
                        if len(generated_token_ids) < action_dim:
                            # Rebuild full context with all generated tokens so far
                            generated_tokens_tensor = torch.tensor([generated_token_ids], device=self.device)
                            full_input_ids = torch.cat([
                                target_inputs["input_ids"],
                                generated_tokens_tensor
                            ], dim=1)
                            
                            # Run forward pass to get logits for next position
                            target_step = self.target(
                                input_ids=full_input_ids,
                                attention_mask=torch.ones_like(full_input_ids),
                                pixel_values=target_inputs.get("pixel_values"),
                                past_key_values=None,
                                use_cache=False,
                            )
                            target_logits = target_step.logits[:, -1, :]
                            call_stats.total_target_forward_passes += 1
                            
                            # Same for draft model
                            generated_tokens_draft = [self._target_token_to_draft(t) for t in generated_token_ids]
                            generated_tokens_draft_tensor = torch.tensor([generated_tokens_draft], device=self.device)
                            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                                # Need to rebuild draft input too
                                draft_full_ids = torch.cat([draft_input_ids, generated_tokens_draft_tensor], dim=1)
                                draft_step = self.draft(
                                    input_ids=draft_full_ids,
                                    attention_mask=torch.ones_like(draft_full_ids),
                                    pixel_values=draft_pixel_values,
                                    past_key_values=None,
                                    use_cache=False,
                                )
                            draft_logits = draft_step.logits[:, -1, :]
                            call_stats.total_draft_forward_passes += 1
                    else:
                        # Sequential mode - prune caches
                        tokens_to_discard = gamma - n_accepted
                        if tokens_to_discard > 0 and self.use_cache:
                            # We need to prune and resync
                            # Use the cache state after the accepted tokens
                            target_cache = target_cache_for_verify
                            if tokens_to_discard > 0:
                                target_cache = prune_cache(target_cache, tokens_to_discard)
                            
                            # Rebuild draft cache
                            draft_cache = prune_cache(current_draft_cache, gamma - n_accepted)
                        
                        # Get logits for next round
                        if len(generated_token_ids) < action_dim:
                            # Last token is in target vocab space
                            last_token_id_target = generated_token_ids[-1]
                            last_token_target = torch.tensor([[last_token_id_target]], device=self.device)
                            
                            target_step = self.target(
                                input_ids=last_token_target,
                                past_key_values=target_cache,
                                use_cache=self.use_cache,
                            )
                            target_cache = target_step.past_key_values
                            target_logits = target_step.logits[:, -1, :]
                            call_stats.total_target_forward_passes += 1
                            
                            # Map to draft vocab for draft model
                            last_token_id_draft = self._target_token_to_draft(last_token_id_target)
                            last_token_draft = torch.tensor([[last_token_id_draft]], device=self.device)
                            
                            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                                draft_step = self.draft(
                                    input_ids=last_token_draft,
                                    past_key_values=draft_cache,
                                    use_cache=self.use_cache,
                                )
                            draft_cache = draft_step.past_key_values
                            draft_logits = draft_step.logits[:, -1, :]
                            call_stats.total_draft_forward_passes += 1
            
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Call stats: {call_stats}")
            
        # Decode tokens to actions
        predicted_action_token_ids = np.array(generated_token_ids[:action_dim], dtype=np.int64)
        
        # DEBUG: Show all generated tokens with their action values
        print(f"\033[38;2;100;200;255m[DEBUG] Final generated tokens summary:\033[0m")
        for dim_idx, tok_id in enumerate(predicted_action_token_ids):
            bin_idx = self._get_action_bin_from_target_token(int(tok_id))
            action_val = self._get_continuous_action_from_bin(bin_idx)
            print(f"  dim[{dim_idx}]: target_tok={tok_id} → bin={bin_idx} → normalized_action={action_val:.4f}")
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Predicted action token ids: {predicted_action_token_ids}")
        
        # Use target model's decoding (vocab_size - token_id approach)
        vocab_size = self.target.vocab_size
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.target.bin_centers.shape[0] - 1)
        normalized_actions = self.target.bin_centers[discretized_actions]
        
        # Un-normalize actions
        action_norm_stats = self.target.get_action_stats(unnorm_key_target)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # Update global stats
        self.stats.total_tokens_generated += call_stats.total_tokens_generated
        self.stats.total_draft_tokens_proposed += call_stats.total_draft_tokens_proposed
        self.stats.total_draft_tokens_accepted += call_stats.total_draft_tokens_accepted
        self.stats.total_target_forward_passes += call_stats.total_target_forward_passes
        self.stats.total_draft_forward_passes += call_stats.total_draft_forward_passes
        
        return actions, call_stats

# %%
class VLASpeculativeDecoderDDDRKVB:
    """
    Speculative decoding for VLA models.
    
    Uses a draft model (MiniVLA) to propose action tokens and a target model 
    (OpenVLA) to verify them.
    
    IMPORTANT: For speculative decoding to work correctly, both models should
    share the same tokenizer/vocabulary. If they don't, action token remapping
    is attempted but this only works for action tokens (last 256 tokens of vocab).
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        target_processor=None,
        gamma: int = 4,  # Number of draft tokens to propose at once
        use_cache: bool = True,
        temperature: float = 0.0,  # 0 = greedy/argmax
        n_action_bins: int = 256,  # Number of action bins (tokens)
        relaxed_acceptance_r: int = 0,  # Relaxed acceptance radius (0 = standard spec dec)
        use_batched_verification: bool = False,  # Use single forward pass for verification (no cache)
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: OpenVLA model (larger, slower, more accurate)
            draft_model: MiniVLA model (smaller, faster)
            target_processor: HuggingFace processor for target model
            gamma: Number of tokens to speculate at each step
            use_cache: Whether to use KV caching
            temperature: Sampling temperature (0 = greedy/argmax)
            n_action_bins: Number of action token bins (typically 256)
            relaxed_acceptance_r: Relaxed acceptance radius. If the draft token's bin
                is within r bins of the target's preferred bin, accept it.
                Set to 0 for standard speculative decoding behavior.
            use_batched_verification: If True, run verification in a single forward pass
                without KV cache. This re-processes the image but requires only 1 forward
                pass for all gamma tokens.
        """
        self.target = target_model
        self.draft = draft_model
        self.target_processor = target_processor
        self.gamma = gamma
        self.use_cache = use_cache
        self.temperature = temperature
        self.n_action_bins = n_action_bins
        self.relaxed_acceptance_r = relaxed_acceptance_r
        self.use_batched_verification = use_batched_verification
        
        # Get device
        self.device = next(target_model.parameters()).device
        
        # Stats tracking
        self.stats = SpeculativeDecodingStats()
        
        # Check vocabulary compatibility and setup token mapping
        self._setup_token_mapping()
    
    def _setup_token_mapping(self):
        """Setup token mapping between draft and target vocabularies."""
        # Get vocabulary sizes
        # Target model (OpenVLA/HF style) - use ACTUAL embedding dimension, not vocab_size attribute
        # The embedding may be padded to "multiple of" for efficiency
        if hasattr(self.target, 'language_model') and hasattr(self.target.language_model, 'model'):
            # Actual embedding dimension (includes padding)
            self.target_logit_dim = self.target.language_model.model.embed_tokens.weight.shape[0]
        elif hasattr(self.target, 'get_output_embeddings'):
            self.target_logit_dim = self.target.get_output_embeddings().weight.shape[0]
        else:
            self.target_logit_dim = self.target.config.vocab_size
        
        # Also get the "logical" vocab size (without padding) for action token calculation
        if hasattr(self.target, 'vocab_size'):
            self.target_vocab_size = self.target.vocab_size
        elif hasattr(self.target, 'config') and hasattr(self.target.config, 'vocab_size'):
            self.target_vocab_size = self.target.config.vocab_size
        else:
            self.target_vocab_size = self.target_logit_dim
        
        # Draft model (Prismatic style)
        if hasattr(self.draft, 'llm_backbone'):
            draft_tokenizer = self.draft.llm_backbone.tokenizer
            # Qwen2 uses len(tokenizer) for full vocab including added tokens
            self.draft_vocab_size = len(draft_tokenizer) if hasattr(draft_tokenizer, '__len__') else draft_tokenizer.vocab_size
            # Get actual logit dimension from draft model
            if hasattr(self.draft.llm_backbone, 'llm') and hasattr(self.draft.llm_backbone.llm, 'lm_head'):
                self.draft_logit_dim = self.draft.llm_backbone.llm.lm_head.weight.shape[0]
            else:
                self.draft_logit_dim = self.draft_vocab_size
        else:
            self.draft_vocab_size = self.draft.config.vocab_size
            self.draft_logit_dim = self.draft_vocab_size
        
        # Check if vocabularies match
        self.vocab_compatible = (self.target_logit_dim == self.draft_logit_dim)
        
        # Compute action token ranges
        # Action tokens are the LAST n_action_bins tokens before padding
        # For target: use vocab_size (not padded logit_dim)
        self.target_action_start = self.target_vocab_size - self.n_action_bins
        self.draft_action_start = self.draft_vocab_size - self.n_action_bins
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Target vocab_size: {self.target_vocab_size}, logit_dim: {self.target_logit_dim}, action tokens: [{self.target_action_start}, {self.target_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Draft vocab_size: {self.draft_vocab_size}, logit_dim: {self.draft_logit_dim}, action tokens: [{self.draft_action_start}, {self.draft_vocab_size})")
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"Vocabularies compatible: {self.vocab_compatible}")
        
        if not self.vocab_compatible:
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"WARNING: Vocabulary mismatch! Token remapping will be used for action tokens only.")
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"This may affect acceptance rates. Consider using models with matching tokenizers.")
        
        # DEBUG: Verify mapping with example tokens
        print("\033[38;2;100;200;255m[DEBUG] Token mapping verification examples:\033[0m")
        for bin_idx in [0, 127, 255]:  # First, middle, last action bins
            draft_tok = self.draft_action_start + bin_idx
            target_tok = self._draft_token_to_target(draft_tok)
            draft_bin = self._get_action_bin_from_draft_token(draft_tok)
            target_bin = self._get_action_bin_from_target_token(target_tok)
            print(f"  Bin {bin_idx}: draft_token={draft_tok} → target_token={target_tok} | draft_bin={draft_bin}, target_bin={target_bin} | {'✓' if draft_bin == target_bin else '✗'}")
    
    def _draft_token_to_target(self, draft_token_id: int) -> int:
        """Map a draft token ID to target vocabulary."""
        if self.vocab_compatible:
            return draft_token_id
        
        # Check if this is an action token (from end of draft vocabulary)
        if draft_token_id >= self.draft_action_start:
            # Map to corresponding action token in target vocabulary
            action_bin = draft_token_id - self.draft_action_start
            target_token = self.target_action_start + action_bin
            return target_token
        else:
            # Non-action token - this shouldn't happen during action generation
            # Return as-is but clamp to valid range
            return min(draft_token_id, self.target_vocab_size - 1)
    
    def _target_token_to_draft(self, target_token_id: int) -> int:
        """Map a target token ID to draft vocabulary."""
        if self.vocab_compatible:
            return target_token_id
        
        # Check if this is an action token
        if target_token_id >= self.target_action_start:
            action_bin = target_token_id - self.target_action_start
            draft_token = self.draft_action_start + action_bin
            return draft_token
        else:
            return min(target_token_id, self.draft_vocab_size - 1)
    
    def _remap_logits_draft_to_target(self, draft_logits: torch.Tensor, target_logit_dim: int = None) -> torch.Tensor:
        """
        Remap draft logits to target vocabulary space.
        Only remaps action tokens; non-action tokens get -inf.
        
        Args:
            draft_logits: Logits from draft model
            target_logit_dim: Actual dimension of target logits (may differ from vocab_size due to padding)
        """
        if self.vocab_compatible:
            return draft_logits
        
        # Use provided dimension or fall back to stored logit_dim
        if target_logit_dim is None:
            target_logit_dim = self.target_logit_dim
        
        # Create target-sized logits filled with -inf (use actual logit dimension, not vocab_size)
        target_logits = torch.full(
            (draft_logits.shape[0], target_logit_dim),
            float('-inf'),
            device=draft_logits.device,
            dtype=draft_logits.dtype
        )
        
        # Copy action token logits from draft to corresponding target positions
        # Draft action tokens: [draft_action_start, draft_vocab_size)
        # Target action tokens: [target_action_start, target_vocab_size)
        draft_action_logits = draft_logits[:, self.draft_action_start:self.draft_vocab_size]
        target_logits[:, self.target_action_start:self.target_vocab_size] = draft_action_logits
        
        return target_logits
    
    # =========================================================================
    # DEBUG: Token mapping verification methods
    # =========================================================================
    
    def _get_action_bin_from_draft_token(self, draft_token_id: int) -> int:
        """Get action bin index from draft token ID."""
        if draft_token_id >= self.draft_action_start:
            return draft_token_id - self.draft_action_start
        return -1  # Not an action token
    
    def _get_action_bin_from_target_token(self, target_token_id: int) -> int:
        """Get action bin index from target token ID."""
        if target_token_id >= self.target_action_start:
            return target_token_id - self.target_action_start
        return -1  # Not an action token
    
    def _get_continuous_action_from_bin(self, bin_idx: int) -> float:
        """Convert action bin index to continuous action value using target's bin centers."""
        if 0 <= bin_idx < len(self.target.bin_centers):
            return self.target.bin_centers[bin_idx]
        return float('nan')
    
    def _debug_token_mapping(self, draft_token_id: int, target_token_id: int, prefix: str = ""):
        """Debug print showing token mapping verification."""
        draft_bin = self._get_action_bin_from_draft_token(draft_token_id)
        target_bin = self._get_action_bin_from_target_token(target_token_id)
        
        draft_action = self._get_continuous_action_from_bin(draft_bin)
        target_action = self._get_continuous_action_from_bin(target_bin)
        
        match_status = "✓ MATCH" if draft_bin == target_bin else "✗ MISMATCH"
        
        print(f"\033[38;2;100;200;255m[DEBUG TOKEN MAP] {prefix}\033[0m")
        print(f"  Draft token:  {draft_token_id} → bin {draft_bin} → action {draft_action:.4f}")
        print(f"  Target token: {target_token_id} → bin {target_bin} → action {target_action:.4f}")
        print(f"  Bins match: {match_status}")
        
        return draft_bin == target_bin
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = SpeculativeDecodingStats()
    
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
    
    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        if self.temperature <= 0:
            # For greedy, use a very low temperature to approximate argmax
            return F.softmax(logits / 0.01, dim=-1)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _prepare_target_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the target (OpenVLA) model."""
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.target_processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        # IMPORTANT: Add the special empty token (29871) to input_ids if not present
        # This is what OpenVLA's predict_action does - the action tokens come AFTER this token
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"], torch.tensor([[29871]], device=self.device)),
                dim=1
            )
            # Also extend attention mask
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    (inputs["attention_mask"], torch.ones((1, 1), device=self.device, dtype=inputs["attention_mask"].dtype)),
                    dim=1
                )
        
        return inputs
    
    def _prepare_draft_inputs(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the draft (MiniVLA) model."""
        # Get prompt using draft model's prompt builder
        prompt_builder = self.draft.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        # Tokenize
        tokenizer = self.draft.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # Handle special token for Llama tokenizer
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], device=self.device)),
                    dim=1
                )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Process image
        image_transform = self.draft.vision_backbone.get_image_transform()
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    @torch.inference_mode()
    def predict_action_speculative(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key_target: str,
    ) -> Tuple[np.ndarray, SpeculativeDecodingStats]:
        """
        Generate action using speculative decoding.
        
        Args:
            image: PIL Image observation
            instruction: Task instruction string
            unnorm_key_target: Key for action un-normalization statistics of the target model
            
        Returns:
            Tuple of (unnormalized action array, decoding statistics)
        """
        # Reset per-call stats
        call_stats = SpeculativeDecodingStats()
        
        action_dim = self.target.get_action_dim(unnorm_key_target)
        
        
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target vocab size: {self.target.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Target embedding size: {self.target.language_model.model.embed_tokens.weight.shape[0]}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Draft tokenizer vocab size: {self.draft.llm_backbone.tokenizer.vocab_size}")
        # print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Action dim: {action_dim}")
        
        # Prepare inputs for both models
        target_inputs = self._prepare_target_inputs(image, instruction)
        draft_input_ids, draft_attention_mask, draft_pixel_values = self._prepare_draft_inputs(image, instruction)
        
        # Cast draft inputs to appropriate dtype
        autocast_dtype = self.draft.llm_backbone.half_precision_dtype
        
        # Initialize caches
        target_cache = None
        draft_cache = None
        
        generated_token_ids = []
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # === Initial forward pass to get KV cache and initial logits ===
            # NOTE: After adding token 29871 to input, the model should output action tokens directly
            
            # Target model initial forward (prefill)
            target_out = self.target(
                **target_inputs,
                past_key_values=None,
                use_cache=self.use_cache,
            )
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # Calculate number of patch embeddings inserted by the VLM
            # After multimodal forward: seq_len = original_len + num_patches
            original_input_len = target_inputs["input_ids"].shape[1]
            seq_len_after_prefill = target_out.logits.shape[1]
            num_patches = seq_len_after_prefill - original_input_len
            print(f"\033[38;2;100;200;255m[DEBUG] VLM inserted {num_patches} patch embeddings\033[0m")
            
            # DEBUG: Check what the target model wants to output first
            top_target_token = torch.argmax(target_logits, dim=-1).item()
            top_target_bin = self._get_action_bin_from_target_token(top_target_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, target top token:\033[0m")
            print(f"  top_token={top_target_token} → bin={top_target_bin} → is_action_token={top_target_bin >= 0}")
            
            # Draft model initial forward (prefill)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_out = self.draft(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    pixel_values=draft_pixel_values,
                    past_key_values=None,
                    use_cache=self.use_cache,
                )
            draft_cache = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # DEBUG: Check what the draft model wants to output first
            top_draft_token = torch.argmax(draft_logits, dim=-1).item()
            top_draft_bin = self._get_action_bin_from_draft_token(top_draft_token)
            print(f"\033[38;2;100;200;255m[DEBUG] After prefill, draft top token:\033[0m")
            print(f"  top_token={top_draft_token} → bin={top_draft_bin} → is_action_token={top_draft_bin >= 0}")
            
            # === Main speculative decoding loop ===
            # We start directly with drafting - no need to sample a first token separately
            while len(generated_token_ids) < action_dim:
                # Determine how many tokens to speculate
                gamma = min(self.gamma, action_dim - len(generated_token_ids))
                
                # Generate gamma draft tokens
                draft_tokens = []
                draft_probs_list = []
                
                current_draft_cache = draft_cache
                current_draft_logits = draft_logits
                
                for _ in range(gamma):
                    draft_probs = self._get_probs(current_draft_logits)
                    draft_token = self._sample_token(current_draft_logits)
                    
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(draft_probs)
                    
                    # Advance draft model
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=draft_token.to(self.device),
                            past_key_values=current_draft_cache,
                            use_cache=self.use_cache,
                        )
                    current_draft_cache = draft_step.past_key_values
                    current_draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                
                call_stats.total_draft_tokens_proposed += gamma
                
                # Map draft tokens to target vocab space for verification
                draft_token_ids_target = []
                print(f"\033[38;2;100;200;255m[DEBUG] Mapping {gamma} draft tokens to target vocab:\033[0m")
                for idx, dt in enumerate(draft_tokens):
                    draft_id = dt.item()
                    target_id = self._draft_token_to_target(draft_id)
                    draft_token_ids_target.append(target_id)
                    
                    # Verify mapping preserves action bin
                    draft_bin = self._get_action_bin_from_draft_token(draft_id)
                    target_bin = self._get_action_bin_from_target_token(target_id)
                    draft_action = self._get_continuous_action_from_bin(draft_bin)
                    target_action = self._get_continuous_action_from_bin(target_bin)
                    match = "✓" if draft_bin == target_bin else "✗ MISMATCH!"
                    print(f"  [{idx}] draft_tok={draft_id} → target_tok={target_id} | bin: {draft_bin}→{target_bin} | action: {draft_action:.4f}→{target_action:.4f} {match}")
                
                # Verify with target model
                if self.use_batched_verification:
                    # BATCHED VERIFICATION: Single forward pass without cache
                    # Re-processes the image but only 1 forward pass for all tokens
                    
                    # Build full input: original prompt + PREVIOUSLY VERIFIED tokens + new draft tokens
                    draft_tokens_tensor = torch.tensor([draft_token_ids_target], device=self.device)  # [1, gamma]
                    
                    # Include previously generated tokens (if any)
                    if len(generated_token_ids) > 0:
                        prev_tokens_tensor = torch.tensor([generated_token_ids], device=self.device)
                        full_input_ids = torch.cat([
                            target_inputs["input_ids"],
                            prev_tokens_tensor,
                            draft_tokens_tensor
                        ], dim=1)
                        num_prev_tokens = len(generated_token_ids)
                    else:
                        full_input_ids = torch.cat([
                            target_inputs["input_ids"],
                            draft_tokens_tensor
                        ], dim=1)
                        num_prev_tokens = 0
                    
                    # Run full forward pass (no cache, with image)
                    target_verify_out = self.target(
                        input_ids=full_input_ids,
                        attention_mask=torch.ones_like(full_input_ids),
                        pixel_values=target_inputs.get("pixel_values"),
                        past_key_values=None,  # No cache!
                        use_cache=False,
                    )
                    call_stats.total_target_forward_passes += 1
                    
                    # Extract logits for draft token positions
                    # IMPORTANT: The VLM inserts patch embeddings after BOS, so the sequence is:
                    # [BOS] [num_patches embeddings] [rest of prompt] [prev verified tokens] [draft tokens]
                    # Output logits have shape [1, original_len + num_patches + num_prev + gamma, vocab]
                    # 
                    # To evaluate draft_token[i], we need logits at position:
                    #   original_len + num_patches + num_prev_tokens - 1 + i
                    # (because logits at position P predict token at position P+1)
                    
                    start_idx = original_input_len + num_patches + num_prev_tokens - 1
                    target_logits_batch = target_verify_out.logits[:, start_idx:start_idx+gamma, :]  # [1, gamma, vocab]
                    target_probs_batch = self._get_probs(target_logits_batch)
                    
                    # Last logits for bonus token
                    last_target_logits = target_verify_out.logits[:, -1, :]
                    
                    # No cache to update in batched mode
                    target_cache_for_verify = None
                    
                    print(f"\033[38;2;100;200;255m[DEBUG] BATCHED verification: 1 forward pass for {gamma} tokens\033[0m")
                    print(f"  full_input_ids shape: {full_input_ids.shape}, output logits shape: {target_verify_out.logits.shape}")
                    print(f"  num_patches={num_patches}, num_prev_tokens={num_prev_tokens}")
                    print(f"  extracting logits from positions [{start_idx}, {start_idx+gamma})")
                    
                else:
                    # SEQUENTIAL VERIFICATION: One forward pass per token with KV cache
                    # NOTE: OpenVLA/Prismatic models only support single-token inference with KV cache
                    
                    target_cache_for_verify = target_cache
                    # Start with current target_logits for evaluating first draft token
                    target_logits_for_verification = [target_logits.unsqueeze(1)]  # [1, 1, vocab]
                    last_target_logits = None
                    
                    for i in range(gamma):
                        target_token_input = torch.tensor([[draft_token_ids_target[i]]], device=self.device)
                        target_step = self.target(
                            input_ids=target_token_input,
                            past_key_values=target_cache_for_verify,
                            use_cache=self.use_cache,
                        )
                        target_cache_for_verify = target_step.past_key_values
                        last_target_logits = target_step.logits[:, -1, :]
                        # Store logits for evaluating the NEXT token (positions 1 to gamma-1)
                        if i < gamma - 1:
                            target_logits_for_verification.append(target_step.logits[:, -1:, :])
                        call_stats.total_target_forward_passes += 1
                    
                    # Stack target logits - target_logits_for_verification[i] evaluates draft_token[i]
                    target_logits_batch = torch.cat(target_logits_for_verification, dim=1)  # [1, gamma, vocab]
                    target_probs_batch = self._get_probs(target_logits_batch)
                
                # DEBUG: Show which token the target actually wants at each position
                print(f"\033[38;2;100;200;255m[DEBUG] Target's preferred tokens at each position:\033[0m")
                for i in range(gamma):
                    top_tok = torch.argmax(target_logits_batch[0, i, :]).item()
                    top_bin = self._get_action_bin_from_target_token(top_tok)
                    draft_tok = draft_token_ids_target[i]
                    draft_bin = self._get_action_bin_from_target_token(draft_tok)
                    p_top = target_probs_batch[0, i, top_tok].item()
                    p_draft = target_probs_batch[0, i, draft_tok].item()
                    print(f"  pos[{i}]: target_wants={top_tok}(bin={top_bin}, p={p_top:.4f}) | draft_proposed={draft_tok}(bin={draft_bin}, p={p_draft:.4f})")
                
                # Get actual target logit dimension from the output
                actual_target_logit_dim = target_logits_batch.shape[-1]
                
                # Remap draft probs to target vocab space for comparison
                # Use actual target logit dimension to ensure tensor size match
                draft_probs_remapped = [self._remap_logits_draft_to_target(dp, actual_target_logit_dim) for dp in draft_probs_list]
                draft_probs_remapped = [self._get_probs(dp) for dp in draft_probs_remapped]
                
                # Rejection sampling loop
                n_accepted = 0
                for i in range(gamma):
                    draft_token_id_draft = draft_tokens[i].item()  # In draft vocab
                    draft_token_id_target = draft_token_ids_target[i]  # Mapped to target vocab
                    
                    draft_prob_remapped = draft_probs_remapped[i]
                    target_prob = target_probs_batch[:, i, :]
                    
                    # Get probability of the token under both models (in target vocab space)
                    p_target = target_prob[0, draft_token_id_target].item()
                    p_draft = draft_prob_remapped[0, draft_token_id_target].item()
                    
                    # Get the target's preferred token and compute bin distances
                    target_preferred_token = torch.argmax(target_prob, dim=-1).item()
                    draft_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                    target_bin = self._get_action_bin_from_target_token(target_preferred_token)
                    
                    # Compute bin distance (for relaxed acceptance)
                    if draft_bin >= 0 and target_bin >= 0:
                        bin_distance = abs(draft_bin - target_bin)
                    else:
                        bin_distance = float('inf')  # Non-action tokens don't benefit from relaxed acceptance
                    
                    # Relaxed acceptance: accept if within r bins of target's preference
                    relaxed_accept = (self.relaxed_acceptance_r > 0 and bin_distance <= self.relaxed_acceptance_r)
                    
                    # Standard rejection sampling
                    if p_draft > 0:
                        acceptance_prob = min(1.0, p_target / p_draft)
                    else:
                        acceptance_prob = 1.0 if p_target > 0 else 0.0
                    
                    standard_accept = (torch.rand(1).item() < acceptance_prob)
                    
                    # Accept if either relaxed acceptance or standard acceptance passes
                    if relaxed_accept or standard_accept:
                        # Accept this token (store in target vocab space)
                        accepted_action = self._get_continuous_action_from_bin(draft_bin)
                        accept_reason = "RELAXED" if relaxed_accept and not standard_accept else "STANDARD"
                        print(f"\033[38;2;0;255;0m[ACCEPT-{accept_reason}]\033[0m token[{i}]: target_tok={draft_token_id_target} → bin={draft_bin} → action={accepted_action:.4f}")
                        print(f"  target_preferred: bin={target_bin} | bin_distance={bin_distance} | r={self.relaxed_acceptance_r}")
                        print(f"  p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        if len(generated_token_ids) >= action_dim:
                            print(f"\033[38;2;255;165;0m[SRP] -> \033[0m Generated {len(generated_token_ids)}/{action_dim} tokens, done.")
                            break
                    else:
                        # Reject - sample from adjusted distribution
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        if adjusted_probs.sum() > 0:
                            corrected_token = torch.multinomial(adjusted_probs, num_samples=1)
                        else:
                            corrected_token = self._sample_token(target_prob.unsqueeze(0))
                        
                        corrected_token_id = int(corrected_token.item())
                        corrected_bin = self._get_action_bin_from_target_token(corrected_token_id)
                        rejected_action = self._get_continuous_action_from_bin(draft_bin)
                        corrected_action = self._get_continuous_action_from_bin(corrected_bin)
                        bin_diff = abs(draft_bin - corrected_bin) if draft_bin >= 0 and corrected_bin >= 0 else -1
                        
                        print(f"\033[38;2;255;100;100m[REJECT]\033[0m token[{i}]: bin_distance={bin_distance} > r={self.relaxed_acceptance_r}")
                        print(f"  Draft proposed: target_tok={draft_token_id_target} → bin={draft_bin} → action={rejected_action:.4f}")
                        print(f"  Target prefers: target_tok={target_preferred_token} → bin={target_bin}")
                        print(f"  Corrected to:   target_tok={corrected_token_id} → bin={corrected_bin} → action={corrected_action:.4f}")
                        print(f"  p_target={p_target:.4f}, p_draft={p_draft:.4f}, accept_prob={acceptance_prob:.4f}")
                        
                        # Store corrected token (already in target vocab space)
                        generated_token_ids.append(corrected_token_id)
                        call_stats.total_tokens_generated += 1
                        n_accepted = i  # Number of accepted tokens (before this rejection)
                        break
                                    
                # Update caches after acceptance/rejection
                if n_accepted == gamma and len(generated_token_ids) < action_dim:
                    # All accepted - need to sample one more from target
                    print(f"\033[38;2;0;255;0m[ALL ACCEPTED]\033[0m All {gamma} draft tokens accepted! Sampling bonus token from target...")
                    if not self.use_batched_verification:
                        target_cache = target_cache_for_verify
                    target_logits = last_target_logits  # Logits after feeding all gamma tokens
                    
                    # Sample additional token from target (in target vocab space)
                    bonus_token_target = self._sample_token(target_logits)
                    bonus_token_id_target = int(bonus_token_target.item())
                    bonus_bin = self._get_action_bin_from_target_token(bonus_token_id_target)
                    bonus_action = self._get_continuous_action_from_bin(bonus_bin)
                    print(f"  Bonus token: target_tok={bonus_token_id_target} → bin={bonus_bin} → action={bonus_action:.4f}")
                    generated_token_ids.append(bonus_token_id_target)
                    call_stats.total_tokens_generated += 1
                    
                    if self.use_batched_verification:
                        # In batched mode, rebuild context for next round
                        # Actually, if all tokens were accepted and we generated a bonus token,
                        # we need to rebuild context with all tokens for the next speculation round
                        if len(generated_token_ids) < action_dim:
                            generated_tokens_tensor = torch.tensor([generated_token_ids], device=self.device)
                            full_input_ids = torch.cat([
                                target_inputs["input_ids"],
                                generated_tokens_tensor
                            ], dim=1)
                            
                            target_step = self.target(
                                input_ids=full_input_ids,
                                attention_mask=torch.ones_like(full_input_ids),
                                pixel_values=target_inputs.get("pixel_values"),
                                past_key_values=None,
                                use_cache=False,
                            )
                            target_logits = target_step.logits[:, -1, :]
                            call_stats.total_target_forward_passes += 1
                            
                            # Same for draft
                            generated_tokens_draft = [self._target_token_to_draft(t) for t in generated_token_ids]
                            generated_tokens_draft_tensor = torch.tensor([generated_tokens_draft], device=self.device)
                            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                                draft_full_ids = torch.cat([draft_input_ids, generated_tokens_draft_tensor], dim=1)
                                draft_step = self.draft(
                                    input_ids=draft_full_ids,
                                    attention_mask=torch.ones_like(draft_full_ids),
                                    pixel_values=draft_pixel_values,
                                    past_key_values=None,
                                    use_cache=False,
                                )
                            draft_logits = draft_step.logits[:, -1, :]
                            call_stats.total_draft_forward_passes += 1
                    else:
                        # Update target cache
                        target_step = self.target(
                            input_ids=bonus_token_target,
                            past_key_values=target_cache,
                            use_cache=self.use_cache,
                        )
                        target_cache = target_step.past_key_values
                        target_logits = target_step.logits[:, -1, :]
                        call_stats.total_target_forward_passes += 1
                        
                        # Map bonus token to draft vocab and update draft cache
                        bonus_token_id_draft = self._target_token_to_draft(bonus_token_id_target)
                        bonus_draft_bin = self._get_action_bin_from_draft_token(bonus_token_id_draft)
                        print(f"  Mapped to draft: draft_tok={bonus_token_id_draft} → bin={bonus_draft_bin} {'✓' if bonus_bin == bonus_draft_bin else '✗ MISMATCH!'}")
                        bonus_token_draft = torch.tensor([[bonus_token_id_draft]], device=self.device)
                        
                        draft_cache = current_draft_cache
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                            draft_step = self.draft(
                                input_ids=bonus_token_draft,
                                past_key_values=draft_cache,
                                use_cache=self.use_cache,
                            )
                        draft_cache = draft_step.past_key_values
                        draft_logits = draft_step.logits[:, -1, :]
                        call_stats.total_draft_forward_passes += 1
                    
                else:
                    # Some tokens rejected
                    if self.use_batched_verification:
                        # In batched mode, we don't have caches
                        # For next round, we'll rebuild everything from scratch
                        # We just need to get the logits for the next draft round
                        if len(generated_token_ids) < action_dim:
                            # Rebuild full context with all generated tokens so far
                            generated_tokens_tensor = torch.tensor([generated_token_ids], device=self.device)
                            full_input_ids = torch.cat([
                                target_inputs["input_ids"],
                                generated_tokens_tensor
                            ], dim=1)
                            
                            # Run forward pass to get logits for next position
                            target_step = self.target(
                                input_ids=full_input_ids,
                                attention_mask=torch.ones_like(full_input_ids),
                                pixel_values=target_inputs.get("pixel_values"),
                                past_key_values=None,
                                use_cache=False,
                            )
                            target_logits = target_step.logits[:, -1, :]
                            call_stats.total_target_forward_passes += 1
                            
                            # Same for draft model
                            generated_tokens_draft = [self._target_token_to_draft(t) for t in generated_token_ids]
                            generated_tokens_draft_tensor = torch.tensor([generated_tokens_draft], device=self.device)
                            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                                # Need to rebuild draft input too
                                draft_full_ids = torch.cat([draft_input_ids, generated_tokens_draft_tensor], dim=1)
                                draft_step = self.draft(
                                    input_ids=draft_full_ids,
                                    attention_mask=torch.ones_like(draft_full_ids),
                                    pixel_values=draft_pixel_values,
                                    past_key_values=None,
                                    use_cache=False,
                                )
                            draft_logits = draft_step.logits[:, -1, :]
                            call_stats.total_draft_forward_passes += 1
                    else:
                        # Sequential mode - prune caches
                        tokens_to_discard = gamma - n_accepted
                        if tokens_to_discard > 0 and self.use_cache:
                            # We need to prune and resync
                            # Use the cache state after the accepted tokens
                            target_cache = target_cache_for_verify
                            if tokens_to_discard > 0:
                                target_cache = prune_cache(target_cache, tokens_to_discard)
                            
                            # Rebuild draft cache
                            draft_cache = prune_cache(current_draft_cache, gamma - n_accepted)
                        
                        # Get logits for next round
                        if len(generated_token_ids) < action_dim:
                            # Last token is in target vocab space
                            last_token_id_target = generated_token_ids[-1]
                            last_token_target = torch.tensor([[last_token_id_target]], device=self.device)
                            
                            target_step = self.target(
                                input_ids=last_token_target,
                                past_key_values=target_cache,
                                use_cache=self.use_cache,
                            )
                            target_cache = target_step.past_key_values
                            target_logits = target_step.logits[:, -1, :]
                            call_stats.total_target_forward_passes += 1
                            
                            # Map to draft vocab for draft model
                            last_token_id_draft = self._target_token_to_draft(last_token_id_target)
                            last_token_draft = torch.tensor([[last_token_id_draft]], device=self.device)
                            
                            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                                draft_step = self.draft(
                                    input_ids=last_token_draft,
                                    past_key_values=draft_cache,
                                    use_cache=self.use_cache,
                                )
                            draft_cache = draft_step.past_key_values
                            draft_logits = draft_step.logits[:, -1, :]
                            call_stats.total_draft_forward_passes += 1
            
            print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Call stats: {call_stats}")
            
        # Decode tokens to actions
        predicted_action_token_ids = np.array(generated_token_ids[:action_dim], dtype=np.int64)
        
        # DEBUG: Show all generated tokens with their action values
        print(f"\033[38;2;100;200;255m[DEBUG] Final generated tokens summary:\033[0m")
        for dim_idx, tok_id in enumerate(predicted_action_token_ids):
            bin_idx = self._get_action_bin_from_target_token(int(tok_id))
            action_val = self._get_continuous_action_from_bin(bin_idx)
            print(f"  dim[{dim_idx}]: target_tok={tok_id} → bin={bin_idx} → normalized_action={action_val:.4f}")
        
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"[DEBUG] Predicted action token ids: {predicted_action_token_ids}")
        
        # Use target model's decoding (vocab_size - token_id approach)
        vocab_size = self.target.vocab_size
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.target.bin_centers.shape[0] - 1)
        normalized_actions = self.target.bin_centers[discretized_actions]
        
        # Un-normalize actions
        action_norm_stats = self.target.get_action_stats(unnorm_key_target)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # Update global stats
        self.stats.total_tokens_generated += call_stats.total_tokens_generated
        self.stats.total_draft_tokens_proposed += call_stats.total_draft_tokens_proposed
        self.stats.total_draft_tokens_accepted += call_stats.total_draft_tokens_accepted
        self.stats.total_target_forward_passes += call_stats.total_target_forward_passes
        self.stats.total_draft_forward_passes += call_stats.total_draft_forward_passes
        
        return actions, call_stats

# %%
# =============================================================================
# VLASpeculativeDecoderBatchedLM: Uses language model directly for batched verification
# Key insight from SpecVLA: After multimodal prefill, call language_model directly
# with inputs_embeds and cached KV, bypassing the restrictive multimodal forward.
# =============================================================================

class VLASpeculativeDecoderBatchedLM:
    """
    Speculative decoding for VLA models with EFFICIENT batched verification.
    
    Key difference from VLASpeculativeDecoderDDDRKVB:
    - Initial prefill: Full multimodal forward (processes image once)
    - Verification: Calls language_model DIRECTLY with embeddings + cached KV
    
    Why 10 forward passes for 7 tokens?
    1 prefill
    4 rejection rounds * (1 verify + 1 advance) = 8
    1 acceptance round * 1 verify = 1
    Total = 10
    The problem is: after each rejection, we do 2 target forward passes:
    Batched verification (wasted - we only needed position 0's logits)
    Advance with corrected token (to get logits for next round)
    - Early rejection: Skips batched verification if first draft token will definitely be rejected
    
    This bypasses the Prismatic multimodal forward restriction that only allows
    single-token inference with KV cache, enabling true batched verification.
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        target_processor=None,
        gamma: int = 4,
        temperature: float = 0.0,
        n_action_bins: int = 256,
        relaxed_acceptance_r: int = 0,
    ):
        self.target = target_model
        self.draft = draft_model
        self.target_processor = target_processor
        self.gamma = gamma
        self.temperature = temperature
        self.n_action_bins = n_action_bins
        self.relaxed_acceptance_r = relaxed_acceptance_r
        
        self.device = next(target_model.parameters()).device
        self.stats = SpeculativeDecodingStats()
        self._setup_token_mapping()
    
    def _setup_token_mapping(self):
        """Setup token mapping between draft and target vocabularies."""
        # Target model
        if hasattr(self.target, 'language_model') and hasattr(self.target.language_model, 'model'):
            self.target_logit_dim = self.target.language_model.model.embed_tokens.weight.shape[0]
        elif hasattr(self.target, 'get_output_embeddings'):
            self.target_logit_dim = self.target.get_output_embeddings().weight.shape[0]
        else:
            self.target_logit_dim = self.target.config.vocab_size
        
        if hasattr(self.target, 'vocab_size'):
            self.target_vocab_size = self.target.vocab_size
        elif hasattr(self.target, 'config') and hasattr(self.target.config, 'vocab_size'):
            self.target_vocab_size = self.target.config.vocab_size
        else:
            self.target_vocab_size = self.target_logit_dim
        
        # Draft model
        if hasattr(self.draft, 'llm_backbone'):
            draft_tokenizer = self.draft.llm_backbone.tokenizer
            self.draft_vocab_size = len(draft_tokenizer) if hasattr(draft_tokenizer, '__len__') else draft_tokenizer.vocab_size
            if hasattr(self.draft.llm_backbone, 'llm') and hasattr(self.draft.llm_backbone.llm, 'lm_head'):
                self.draft_logit_dim = self.draft.llm_backbone.llm.lm_head.weight.shape[0]
            else:
                self.draft_logit_dim = self.draft_vocab_size
        else:
            self.draft_vocab_size = self.draft.config.vocab_size
            self.draft_logit_dim = self.draft_vocab_size
        
        self.vocab_compatible = (self.target_logit_dim == self.draft_logit_dim)
        
        # Action token ranges
        self.target_action_start = self.target_vocab_size - self.n_action_bins
        self.draft_action_start = self.draft_vocab_size - self.n_action_bins
        
        print("\033[38;2;255;165;0m[BatchedLM] -> \033[0m", f"Target vocab_size: {self.target_vocab_size}, logit_dim: {self.target_logit_dim}")
        print("\033[38;2;255;165;0m[BatchedLM] -> \033[0m", f"Draft vocab_size: {self.draft_vocab_size}, logit_dim: {self.draft_logit_dim}")
        print("\033[38;2;255;165;0m[BatchedLM] -> \033[0m", f"Vocabularies compatible: {self.vocab_compatible}")
    
    def _draft_token_to_target(self, draft_token_id: int) -> int:
        if self.vocab_compatible:
            return draft_token_id
        if draft_token_id >= self.draft_action_start:
            action_bin = draft_token_id - self.draft_action_start
            return self.target_action_start + action_bin
        return min(draft_token_id, self.target_vocab_size - 1)
    
    def _target_token_to_draft(self, target_token_id: int) -> int:
        if self.vocab_compatible:
            return target_token_id
        if target_token_id >= self.target_action_start:
            action_bin = target_token_id - self.target_action_start
            return self.draft_action_start + action_bin
        return min(target_token_id, self.draft_vocab_size - 1)
    
    def _remap_logits_draft_to_target(self, draft_logits: torch.Tensor, target_logit_dim: int = None) -> torch.Tensor:
        if self.vocab_compatible:
            return draft_logits
        if target_logit_dim is None:
            target_logit_dim = self.target_logit_dim
        target_logits = torch.full(
            (draft_logits.shape[0], target_logit_dim), float('-inf'),
            device=draft_logits.device, dtype=draft_logits.dtype
        )
        draft_action_logits = draft_logits[:, self.draft_action_start:self.draft_vocab_size]
        target_logits[:, self.target_action_start:self.target_vocab_size] = draft_action_logits
        return target_logits
    
    def _get_action_bin_from_draft_token(self, draft_token_id: int) -> int:
        if draft_token_id >= self.draft_action_start:
            return draft_token_id - self.draft_action_start
        return -1
    
    def _get_action_bin_from_target_token(self, target_token_id: int) -> int:
        if target_token_id >= self.target_action_start:
            return target_token_id - self.target_action_start
        return -1
    
    def _get_continuous_action_from_bin(self, bin_idx: int) -> float:
        if 0 <= bin_idx < len(self.target.bin_centers):
            return self.target.bin_centers[bin_idx]
        return float('nan')
    
    def reset_stats(self):
        self.stats = SpeculativeDecodingStats()
    
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
    
    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            return F.softmax(logits / 0.01, dim=-1)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _prepare_target_inputs(self, image: Image.Image, instruction: str) -> Dict[str, torch.Tensor]:
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.target_processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        # Add special token 29871 if not present
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"], torch.tensor([[29871]], device=self.device)), dim=1
            )
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    (inputs["attention_mask"], torch.ones((1, 1), device=self.device, dtype=inputs["attention_mask"].dtype)), dim=1
                )
        return inputs
    
    def _prepare_draft_inputs(self, image: Image.Image, instruction: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_builder = self.draft.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        tokenizer = self.draft.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=self.device)), dim=1)
        
        attention_mask = torch.ones_like(input_ids)
        
        image_transform = self.draft.vision_backbone.get_image_transform()
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        return input_ids, attention_mask, pixel_values
    
    @torch.inference_mode()
    def predict_action_speculative(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key_target: str,
    ) -> Tuple[np.ndarray, SpeculativeDecodingStats]:
        """
        Generate action using speculative decoding with efficient batched verification.
        
        The key innovation: After multimodal prefill, we call the language model DIRECTLY
        with embeddings and cached KV, bypassing the restrictive multimodal forward.
        """
        call_stats = SpeculativeDecodingStats()
        action_dim = self.target.get_action_dim(unnorm_key_target)
        
        # Prepare inputs
        target_inputs = self._prepare_target_inputs(image, instruction)
        draft_input_ids, draft_attention_mask, draft_pixel_values = self._prepare_draft_inputs(image, instruction)
        autocast_dtype = self.draft.llm_backbone.half_precision_dtype
        
        generated_token_ids = []
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # === PHASE 1: Multimodal Prefill (processes image ONCE) ===
            print("\033[38;2;100;200;255m[BatchedLM] Phase 1: Multimodal Prefill\033[0m")
            
            # Target prefill - get KV cache with image embeddings
            target_out = self.target(
                **target_inputs,
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True,
            )
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            call_stats.total_target_forward_passes += 1
            
            # Get the sequence length after multimodal prefill (includes patch embeddings)
            kv_seq_len = target_cache[0][0].shape[2]  # [batch, heads, seq_len, head_dim]
            print(f"  Target KV cache seq_len after prefill: {kv_seq_len}")
            
            # Draft prefill
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                draft_out = self.draft(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    pixel_values=draft_pixel_values,
                    past_key_values=None,
                    use_cache=True,
                )
            draft_cache = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
            call_stats.total_draft_forward_passes += 1
            
            # Get embed_tokens function for converting tokens to embeddings
            embed_tokens = self.target.language_model.model.embed_tokens
            
            # === PHASE 2: Speculative Decoding Loop ===
            print("\033[38;2;100;200;255m[BatchedLM] Phase 2: Speculative Decoding\033[0m")
            
            while len(generated_token_ids) < action_dim:
                gamma = min(self.gamma, action_dim - len(generated_token_ids))
                
                # Generate FIRST draft token to check for early rejection
                draft_probs_first = self._get_probs(draft_logits)
                draft_token_first = self._sample_token(draft_logits)
                draft_token_first_target = self._draft_token_to_target(draft_token_first.item())
                
                # EARLY REJECTION CHECK: Can we determine rejection without batched verification?
                # For greedy decoding (temperature=0), we can check bin distance directly
                target_preferred = torch.argmax(target_logits, dim=-1).item()
                first_draft_bin = self._get_action_bin_from_target_token(draft_token_first_target)
                target_preferred_bin = self._get_action_bin_from_target_token(target_preferred)
                first_bin_distance = abs(first_draft_bin - target_preferred_bin) if first_draft_bin >= 0 and target_preferred_bin >= 0 else float('inf')
                
                # Check if first token would be accepted (relaxed or standard greedy)
                p_target_first = self._get_probs(target_logits)[0, draft_token_first_target].item()
                p_draft_first = draft_probs_first[0, draft_token_first_target].item()
                acceptance_prob_first = min(1.0, p_target_first / p_draft_first) if p_draft_first > 0 else (1.0 if p_target_first > 0 else 0.0)
                
                first_relaxed_accept = (self.relaxed_acceptance_r > 0 and first_bin_distance <= self.relaxed_acceptance_r)
                first_standard_accept = (torch.rand(1).item() < acceptance_prob_first)
                first_will_accept = first_relaxed_accept or first_standard_accept
                
                if not first_will_accept:
                    # EARLY REJECTION: Skip batched verification entirely!
                    # We already have target_logits, so just sample corrected token from it
                    call_stats.total_draft_tokens_proposed += 1
                    call_stats.total_draft_forward_passes += 1  # We did generate first draft token
                    
                    # Sample corrected token from target's distribution
                    draft_probs_first_remapped = self._get_probs(self._remap_logits_draft_to_target(draft_probs_first, target_logits.shape[-1]))
                    target_probs = self._get_probs(target_logits)
                    adjusted_probs = max_fn(target_probs - draft_probs_first_remapped)
                    if adjusted_probs.sum() > 0:
                        corrected_token = torch.multinomial(adjusted_probs, num_samples=1).item()
                    else:
                        corrected_token = target_preferred  # Use argmax
                    
                    corrected_bin = self._get_action_bin_from_target_token(corrected_token)
                    print(f"\033[38;2;255;200;100m[EARLY REJECT]\033[0m draft_bin={first_draft_bin}, target_bin={target_preferred_bin}, corrected_bin={corrected_bin} (saved verify pass!)")
                    
                    generated_token_ids.append(corrected_token)
                    call_stats.total_tokens_generated += 1
                    
                    # Advance target with corrected token (1 forward pass - this is necessary)
                    if len(generated_token_ids) < action_dim:
                        corrected_embeds = embed_tokens(torch.tensor([[corrected_token]], device=self.device))
                        corrected_pos = torch.tensor([[target_cache[0][0].shape[2]]], device=self.device)
                        lm_step = self.target.language_model(
                            inputs_embeds=corrected_embeds,
                            past_key_values=target_cache,
                            position_ids=corrected_pos,
                            use_cache=True,
                        )
                        target_cache = lm_step.past_key_values
                        target_logits = lm_step.logits[:, -1, :]
                        call_stats.total_target_forward_passes += 1
                        
                        # Update draft with corrected token (in draft vocab)
                        corrected_draft = self._target_token_to_draft(corrected_token)
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                            draft_step = self.draft(
                                input_ids=torch.tensor([[corrected_draft]], device=self.device),
                                past_key_values=draft_cache,
                                use_cache=True,
                            )
                        draft_cache = draft_step.past_key_values
                        draft_logits = draft_step.logits[:, -1, :]
                        call_stats.total_draft_forward_passes += 1
                    
                    continue  # Next iteration
                
                # First token looks good - generate remaining draft tokens
                draft_tokens = [draft_token_first]
                draft_probs_list = [draft_probs_first]
                current_draft_cache = draft_cache
                
                # Advance draft with first token
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                    draft_step = self.draft(
                        input_ids=draft_token_first.to(self.device),
                        past_key_values=current_draft_cache,
                        use_cache=True,
                    )
                current_draft_cache = draft_step.past_key_values
                current_draft_logits = draft_step.logits[:, -1, :]
                call_stats.total_draft_forward_passes += 1
                
                # Generate remaining gamma-1 draft tokens
                for _ in range(gamma - 1):
                    draft_probs = self._get_probs(current_draft_logits)
                    draft_token = self._sample_token(current_draft_logits)
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(draft_probs)
                    
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=draft_token.to(self.device),
                            past_key_values=current_draft_cache,
                            use_cache=True,
                        )
                    current_draft_cache = draft_step.past_key_values
                    current_draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                
                call_stats.total_draft_tokens_proposed += gamma
                
                # Map draft tokens to target vocab
                draft_token_ids_target = [self._draft_token_to_target(dt.item()) for dt in draft_tokens]
                
                # === BATCHED VERIFICATION via Language Model ===
                draft_tokens_tensor = torch.tensor([draft_token_ids_target], device=self.device)
                draft_embeds = embed_tokens(draft_tokens_tensor)
                
                current_kv_len = target_cache[0][0].shape[2]
                position_ids = torch.arange(current_kv_len, current_kv_len + gamma, device=self.device).unsqueeze(0)
                
                lm_out = self.target.language_model(
                    inputs_embeds=draft_embeds,
                    past_key_values=target_cache,
                    position_ids=position_ids,
                    use_cache=True,
                )
                call_stats.total_target_forward_passes += 1
                
                target_logits_batch = lm_out.logits
                new_target_cache = lm_out.past_key_values
                
                # eval_logits[i] evaluates draft_token[i]
                eval_logits = torch.cat([target_logits.unsqueeze(1), target_logits_batch[:, :-1, :]], dim=1)
                target_probs_batch = self._get_probs(eval_logits)
                last_target_logits = target_logits_batch[:, -1, :]
                
                print(f"\033[38;2;100;200;255m[BatchedLM] Batched verification: 1 forward pass for {gamma} tokens\033[0m")
                
                actual_target_logit_dim = target_probs_batch.shape[-1]
                draft_probs_remapped = [self._get_probs(self._remap_logits_draft_to_target(dp, actual_target_logit_dim)) for dp in draft_probs_list]
                
                # Rejection sampling (first token already checked, but re-verify for consistency)
                n_accepted = 0
                
                for i in range(gamma):
                    draft_token_id_target = draft_token_ids_target[i]
                    draft_prob_remapped = draft_probs_remapped[i]
                    target_prob = target_probs_batch[:, i, :]
                    
                    p_target = target_prob[0, draft_token_id_target].item()
                    p_draft = draft_prob_remapped[0, draft_token_id_target].item()
                    
                    target_preferred = torch.argmax(target_prob, dim=-1).item()
                    draft_bin = self._get_action_bin_from_target_token(draft_token_id_target)
                    target_bin = self._get_action_bin_from_target_token(target_preferred)
                    
                    bin_distance = abs(draft_bin - target_bin) if draft_bin >= 0 and target_bin >= 0 else float('inf')
                    relaxed_accept = (self.relaxed_acceptance_r > 0 and bin_distance <= self.relaxed_acceptance_r)
                    
                    acceptance_prob = min(1.0, p_target / p_draft) if p_draft > 0 else (1.0 if p_target > 0 else 0.0)
                    standard_accept = (torch.rand(1).item() < acceptance_prob)
                    
                    if relaxed_accept or standard_accept:
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        accept_reason = "RELAXED" if relaxed_accept and not standard_accept else "STANDARD"
                        print(f"\033[38;2;0;255;0m[ACCEPT-{accept_reason}]\033[0m [{i}] bin={draft_bin}, target_bin={target_bin}, dist={bin_distance}")
                        
                        if len(generated_token_ids) >= action_dim:
                            break
                    else:
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        corrected_token = torch.multinomial(adjusted_probs, num_samples=1).item() if adjusted_probs.sum() > 0 else target_preferred
                        
                        corrected_bin = self._get_action_bin_from_target_token(corrected_token)
                        print(f"\033[38;2;255;100;100m[REJECT]\033[0m [{i}] draft_bin={draft_bin}, target_bin={target_bin}, corrected_bin={corrected_bin}")
                        
                        generated_token_ids.append(corrected_token)
                        call_stats.total_tokens_generated += 1
                        break
                
                # Update caches based on acceptance
                if n_accepted == gamma and len(generated_token_ids) < action_dim:
                    # All accepted - sample bonus token
                    target_cache = new_target_cache
                    target_logits = last_target_logits
                    
                    bonus_token = self._sample_token(target_logits).item()
                    bonus_bin = self._get_action_bin_from_target_token(bonus_token)
                    print(f"\033[38;2;0;255;0m[ALL ACCEPTED]\033[0m Bonus token: bin={bonus_bin}")
                    generated_token_ids.append(bonus_token)
                    call_stats.total_tokens_generated += 1
                    
                    # Advance caches
                    bonus_embeds = embed_tokens(torch.tensor([[bonus_token]], device=self.device))
                    bonus_pos = torch.tensor([[target_cache[0][0].shape[2]]], device=self.device)
                    lm_step = self.target.language_model(
                        inputs_embeds=bonus_embeds,
                        past_key_values=target_cache,
                        position_ids=bonus_pos,
                        use_cache=True,
                    )
                    target_cache = lm_step.past_key_values
                    target_logits = lm_step.logits[:, -1, :]
                    call_stats.total_target_forward_passes += 1
                    
                    draft_cache = current_draft_cache
                    bonus_draft = self._target_token_to_draft(bonus_token)
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=torch.tensor([[bonus_draft]], device=self.device),
                            past_key_values=draft_cache,
                            use_cache=True,
                        )
                    draft_cache = draft_step.past_key_values
                    draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
                    
                elif len(generated_token_ids) < action_dim:
                    # Some rejected - update caches
                    if n_accepted > 0:
                        tokens_to_discard = gamma - n_accepted
                        target_cache = prune_cache(new_target_cache, tokens_to_discard) if tokens_to_discard > 0 else new_target_cache
                    
                    last_token = generated_token_ids[-1]
                    last_embeds = embed_tokens(torch.tensor([[last_token]], device=self.device))
                    last_pos = torch.tensor([[target_cache[0][0].shape[2]]], device=self.device)
                    
                    lm_step = self.target.language_model(
                        inputs_embeds=last_embeds,
                        past_key_values=target_cache,
                        position_ids=last_pos,
                        use_cache=True,
                    )
                    target_cache = lm_step.past_key_values
                    target_logits = lm_step.logits[:, -1, :]
                    call_stats.total_target_forward_passes += 1
                    
                    draft_cache = prune_cache(current_draft_cache, gamma - n_accepted)
                    last_draft = self._target_token_to_draft(last_token)
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.draft.enable_mixed_precision_training):
                        draft_step = self.draft(
                            input_ids=torch.tensor([[last_draft]], device=self.device),
                            past_key_values=draft_cache,
                            use_cache=True,
                        )
                    draft_cache = draft_step.past_key_values
                    draft_logits = draft_step.logits[:, -1, :]
                    call_stats.total_draft_forward_passes += 1
            
            print(f"\033[38;2;255;165;0m[BatchedLM] -> \033[0m Stats: {call_stats}")
        
        # Decode tokens to actions
        predicted_action_token_ids = np.array(generated_token_ids[:action_dim], dtype=np.int64)
        vocab_size = self.target.vocab_size
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.target.bin_centers.shape[0] - 1)
        normalized_actions = self.target.bin_centers[discretized_actions]
        
        # Un-normalize
        action_norm_stats = self.target.get_action_stats(unnorm_key_target)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # Update global stats
        self.stats.total_tokens_generated += call_stats.total_tokens_generated
        self.stats.total_draft_tokens_proposed += call_stats.total_draft_tokens_proposed
        self.stats.total_draft_tokens_accepted += call_stats.total_draft_tokens_accepted
        self.stats.total_target_forward_passes += call_stats.total_target_forward_passes
        self.stats.total_draft_forward_passes += call_stats.total_draft_forward_passes
        
        return actions, call_stats


# %%
# from experiments.specdec.vla_speculative_decoding import (
#     VLASpeculativeDecoder,
#     prepare_image,
# )
# FIXME: when temperature=0.0, current implementation uses 0.01...
cfg.gamma = 7

# Create speculative decoder
print("\n[3/4] Creating speculative decoder...")
# specdec_decoder = VLASpeculativeDecoderDDD(
#     target_model=target_model,
#     draft_model=draft_model,
#     target_processor=target_processor,
#     gamma=cfg.gamma,
#     use_cache=True,
#     temperature=cfg.temperature,
# )
# specdec_decoder = VLASpeculativeDecoderDDDR(
#     target_model=target_model,
#     draft_model=draft_model,
#     target_processor=target_processor,
#     gamma=cfg.gamma,
#     use_cache=True,
#     temperature=cfg.temperature,
#     relaxed_acceptance_r=7,
# )
# Use the new batched LM decoder for efficient verification
specdec_decoder = VLASpeculativeDecoderBatchedLM(
    target_model=target_model,
    draft_model=draft_model,
    target_processor=target_processor,
    gamma=cfg.gamma,
    temperature=cfg.temperature,
    relaxed_acceptance_r=7,  # Relaxed acceptance with r=7 bins
)

# Prepare PIL image for specdec
pil_image = prepare_image(observation["full_image"], center_crop=cfg.center_crop)

def run_specdec_inference():
    action, stats = specdec_decoder.predict_action_speculative(
        pil_image, task_description, unnorm_key_target
    )
    return action, stats

print("  Warming up SPECDEC...")
for _ in range(cfg.warmup_iterations):
    run_specdec_inference()
    torch.cuda.synchronize()

# Benchmark SPECDEC
print(f"\nBenchmarking SPECDEC ({cfg.num_iterations} iterations)...")
specdec_times = []
specdec_acceptance_rates = []
specdec_tokens_per_forward = []
specdec_decoder.reset_stats()

for i in range(cfg.num_iterations):
    (action, stats), dt = timed_cuda(run_specdec_inference)
    specdec_times.append(dt)
    specdec_acceptance_rates.append(stats.acceptance_rate)
    specdec_tokens_per_forward.append(stats.tokens_per_target_forward)
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{cfg.num_iterations}, last: {dt*1000:.1f}ms, "
                f"accept: {stats.acceptance_rate:.2%}")

try:
    env.close()
except:
    pass

def compute_statistics():
    # Compute statistics
    target_mean = np.mean(target_times)
    target_std = np.std(target_times)
    draft_mean = np.mean(draft_times)
    draft_std = np.std(draft_times)
    specdec_mean = np.mean(specdec_times)
    specdec_std = np.std(specdec_times)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTARGET:")
    print(f"  Mean time: {target_mean*1000:.2f} ± {target_std*1000:.2f} ms")
    print(f"  Throughput: {1/target_mean:.2f} Hz")

    print(f"\nDRAFT:")
    print(f"  Mean time: {draft_mean*1000:.2f} ± {draft_std*1000:.2f} ms")
    print(f"  Throughput: {1/draft_mean:.2f} Hz")

    print(f"\nSPECULATIVE DECODING (gamma={cfg.gamma}):")
    print(f"  Mean time: {specdec_mean*1000:.2f} ± {specdec_std*1000:.2f} ms")
    print(f"  Throughput: {1/specdec_mean:.2f} Hz")
    print(f"  Acceptance rate: {np.mean(specdec_acceptance_rates):.2%} ± {np.std(specdec_acceptance_rates):.2%}")
    print(f"  Tokens/target forward: {np.mean(specdec_tokens_per_forward):.2f}")

    print(f"\nSPEEDUPS:")
    print(f"  SpecDec vs Target: {target_mean/specdec_mean:.2f}x")
    print(f"  Draft vs Target: {target_mean/draft_mean:.2f}x")

    # Overall stats from decoder
    global_stats = specdec_decoder.stats
    print(f"\nOVERALL SPECDEC STATS:")
    print(f"  Total tokens generated: {global_stats.total_tokens_generated}")
    print(f"  Total draft tokens proposed: {global_stats.total_draft_tokens_proposed}")
    print(f"  Total draft tokens accepted: {global_stats.total_draft_tokens_accepted}")
    print(f"  Overall acceptance rate: {global_stats.acceptance_rate:.2%}")
    print(f"  Total target forward passes: {global_stats.total_target_forward_passes}")
    print(f"  Total draft forward passes: {global_stats.total_draft_forward_passes}")

    print("=" * 80)

compute_statistics()

# %%


# %%



