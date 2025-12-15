"""
vla_speculative_decoding.py

Speculative decoding for Vision-Language-Action models.
Uses MiniVLA as the draft model and OpenVLA as the target model.

Based on: Leviathan et al. "Fast Inference from Transformers via Speculative Decoding"
https://arxiv.org/abs/2211.17192

The key idea: use a smaller, faster model (draft) to propose multiple tokens,
then verify them in parallel with the larger model (target).
"""

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
                        generated_token_ids.append(draft_token_id_target)
                        n_accepted += 1
                        call_stats.total_tokens_generated += 1
                        call_stats.total_draft_tokens_accepted += 1
                        
                        if len(generated_token_ids) >= action_dim:
                            break
                    else:
                        # Reject - sample from adjusted distribution
                        adjusted_probs = max_fn(target_prob - draft_prob_remapped)
                        if adjusted_probs.sum() > 0:
                            corrected_token = torch.multinomial(adjusted_probs, num_samples=1)
                        else:
                            corrected_token = self._sample_token(target_prob.unsqueeze(0))
                        
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

