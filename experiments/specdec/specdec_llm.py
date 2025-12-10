# SRP
# code in this file is based on https://github.com/romsto/Speculative-Decoding

import time
import torch
import abc
from typing import List, Tuple, Union
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from transformers.cache_utils import DynamicCache
from termcolor import colored

# ==========================================
# Logits Processor
# ==========================================

class LogitsProcessor(abc.ABC):
    """Logits processors for sampling."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        proc = self._process(logits)
        return F.softmax(proc / self.temperature, dim=-1)

    @abc.abstractmethod
    def _process(self, logits: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        pass


class GreedyProcessor(LogitsProcessor):
    """Greedy: Most probable token."""

    def __init__(self, temperature: float = 1):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits

    def sample(self, probs: Tensor) -> Tensor:
        return torch.argmax(probs, dim=-1).unsqueeze(-1)
    
# ==========================================
# Caching Utils
# ==========================================

def prune_cache(cache: Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache], num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end.
    """
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return prune_tuple_cache(cache, num_tokens_to_discard)
    elif isinstance(cache, DynamicCache):
        return prune_dynamic_cache(cache, num_tokens_to_discard)
    else:
        raise ValueError("Unsupported cache type.")


def prune_tuple_cache(cache: Tuple[Tuple[Tensor, Tensor]], num_tokens_to_discard: int):
    if cache is None:
        return None

    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue

        layer = []
        for i in range(len(layer_cache)):
            tensor = layer_cache[i]
            new_tensor = tensor[:, :, :-num_tokens_to_discard, :]
            layer.append(new_tensor)
        new_cache.append(tuple(layer))

    return tuple(new_cache)


def prune_dynamic_cache(cache: DynamicCache, num_tokens_to_discard: int):
    if cache is None:
        return None

    for layer in range(len(cache)):
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :-num_tokens_to_discard, :]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :-num_tokens_to_discard, :]
    cache._seen_tokens -= num_tokens_to_discard

    return cache

# ==========================================
# Generation Functions
# ==========================================

@torch.no_grad()
def autoregressive_generate(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
) -> List[int]:
    """
    Generate text sequence autoregressively based on the input sequence.
    """
    cache = None
    prompt_len = len(inputs)
    # prepare input tensor
    max_seq_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else (model.config.max_context_length if hasattr(model.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    # input tensor intially just pad tokens
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=model.device)
    # then fill in with prompt, remaining are pad tokens!
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)

    # stop token(s)
    list_tokens_id = (eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id])
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=model.device)

    # start from end of prompt up to total generation length
    for curr in range(prompt_len, total_len):
        o = model(input_ids[..., :curr], past_key_values=cache, use_cache=use_cache)
        # only care about prediction for last position
        logits = o.logits[..., -1, :]  # [1, vocab_size]
        # raw logits |-> probabilities (softmax)
        probs = logits_processor(logits)  # [1, vocab_size]
        x = logits_processor.sample(probs)  # [1, 1]
        # save newly sampled token x
        input_ids[0, curr] = x
        # update kv cache
        cache = o.past_key_values

        # check for end token
        if torch.isin(x, stop_tokens):
            break

    return input_ids[0, prompt_len : curr + 1].tolist()

def max_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Max function.
        x: input tensor.
    Returns:
        tensor norm(max(0, x)).
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum

@torch.no_grad()
def speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
) -> Tuple[List[int], float]:
    """
    Generate text sequence using the speculative decoding algorithm.
    """
    
    drafter_cache, target_cache = None, None

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
    
    drafts_accepted, drafts_speculated = .0, .0
    
    vocabulary_size = target.config.vocab_size
        
    # prepare input tensor
    prompt_len = len(inputs)
    max_seq_length = target.config.max_position_embeddings if hasattr(target.config, 'max_position_embeddings') else (target.config.max_context_length if hasattr(target.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)
    
    current_position = prompt_len
    
    if first_target:
        # run the target model before the speculative algorithm. Allows to prefill the kvcache and get a first token.
        M_TARGET = target(
            input_ids=input_ids[..., :current_position],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = M_TARGET.past_key_values
        p_p = logits_processor(M_TARGET.logits[..., -1, :])
        t = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1
        
        if torch.isin(t, stop_tokens):
            return input_ids[0, prompt_len:current_position].tolist(), 0
        
    
    while current_position < total_len:
        # clamp gamma
        corrected_gamma = min(gamma, total_len - current_position - 1)
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)
        
        input_ids = input_ids.to(drafter.device)
        
        # generate gamma drafts
        for k in range(corrected_gamma):
            M_DRAFT = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = M_DRAFT.past_key_values
            
            draft_logits = M_DRAFT.logits[..., -1, :]
            draft_probs = logits_processor(draft_logits)
            # q is draft token probas
            q[0, k] = draft_probs.to(target.device)
            # xi are "guesses"
            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi
        drafts_speculated += corrected_gamma
        input_ids = input_ids.to(target.device)
        
        # run target model on drafts and get logits of the previous tokens plus one more token (verification)
        M_TARGET = target(
            input_ids=input_ids[..., :current_position + corrected_gamma], #MAGIC HERE
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = M_TARGET.past_key_values
        draft_logits = M_TARGET.logits[..., current_position - 1:current_position + corrected_gamma - 1, :] # [1, corrected_gamma, vocab_size]
        p = logits_processor(draft_logits) # [1, gamma, vocab_size]
        
        # compute the last accepted draft position (rejection sampling)
        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]: # if r > P/Q reject, i.e. if Mp thinks the token is much less likely than Mq, reject it
                n = i
                break
        
        drafts_accepted += n
        
        # check if the end token is in the drafts
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), drafts_accepted / drafts_speculated

        if n == corrected_gamma:
            p_p = M_TARGET.logits[..., current_position + corrected_gamma - 1, :]
            p_p = logits_processor(p_p)
        else:
            # prune the cache ("polluted" by calculations from the bad guesses)
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
                target_cache = prune_cache(target_cache, corrected_gamma - n + 1)
            
            # adjust the distribution (the trick that preserves the output distribution!)
            if not skip_sample_adjustment:
                p_p = max_fn(p[..., n, :] - q[0, n, :])
            else:
                p_p = p[..., n, :]
        # generate the correct token to replace the FIRST bad draft
        x = logits_processor.sample(p_p)
        
        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = x
        
        current_position += n + 1
        
        if torch.isin(x, stop_tokens):
            return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / drafts_speculated
    
    return input_ids[0, prompt_len:].tolist(), drafts_accepted / drafts_speculated

if __name__ == "__main__":
    
    print_outputs = False # print generated outputs by models
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    target_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    drafter_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    quantize_config = QuantoConfig(weights="int8")
    
    # prompt_text = "Explain the theory of relativity in simple terms."
    # prompt_text = "im currently a 3rd year bachelor pursuing my first research project on inference speed optimization. any tips for success?"
    prompt_text = "Explain speculative decoding for Transformers in simple terms."
    gen_len = 256
    gamma = 4
    use_cache = False # True to use KV caching #TODO: what does openvla do ?
    
    print(f"Loading target: {target_model_name}")
    target = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        quantization_config=quantize_config,
        device_map=device,
        trust_remote_code=True,
    )
    target.eval()
    # ASSUME TARGET AND DRAFT HAVE SAME TOKENIZER!
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
    
    print(f"Loading drafter: {drafter_model_name}")
    drafter = AutoModelForCausalLM.from_pretrained(
        drafter_model_name,
        quantization_config=quantize_config,
        device_map=device,
        trust_remote_code=True,
    )
    drafter.eval()
    
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False)
    tokenized = tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()
    
    end_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    processor = GreedyProcessor()

    print(f"\nPrompt: {prompt_text}")
    print(f"Generation Length: {gen_len}")
    print(f"Gamma: {gamma}")
    print("-" * 50)
    
    # warmup
    print("Warmup target pass...")
    output_ids = autoregressive_generate(
        tokenized,
        target,
        use_cache=use_cache,
        max_gen_len=32,
        eos_tokens_id=end_tokens,
        logits_processor=processor,
    )
    print("Warmup draft pass...")
    output_ids = autoregressive_generate(
        tokenized,
        drafter,
        use_cache=use_cache,
        max_gen_len=32,
        eos_tokens_id=end_tokens,
        logits_processor=processor,
    )

    # 1. Target AR
    torch.manual_seed(42)
    start_time = time.time()
    output_ids = autoregressive_generate(
        tokenized,
        target,
        use_cache=use_cache,
        max_gen_len=gen_len,
        eos_tokens_id=end_tokens,
        logits_processor=processor,
    )
    end_time = time.time()
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    # print(output)
    # print()
    # print(output_ids)
    target_chars_per_sec = len(output) / (end_time - start_time)
    num_tokens = len(output_ids)
    target_tokens_per_sec = num_tokens / (end_time - start_time)
    
    print(colored("\n=========== Target AR ===========", "cyan"))
    if print_outputs:
        print(colored("Out:", "cyan"), output)
    print(colored(f"Time: {end_time - start_time:.2f}s", "light_red"))
    print(colored(f"Throughput: {target_chars_per_sec:.1f} chars/s", "light_red"))
    print(colored(f"Throughput: {target_tokens_per_sec:.1f} tokens/s", "red"))
    print(colored("=========== Target AR ===========", "cyan"))

    # 2. Drafter AR
    torch.manual_seed(42)
    start_time = time.time()
    output_ids = autoregressive_generate(
        tokenized,
        drafter,
        use_cache=use_cache,
        max_gen_len=gen_len,
        eos_tokens_id=end_tokens,
        logits_processor=processor,
    )
    end_time = time.time()
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    drafter_chars_per_sec = len(output) / (end_time - start_time)
    num_tokens = len(output_ids)
    drafter_tokens_per_sec = num_tokens / (end_time - start_time)

    print(colored("\n========== Drafter AR ==========", "cyan"))
    if print_outputs:
        print(colored("Out:", "cyan"), output)
    print(colored(f"Time: {end_time - start_time:.2f}s", "light_red"))
    print(colored(f"Throughput: {drafter_chars_per_sec:.1f} chars/s", "light_red"))
    print(colored(f"Throughput: {drafter_tokens_per_sec:.1f} tokens/s", "red"))
    print(colored("========== Drafter AR ==========", "cyan"))

    # 3. Speculative Decoding
    torch.manual_seed(42)
    start_time = time.time()
    output_ids, accept_rate = speculative_generate(
        tokenized,
        drafter,
        target,
        gamma=gamma,
        logits_processor=processor,
        max_gen_len=gen_len,
        eos_tokens_id=end_tokens,
        use_cache=use_cache,
    )
    end_time = time.time()
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    spec_chars_per_sec = len(output) / (end_time - start_time)
    num_tokens = len(output_ids)
    spec_tokens_per_sec = num_tokens / (end_time - start_time)

    print(colored("\n========== Speculative ==========", "green"))
    if print_outputs:
        print(colored("Out:", "green"), output)
    print(colored(f"Acceptance rate (α): {accept_rate:.3f}", "light_red"))
    print(colored(f"Time: {end_time - start_time:.2f}s", "light_red"))
    print(colored(f"Throughput: {spec_chars_per_sec:.1f} chars/s", "light_red"))
    print(colored(f"Throughput: {spec_tokens_per_sec:.1f} tokens/s", "red"))
    print(colored("========== Speculative ==========", "green"))

    if target_tokens_per_sec > 0:
        speedup_chars = spec_chars_per_sec / target_chars_per_sec
        speedup_tokens = spec_tokens_per_sec / target_tokens_per_sec
        print(colored(f"Speedup vs Target: {speedup_chars:.2f}x (chars/s)", "light_magenta"))
        print(colored(f"Speedup vs Target: {speedup_tokens:.2f}x (tokens/s)", "light_magenta"))

