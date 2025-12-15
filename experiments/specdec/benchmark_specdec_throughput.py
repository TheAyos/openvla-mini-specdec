"""
benchmark_specdec_throughput.py

Benchmarks inference throughput for speculative decoding vs baseline models.
Uses a single LIBERO task observation for controlled comparison.

Usage:
    python -m experiments.specdec.benchmark_specdec_throughput \
        --target_checkpoint /pub/scratch/aagouzoul/ovla/openvla-mini/ft_experiments_logs/openvla-7b+libero_90_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug+libero_90_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --draft_checkpoint Stanford-ILIAD/minivla-libero90-prismatic \
        --task_suite_name libero_90 \
        --num_iterations 20 \
        --warmup_iterations 5 \
        --gamma 4 \
        --temperature 0.0 \
        --center_crop True \
        --seed 42
"""

import os
os.environ['PRISMATIC_DATA_ROOT'] = ''

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import torch
from libero.libero import benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import get_processor, get_vla, get_prismatic_vla
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
from experiments.specdec.vla_speculative_decoding import (
    VLASpeculativeDecoder,
    prepare_image,
)


@dataclass
class BenchmarkConfig:
    # fmt: off
    
    # Model checkpoints
    target_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"
    draft_checkpoint: Union[str, Path] = "Stanford-ILIAD/minivla-libero90-prismatic"
    
    # Speculative decoding parameters
    gamma: int = 4
    temperature: float = 0.0
    
    # Benchmark parameters
    task_suite_name: str = "libero_90"
    task_id: int = 0
    num_iterations: int = 50
    warmup_iterations: int = 10
    
    # Model loading
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    
    seed: int = 42
    hf_token: str = Path(".hf_token")
    
    # fmt: on


def timed_cuda(fn):
    """Time a function using CUDA events for accurate GPU timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000  # Convert to seconds


@draccus.wrap()
def run_benchmark(cfg: BenchmarkConfig) -> None:
    """Run throughput benchmark."""
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
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
    
    # Load models
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
    
    # Create speculative decoder
    print("\n[3/4] Creating speculative decoder...")
    specdec_decoder = VLASpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        target_processor=target_processor,
        gamma=cfg.gamma,
        use_cache=True,
        temperature=cfg.temperature,
    )
    
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
    
    # Prepare PIL image for specdec
    pil_image = prepare_image(observation["full_image"], center_crop=cfg.center_crop)
    
    # Define inference functions
    def run_target_inference():
        from experiments.robot.openvla_utils import get_vla_action
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"calling get_vla_action(target_model..., target_processor..., cfg.target_checkpoint={cfg.target_checkpoint}, observation=..., task_description={task_description}, unnorm_key={unnorm_key_target}, center_crop={cfg.center_crop})")
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
        print("\033[38;2;255;165;0m[SRP] -> \033[0m", f"calling get_prismatic_vla_action(draft_model..., processor=None, cfg.draft_checkpoint={cfg.draft_checkpoint}, observation..., task_description={task_description}, unnorm_key={unnorm_key_draft}, center_crop={cfg.center_crop})")
        return get_prismatic_vla_action(
            draft_model,
            None,
            str(cfg.draft_checkpoint),
            observation,
            task_description,
            unnorm_key_draft,
            center_crop=cfg.center_crop,
        )
    
    def run_specdec_inference():
        action, stats = specdec_decoder.predict_action_speculative(
            pil_image, task_description, unnorm_key_target
        )
        return action, stats
    
    # Warmup
    print(f"\nRunning warmup ({cfg.warmup_iterations} iterations each)...")
    
    print("  Warming up TARGET...")
    for _ in range(cfg.warmup_iterations):
        run_target_inference()
        torch.cuda.synchronize()
    
    print("  Warming up DRAFT...")
    for _ in range(cfg.warmup_iterations):
        run_draft_inference()
        torch.cuda.synchronize()
    
    print("  Warming up SPECDEC...")
    for _ in range(cfg.warmup_iterations):
        run_specdec_inference()
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
    
    env.close()
    
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


if __name__ == "__main__":
    run_benchmark()

