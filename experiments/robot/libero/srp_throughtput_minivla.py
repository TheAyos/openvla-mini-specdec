"""
Measures inference throughput (actions/s) on a single LIBERO task.
Loads a task environment, gets a single observation, and runs inference multiple times to measure
average inference speed.

Usage:
    # For OpenVLA (base model):
    python experiments/robot/libero/srp_throughtput_minivla.py \
        --model_type openvla \
        --pretrained_checkpoint openvla/openvla-7b \
        --task_suite_name libero_spatial \
        --num_iterations 100

    # For Prismatic VLA (minivla):
    python experiments/robot/libero/srp_throughtput_minivla.py \
        --model_type prismatic \
        --pretrained_checkpoint Stanford-ILIAD/minivla-libero90-prismatic \
        --task_suite_name libero_90 \
        --num_iterations 100
"""

import os
os.environ['PRISMATIC_DATA_ROOT'] = '' # for minivla not to crash...
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import torch

# Append current directory so that interpreter can find experiments.robot
sys.path.append("./")
sys.path.append("../..")
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import get_processor, get_vla
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
from libero.libero import benchmark


@dataclass
class ThroughputConfig:
    # fmt: off
    
    # Model Type Selection
    model_type: str = "prismatic"                                    # Model type: "openvla" or "prismatic"
    
    # pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"  # Pretrained checkpoint path or HF model ID
    pretrained_checkpoint: Union[str, Path] = "Stanford-ILIAD/minivla-libero90-prismatic"  # Pretrained checkpoint path or HF model ID
    # pretrained_checkpoint: Union[str, Path] = "Stanford-ILIAD/minivla-vq-libero90-prismatic"  # Pretrained checkpoint path or HF model ID
    task_suite_name: str = "libero_90"                               # Task suite (libero_spatial, libero_object, etc.)
    task_id: int = 0                                                 # Which task to use from the suite (default: first task)
    num_iterations: int = 100                                        # Number of inference iterations to run
    warmup_iterations: int = 10                                      # Number of warmup iterations (not counted)
    load_in_8bit: bool = False                                       # Load with 8-bit quantization
    load_in_4bit: bool = False                                       # Load with 4-bit quantization
    center_crop: bool = True                                         # Center crop (set to True if model trained with augmentations)
    seed: int = 42                                                   # Random seed
    model_family: str = "openvla"                                  # Model family for environment compatibility
    hf_token: str = Path(".hf_token")                                # HuggingFace token path
    
    # fmt: on


@draccus.wrap()
def measure_throughput_with_single_libero_task(cfg: ThroughputConfig) -> dict:
    
    if not torch.cuda.is_available():
        print("hmmm, no gpu?!")
        return
    
    print("=" * 80)
    print("SRP throughput benchmark")
    print("=" * 80)
    print(cfg)
    print("=" * 80)

    assert cfg.model_type in ["openvla", "prismatic"], f"Invalid model_type: {cfg.model_type}"

    set_seed_everywhere(cfg.seed)

    # Set action un-normalization key based on task suite
    cfg.unnorm_key = cfg.task_suite_name

    # Load model and components based on model type
    print(f"\n[1/4] Loading {cfg.model_type.upper()} model...")
    if cfg.model_type == "prismatic":
        from experiments.robot.openvla_utils import get_prismatic_vla
        model = get_prismatic_vla(cfg)
        processor = None  # Prismatic doesn't use processor the same way
    else:
        model = get_vla(cfg)
        processor = get_processor(cfg)
    
    # Handle potential dataset name variations (e.g., "_no_noops" suffix)
    if hasattr(model, 'norm_stats'):
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        
        if cfg.unnorm_key not in model.norm_stats:
            print(f"WARNING: Un-normalization key '{cfg.unnorm_key}' not found in model norm_stats.")
            print(f"Available keys: {list(model.norm_stats.keys())}")
            print("Using first available key...")
            cfg.unnorm_key = list(model.norm_stats.keys())[0]
        
        print(f"Using un-normalization key: {cfg.unnorm_key}")

    print(f"[2/4] Initializing LIBERO task suite '{cfg.task_suite_name}'...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    
    task = task_suite.get_task(cfg.task_id)
    env, task_description = get_libero_env(task, model_family="openvla", resolution=224)
    
    print(f"Task description: {task_description}")
    
    # Get initial states and reset environment
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    env.reset()
    obs = env.set_init_state(initial_states[0])  # Use first initial state
    
    # Get image resize size
    resize_size = get_image_resize_size(cfg)
    
    # Prepare observation
    img = get_libero_image(obs, resize_size)
    observation = {
        "full_image": img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    
    print(f"Observation image shape: {img.shape}")
    print(f"Observation state shape: {observation['state'].shape}")

    # Run inference benchmark
    print(f"\n[3/4] Running throughput benchmark...")
    print(f"Warmup iterations: {cfg.warmup_iterations}")
    
    # Define inference function based on model type
    if cfg.model_type == "openvla":
        # Use standard OpenVLA inference
        from experiments.robot.openvla_utils import get_vla_action
        
        def run_inference():
            return get_vla_action(
                model,
                processor,
                cfg.pretrained_checkpoint,
                observation,
                task_description,
                cfg.unnorm_key,
                center_crop=cfg.center_crop
            )
    
    elif cfg.model_type == "prismatic":
        # Use Prismatic VLA inference
        from experiments.robot.openvla_utils import get_prismatic_vla_action
        
        def run_inference():
            return get_prismatic_vla_action(
                model,
                processor,
                cfg.pretrained_checkpoint,
                observation,
                task_description,
                cfg.unnorm_key,
                center_crop=cfg.center_crop
            )
    
    print("Running warmup...")
    for i in range(cfg.warmup_iterations):
        action = run_inference()
        if (i + 1) % 2 == 0:
            print(f"  Warmup: {i + 1}/{cfg.warmup_iterations} --> {action}")
        torch.cuda.synchronize()
        
    print(f"\nRunning {cfg.num_iterations} inference iterations...")
    start_time = time.time()
    
    for i in range(cfg.num_iterations):
        action = run_inference()
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{cfg.num_iterations}")
    
    torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_action = total_time / cfg.num_iterations
    actions_per_second = cfg.num_iterations / total_time
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Model: {cfg.pretrained_checkpoint}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per action: {avg_time_per_action * 1000:.2f} ms")
    print(f"Throughput: {actions_per_second:.2f} actions/second (Hz)")
    print(f"Throughput: {actions_per_second * 60:.2f} actions/minute")
    print("=" * 80)
    
    print(f"\nExample output={action} --> shape={action.shape}")
    
    env.close()
    
    return {
        "total_time": total_time,
        "avg_time_ms": avg_time_per_action * 1000,
        "actions_per_second": actions_per_second,
        "actions_per_minute": actions_per_second * 60,
        "num_iterations": cfg.num_iterations,
    }


# if __name__ == "__main__":
#     measure_throughput_with_single_libero_task()

if __name__ == "__main__":
    MODELS_TO_BENCH = [
        # ("openvla/openvla-7b", "OpenVLA-og", "openvla"),
        ("openvla/openvla-7b-finetuned-libero-spatial", "OpenVLA-og-ft-spatial", "openvla"),
        ("Stanford-ILIAD/minivla-libero90-prismatic", "MiniVLA-base", "prismatic"),
    ]

    common_args = {
        # "task_suite_name": "libero_spatial",
        # "task_id": 0,
        # "num_iterations": 50,
        # "resolution": 224,
    }

    for m in MODELS_TO_BENCH:
        mckpt, mname, mtype = m
        print(f"\n\n{'='*100}")
        print(f"Running benchmark for model: {mname}")
        # print(f"{'='*100}\n")
        cfg = ThroughputConfig(pretrained_checkpoint=mckpt, model_type=mtype)
        measure_throughput_with_single_libero_task.__wrapped__(cfg) # correct call when using draccus
