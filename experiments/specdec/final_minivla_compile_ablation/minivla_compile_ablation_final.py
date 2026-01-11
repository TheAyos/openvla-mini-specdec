# %% [markdown]
#   # LIBERO Rollout SR+Speed Benchmark for MiniVLAFastPath
# 
# 
# 
# 
# 
#   This notebook runs LIBERO rollout benchmarks for compiled MiniVLA variants.

# %% [markdown]
#   ## Setup and Imports

# %%
import os

# Keep these as early as possible for offscreen mujoco + prismatic load.
os.environ["PRISMATIC_DATA_ROOT"] = ""
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm
import sys
sys.path.append("/mnt/scratch/aagouzoul/ovla/openvla-mini")
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.specdec.minivla_fastpath import MiniVLAFastPath



# %% [markdown]
#   ## Helper Functions

# %%
def _select_unnorm_key(model, desired: str) -> str:
    """Match common repo logic for picking an available unnorm key."""
    if not hasattr(model, "norm_stats") or not isinstance(getattr(model, "norm_stats"), dict):
        return desired
    if desired in model.norm_stats:
        return desired
    if f"{desired}_no_noops" in model.norm_stats:
        return f"{desired}_no_noops"
    if desired.replace("_no_noops", "") in model.norm_stats:
        return desired.replace("_no_noops", "")
    # Fallback to first key to avoid crashing in quick benchmarks.
    # print in that case
    first_key = next(iter(model.norm_stats.keys()))
    print(f"WARNING: No unnorm key found for {desired}, using first available key: {first_key}")
    return first_key



# %%
def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    # Safe-ish fallback.
    return 400



# %%
def _summarize_ms(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 0:
        return {"n": 0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": float(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "hz": float(1000.0 / arr.mean()) if arr.mean() > 0 else float("nan"),
    }



# %% [markdown]
#   ## Configuration

# %%
@dataclass
class EvalConfig:
    # Model
    model_family: str = "prismatic"
    hf_token: Union[str, Path] = Path(".hf_token")
    pretrained_checkpoint: Union[str, Path] = "Stanford-ILIAD/minivla-libero90-prismatic"
    center_crop: bool = True

    # LIBERO
    task_suite_name: str = "libero_90"
    task_id: int = 0
    num_tasks: int = 1
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    max_steps_override: Optional[int] = None
    seed: int = 7

    # FastPath
    use_fastpath: bool = True
    compile_llm: bool = False
    compile_vision: bool = False
    compile_mode: str = "default"
    compile_mode_vision: Optional[str] = None  # if None, uses compile_mode; otherwise independent
    warmup_policy_calls: int = 1  # warmup calls per task before timing

    # Logging / stats
    show_progress: bool = True
    save_video: bool = False  # Save a local MP4 replay using `save_rollout_video` (no wandb).



# %% [markdown]
#   ## Run Benchmark

# %%
def run(cfg: EvalConfig) -> None:
    assert cfg.model_family == "prismatic", "This benchmark is for MiniVLA (prismatic) only."
    assert cfg.num_trials_per_task >= 1
    assert cfg.num_tasks >= 1

    set_seed_everywhere(cfg.seed)

    # Load model
    cfg.unnorm_key = cfg.task_suite_name  # expected by `get_model` helpers in this repo
    model = get_model(cfg)
    cfg.unnorm_key = _select_unnorm_key(model, cfg.task_suite_name)

    resize_size = get_image_resize_size(cfg)
    env_resolution = resize_size  # matches `run_libero_eval.py` for prismatic

    # Prepare task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    n_tasks = task_suite.n_tasks
    task_ids = list(range(cfg.task_id, min(cfg.task_id + cfg.num_tasks, n_tasks)))

    # Initialize fastpath once (keeps compilation one-time); update instruction per-task.
    fast: Optional[MiniVLAFastPath] = None

    total_episodes = 0
    total_successes = 0
    all_policy_gpu_ms: List[float] = []

    # CUDA timing helpers
    assert torch.cuda.is_available(), "CUDA required for accurate GPU timing"
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # Track compilation time separately (happens once on first forward pass)
    compilation_time_ms: Optional[float] = None
    global_warmup_done = False

    # Match `run_libero_eval.py` style: tqdm over tasks AND tqdm over trials-per-task.
    for task_id in tqdm.tqdm(task_ids, desc=f"Tasks({cfg.task_suite_name})", disable=not cfg.show_progress):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=env_resolution)
        initial_states = task_suite.get_task_init_states(task_id)

        # Per-task fastpath update (instruction usually constant throughout rollout).
        if cfg.use_fastpath:
            if fast is None:
                vision_mode = cfg.compile_mode_vision if cfg.compile_mode_vision is not None else cfg.compile_mode

                fast = MiniVLAFastPath(
                    model=model,
                    instruction=task_description,
                    unnorm_key=cfg.unnorm_key,
                    center_crop=cfg.center_crop,
                    compile_llm=cfg.compile_llm,
                    compile_mode=cfg.compile_mode, # LLM mode / vision default
                    compile_vision=cfg.compile_vision,
                    compile_mode_vision=vision_mode,
                )
            else:
                fast.set_instruction(task_description)

        max_steps = cfg.max_steps_override if cfg.max_steps_override is not None else _max_steps_for_suite(cfg.task_suite_name)

        # Rollouts
        for episode_idx in tqdm.tqdm(
            range(cfg.num_trials_per_task),
            desc=f"Trials(task={task_id})",
            disable=not cfg.show_progress,
        ):
            total_episodes += 1
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            success = False
            ep_policy_gpu_ms: List[float] = []
            replay_images: List[np.ndarray] = []

            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Preprocess image (matches repo's prismatic pathway).
                img = get_libero_image(obs, resize_size, model_family=cfg.model_family)
                if cfg.save_video:
                    # Save the processed frames (same as `run_libero_eval.py`).
                    replay_images.append(img)

                # Build observation (proprio is not used by prismatic/openvla models, but keep structure).
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                # Global warmup (ONCE before any timed calls) - includes compilation time measurement
                if not global_warmup_done:
                    torch.cuda.synchronize()
                    
                    # First call: measure compilation time (happens lazily on first forward pass)
                    compile_start = torch.cuda.Event(enable_timing=True)
                    compile_end = torch.cuda.Event(enable_timing=True)
                    compile_start.record()
                    
                    if cfg.use_fastpath:
                        _ = fast.predict_action_from_np(observation["full_image"], unnorm_key=cfg.unnorm_key)
                    else:
                        from experiments.robot.robot_utils import get_action
                        _ = get_action(cfg, model, observation, task_description, processor=None)
                    
                    compile_end.record()
                    torch.cuda.synchronize()
                    compilation_time_ms = float(compile_start.elapsed_time(compile_end))
                    
                    # Additional warmup calls to stabilize GPU state
                    for _ in range(max(0, cfg.warmup_policy_calls - 1)):
                        if cfg.use_fastpath:
                            _ = fast.predict_action_from_np(observation["full_image"], unnorm_key=cfg.unnorm_key)
                        else:
                            _ = get_action(cfg, model, observation, task_description, processor=None)
                    
                    torch.cuda.synchronize()
                    global_warmup_done = True

                # === Policy call timing (GPU time only) ===
                torch.cuda.synchronize()
                start_ev.record()

                if cfg.use_fastpath:
                    action = fast.predict_action_from_np(observation["full_image"], unnorm_key=cfg.unnorm_key)
                else:
                    # Baseline path (slower): uses `get_prismatic_vla_action` under the hood.
                    from experiments.robot.robot_utils import get_action
                    action = get_action(cfg, model, observation, task_description, processor=None)

                end_ev.record()
                torch.cuda.synchronize()
                ep_policy_gpu_ms.append(float(start_ev.elapsed_time(end_ev)))

                # Postprocess + env step (matches `run_libero_eval.py`).
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1

            if success:
                total_successes += 1
            if cfg.save_video:
                # Saves to ./rollouts/<DATE>/... (same behavior as `run_libero_eval.py`).
                try:
                    save_rollout_video(
                        replay_images,
                        total_episodes,
                        success=success,
                        task_description=task_description,
                        log_file=None,
                    )
                except Exception as e:
                    print(f"WARNING: Failed to save rollout video: {e}")

            all_policy_gpu_ms.extend(ep_policy_gpu_ms)

        try:
            env.close()
        except Exception:
            pass

    # Summary
    print("\n" + "=" * 80)
    print("LIBERO ROLLOUT BENCHMARK SUMMARY (MiniVLA)")
    print("=" * 80)
    print(f"Suite: {cfg.task_suite_name}")
    print(f"Tasks: {task_ids}")
    print(f"Episodes: {total_episodes}")
    print(f"Successes: {total_successes}")
    print(f"Success rate: {total_successes / max(1, total_episodes):.2%}")
    print(f"Mode: {'fastpath' if cfg.use_fastpath else 'baseline'}")
    if cfg.use_fastpath:
        print(f"FastPath: compile_llm={cfg.compile_llm}, compile_vision={cfg.compile_vision}, compile_mode={cfg.compile_mode}")
    print(f"Unnorm key: {cfg.unnorm_key}")
    print(f"Center crop: {cfg.center_crop}")
    print(f"Warmup calls: {cfg.warmup_policy_calls}")
    print("-" * 80)

    if compilation_time_ms is not None:
        print(f"First-call time (includes compilation if enabled): {compilation_time_ms:.2f} ms")
    
    s_gpu = _summarize_ms(all_policy_gpu_ms)
    print("Policy GPU time (ms) stats (excludes warmup/compilation):")
    print(f"  n={int(s_gpu['n'])} mean={s_gpu['mean_ms']:.2f} std={s_gpu['std_ms']:.2f} "
          f"p50={s_gpu['p50_ms']:.2f} p90={s_gpu['p90_ms']:.2f} p95={s_gpu['p95_ms']:.2f} "
          f"min={s_gpu['min_ms']:.2f} max={s_gpu['max_ms']:.2f}  => {s_gpu['hz']:.2f} Hz")
    print("=" * 80)



# %%
cfg=EvalConfig(
    # Model
    model_family="prismatic",
    hf_token=Path("/pub/scratch/aagouzoul/ovla/openvla-mini/.hf_token"),
    pretrained_checkpoint="Stanford-ILIAD/minivla-libero90-prismatic",
    center_crop=True,
    
    # LIBERO
    task_suite_name="libero_90",
    task_id=0,
    num_tasks=1,
    num_trials_per_task=1,
    num_steps_wait=10,
    max_steps_override=None,
    seed=7,
    
    # FastPath
    use_fastpath=True,
    compile_llm=False,
    compile_vision=False,
    compile_mode="default",
    warmup_policy_calls=3,
    
    # Logging / stats
    show_progress=False,
    save_video=False,
)
# run(cfg)


# %% [markdown]
#  ## Ablation Study: Compilation Options
# 
# 
# 
#  This cell runs ablations over:
# 
#  - `compile_llm`: True/False
# 
#  - `compile_vision`: True/False
# 
#  - `compile_mode`: "default" / "max-autotune-no-cudagraphs"
# 
# 
# 
#  Total of 7 runs:
# 
#  1. Baseline: compile_llm=False, compile_vision=False
# 
#  2. (--4) compile_mode="default" with (T,F), (F,T), (T,T)
# 
#  5. (--7) compile_mode="max-autotune-no-cudagraphs" with (T,F), (F,T), (T,T)

# %%
import matplotlib.pyplot as plt
from dataclasses import replace
from typing import Dict, List, Tuple, Any
import pandas as pd

def run_ablation(cfg: EvalConfig) -> Dict[str, Any]:
    """
    Run a single ablation configuration and return results.
    Modified version of run() that returns metrics instead of just printing.
    """
    import gc
    
    # === CRITICAL: Reset torch.compile/Dynamo state between ablations ===
    # Without this, compiled graphs from previous ablations interfere with new ones,
    # causing wildly inconsistent results (e.g., 13.8Hz vs 6.4Hz for same config).
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except Exception:
        pass
    
    # Force garbage collection to free previous model/compiled modules
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        
    assert cfg.model_family == "prismatic", "This benchmark is for MiniVLA (prismatic) only."
    assert cfg.num_trials_per_task >= 1
    assert cfg.num_tasks >= 1

    set_seed_everywhere(cfg.seed)

    # Load model
    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg)
    cfg.unnorm_key = _select_unnorm_key(model, cfg.task_suite_name)

    resize_size = get_image_resize_size(cfg)
    env_resolution = resize_size

    # Prepare task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    n_tasks = task_suite.n_tasks
    task_ids = list(range(cfg.task_id, min(cfg.task_id + cfg.num_tasks, n_tasks)))

    fast: Optional[MiniVLAFastPath] = None

    total_episodes = 0
    total_successes = 0
    all_policy_gpu_ms: List[float] = []

    # CUDA timing helpers
    assert torch.cuda.is_available(), "CUDA required for accurate GPU timing"
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # Track compilation time separately (happens once on first forward pass)
    compilation_time_ms: Optional[float] = None
    global_warmup_done = False

    for task_id in tqdm.tqdm(task_ids, desc=f"Tasks({cfg.task_suite_name})", disable=not cfg.show_progress):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=env_resolution)
        initial_states = task_suite.get_task_init_states(task_id)

        if cfg.use_fastpath:
            if fast is None:
                fast = MiniVLAFastPath(
                    model=model,
                    instruction=task_description,
                    unnorm_key=cfg.unnorm_key,
                    center_crop=cfg.center_crop,
                    compile_llm=cfg.compile_llm,
                    compile_mode=cfg.compile_mode,
                    compile_vision=cfg.compile_vision,
                )
            else:
                fast.set_instruction(task_description)

        max_steps = cfg.max_steps_override if cfg.max_steps_override is not None else _max_steps_for_suite(cfg.task_suite_name)

        for episode_idx in tqdm.tqdm(
            range(cfg.num_trials_per_task),
            desc=f"Trials(task={task_id})",
            disable=not cfg.show_progress,
        ):
            total_episodes += 1
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            success = False
            ep_policy_gpu_ms: List[float] = []

            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                img = get_libero_image(obs, resize_size, model_family=cfg.model_family)

                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                # Global warmup (ONCE before any timed calls) - includes compilation time measurement
                if not global_warmup_done:
                    torch.cuda.synchronize()
                    
                    # First call: measure compilation time (happens lazily on first forward pass)
                    compile_start = torch.cuda.Event(enable_timing=True)
                    compile_end = torch.cuda.Event(enable_timing=True)
                    compile_start.record()
                    
                    if cfg.use_fastpath:
                        _ = fast.predict_action_from_np(observation["full_image"], unnorm_key=cfg.unnorm_key)
                    else:
                        from experiments.robot.robot_utils import get_action
                        _ = get_action(cfg, model, observation, task_description, processor=None)
                    
                    compile_end.record()
                    torch.cuda.synchronize()
                    compilation_time_ms = float(compile_start.elapsed_time(compile_end))
                    
                    # Additional warmup calls to stabilize GPU state
                    for _ in range(max(0, cfg.warmup_policy_calls - 1)):
                        if cfg.use_fastpath:
                            _ = fast.predict_action_from_np(observation["full_image"], unnorm_key=cfg.unnorm_key)
                        else:
                            _ = get_action(cfg, model, observation, task_description, processor=None)
                    
                    torch.cuda.synchronize()
                    global_warmup_done = True

                # === Policy call timing (GPU time only) ===
                torch.cuda.synchronize()
                start_ev.record()

                if cfg.use_fastpath:
                    action = fast.predict_action_from_np(observation["full_image"], unnorm_key=cfg.unnorm_key)
                else:
                    from experiments.robot.robot_utils import get_action
                    action = get_action(cfg, model, observation, task_description, processor=None)

                end_ev.record()
                torch.cuda.synchronize()
                ep_policy_gpu_ms.append(float(start_ev.elapsed_time(end_ev)))

                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1

            if success:
                total_successes += 1

            all_policy_gpu_ms.extend(ep_policy_gpu_ms)

        try:
            env.close()
        except Exception:
            pass

    # Compute summary statistics
    gpu_stats = _summarize_ms(all_policy_gpu_ms)

    result = {
        "config": {
            "compile_llm": cfg.compile_llm,
            "compile_vision": cfg.compile_vision,
            "compile_mode": cfg.compile_mode,
            "compile_mode_vision": cfg.compile_mode_vision,
            "use_fastpath": cfg.use_fastpath,
        },
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / max(1, total_episodes),
        "gpu_stats": gpu_stats,
        "compilation_time_ms": compilation_time_ms,
        "all_policy_gpu_ms": all_policy_gpu_ms,
    }
    
    # === CRITICAL: Cleanup to prevent interference with subsequent ablations ===
    # Delete model and compiled objects to free GPU memory and clear compiled state
    del fast
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return result


# %%
# Define ablation configurations
ablation_configs = [
    # # Baseline: no compilation
    {"compile_llm": False, "compile_vision": False, "compile_mode": "default", "label": "Baseline (no compile)"},
    
    {"compile_llm": True, "compile_vision": False, "compile_mode": "default", "label": "LLM only (default)"},
    {"compile_llm": True, "compile_vision": False, "compile_mode": "max-autotune-no-cudagraphs", "label": "LLM only (max-autotune)"},
    
    {"compile_llm": False, "compile_vision": True, "compile_mode": "default", "label": "Vision only (default)"},
    {"compile_llm": False, "compile_vision": True, "compile_mode": "max-autotune-no-cudagraphs", "label": "Vision only (max-autotune)"},
    {"compile_llm": False, "compile_vision": True, "compile_mode": "reduce-overhead", "label": "Vision only (reduce-overhead)"},
    
    {"compile_llm": True, "compile_vision": True, "compile_mode": "default", "label": "LLM+Vision (default)"},
    {"compile_llm": True, "compile_vision": True, "compile_mode": "max-autotune-no-cudagraphs", "label": "LLM+Vision (max-autotune)"},
    
    {"compile_llm": True, "compile_vision": True, "compile_mode": "default", "compile_mode_vision": "max-autotune-no-cudagraphs", "label": "LLM(default)+Vision(max-autotune)"},
    {"compile_llm": True, "compile_vision": True, "compile_mode": "default", "compile_mode_vision": "reduce-overhead", "label": "LLM(default)+Vision(reduce-overhead)"},
    # {"compile_llm": True, "compile_vision": True, "compile_mode": "max-autotune-no-cudagraphs", "compile_mode_vision": "default", "label": "LLM(max-autotune)+Vision(default)"},
]


# %%
# Run all ablations
import time as time_module  # for delays between ablations
ablation_results = []

DELAY_BETWEEN_ABLATIONS_SEC = 2.0  # Let GPU settle between ablations

for i, ablation in enumerate(ablation_configs):
    print(f"\n{'='*80}")
    print(f"Running ablation {i+1}/{len(ablation_configs)}: {ablation['label']}")
    print(f"{'='*80}")
    
    # Create config for this ablation
    ablation_cfg = replace(
        cfg,
        compile_llm=ablation["compile_llm"],
        compile_vision=ablation["compile_vision"],
        compile_mode=ablation["compile_mode"],
        compile_mode_vision=ablation.get("compile_mode_vision", None),
    )
    
    # Run and store results
    result = run_ablation(ablation_cfg)
    result["label"] = ablation["label"]
    ablation_results.append(result)
    
    # Print intermediate results
    print(f"Success rate: {result['success_rate']:.2%}")
    if result["compilation_time_ms"] is not None:
        print(f"First-call (compilation) time: {result['compilation_time_ms']:.2f} ms")
    if result["gpu_stats"]:
        print(f"GPU Hz: {result['gpu_stats'].get('hz', 'N/A'):.2f}")
    
    # Delay between ablations to let GPU settle (helps with thermal throttling, memory cleanup)
    if i < len(ablation_configs) - 1:
        time_module.sleep(DELAY_BETWEEN_ABLATIONS_SEC)

print(f"\n{'='*80}")
print("All ablations complete!")
print(f"{'='*80}")

# %%
# Save ablation results to disk

import pickle
import os
from datetime import datetime

# date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# results_file = f"ablation_results_dict_{date_str}.pkl"
# with open(results_file, "wb") as f:
#     pickle.dump(ablation_results, f)

# print(f"Saved ablation results to {results_file}")

# Reload ablation results when needed
date_str = "20260110_212918" # date to load
def load_ablation_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ablation results file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)
    
results_file = f"ablation_results_dict_{date_str}.pkl"
ablation_results = load_ablation_results(results_file)

# %%
# Create summary DataFrame
summary_data = []
for r in ablation_results:
    row = {
        "Configuration": r["label"],
        "compile_llm": r["config"]["compile_llm"],
        "compile_vision": r["config"]["compile_vision"],
        "compile_mode": r["config"]["compile_mode"],
        "compile_mode_vision": r["config"].get("compile_mode_vision", r["config"]["compile_mode"]),
        "Success Rate": r["success_rate"],
        "Episodes": r["total_episodes"],
        "Successes": r["total_successes"],
        "Compilation (ms)": r.get("compilation_time_ms", None),
    }
    
    # Add GPU stats
    if r["gpu_stats"]:
        row["GPU Mean (ms)"] = r["gpu_stats"].get("mean_ms", None)
        row["GPU Std (ms)"] = r["gpu_stats"].get("std_ms", None)
        row["GPU P50 (ms)"] = r["gpu_stats"].get("p50_ms", None)
        row["GPU P95 (ms)"] = r["gpu_stats"].get("p95_ms", None)
        row["GPU Hz"] = r["gpu_stats"].get("hz", None)
    
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
print("\nAblation Summary:")
display(df_summary)


# %%
# # Plotting
# plt.style.use("seaborn-v0_8-whitegrid")

# fig, axes = plt.subplots(2, 2, figsize=(12, 13))
# fig.suptitle("MiniVLA Compilation Ablation Study", fontsize=18, fontweight="bold")

# labels = [r["label"] for r in ablation_results]
# x = np.arange(len(labels))
# bar_width = 0.75

# # Color scheme: baseline=gray, default=blue, max-autotune=green
# colors = ["#595959", "#1f77b4", "#4a9fd4", "#7ec8e3", "#2ca02c", "#5fd35f", "#98e698"]

# common_bar_kwargs = dict(width=bar_width, edgecolor="black", linewidth=0.5)

# def format_axis(ax, title, ylabel, rotate=True):
#     ax.set_title(title, fontsize=12, pad=8)
#     ax.set_ylabel(ylabel, fontsize=11)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=35 if rotate else 0, ha="right", fontsize=9)
#     ax.tick_params(axis="y", labelsize=9)
#     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

# def add_value_labels(ax, bars, fmt="{:.1f}", offset=0.01):
#     ylim = ax.get_ylim()
#     span = ylim[1] - ylim[0]
#     for bar in bars:
#         val = bar.get_height()
#         if val <= 0:
#             continue
#         ax.text(
#             bar.get_x() + bar.get_width() / 2,
#             val + span * offset,
#             fmt.format(val),
#             ha="center",
#             va="bottom",
#             fontsize=8,
#             rotation=0,
#         )

# # # 1. Success Rate
# # ax1 = axes[0, 0]
# # success_rates = [r["success_rate"] * 100 for r in ablation_results]
# # bars1 = ax1.bar(x, success_rates, color=colors[:len(x)], **common_bar_kwargs)
# # format_axis(ax1, "Success Rate", "Success Rate (%)")
# # ax1.set_ylim(0, 105)
# # add_value_labels(ax1, bars1, fmt="{:.1f}", offset=0.02)

# # 2. GPU Hz (Throughput)
# ax2 = axes[0, 0]
# gpu_hz = [r["gpu_stats"].get("hz", 0) if r["gpu_stats"] else 0 for r in ablation_results]
# bars2 = ax2.bar(x, gpu_hz, color=colors[:len(x)], **common_bar_kwargs)
# format_axis(ax2, "GPU Throughput (Hz)", "Frequency (Hz)")
# add_value_labels(ax2, bars2, fmt="{:.1f}", offset=0.02)

# # 3. Compilation Time
# ax3 = axes[0, 1]
# compile_times = [r.get("compilation_time_ms", 0) or 0 for r in ablation_results]
# bars3 = ax3.bar(x, compile_times, color=colors[:len(x)], **common_bar_kwargs)
# format_axis(ax3, "First Call Time (incl. compilation)", "Time (ms)")
# add_value_labels(ax3, bars3, fmt="{:.0f}", offset=0.02)

# # 4. GPU Mean Latency
# ax4 = axes[1, 0]
# gpu_mean = [r["gpu_stats"].get("mean_ms", 0) if r["gpu_stats"] else 0 for r in ablation_results]
# bars4 = ax4.bar(x, gpu_mean, color=colors[:len(x)], **common_bar_kwargs)
# format_axis(ax4, "GPU Mean Latency (lower is better)", "Latency (ms)")
# add_value_labels(ax4, bars4, fmt="{:.1f}", offset=0.02)

# # # 5. GPU P95 Latency
# # ax5 = axes[2, 0]
# # gpu_p95 = [r["gpu_stats"].get("p95_ms", 0) if r["gpu_stats"] else 0 for r in ablation_results]
# # bars5 = ax5.bar(x, gpu_p95, color=colors[:len(x)], **common_bar_kwargs)
# # format_axis(ax5, "GPU P95 Latency (lower is better)", "Latency (ms)")
# # add_value_labels(ax5, bars5, fmt="{:.1f}", offset=0.02)

# # 6. Speedup vs Baseline
# ax6 = axes[1, 1]
# baseline_gpu_mean = ablation_results[0]["gpu_stats"].get("mean_ms", 1) if ablation_results[0]["gpu_stats"] else 1
# gpu_speedup = [
#     baseline_gpu_mean / r["gpu_stats"].get("mean_ms", baseline_gpu_mean)
#     if r["gpu_stats"] and r["gpu_stats"].get("mean_ms", 0) > 0 else 1.0
#     for r in ablation_results
# ]
# bars6 = ax6.bar(x, gpu_speedup, color=colors[:len(x)], **common_bar_kwargs)
# ax6.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline = 1×")
# format_axis(ax6, "GPU Speedup vs Baseline (higher is better)", "Speedup (×)")
# add_value_labels(ax6, bars6, fmt="{:.2f}×", offset=0.02)
# ax6.legend(fontsize=9, loc="upper right", frameon=True)

# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig("ablation_results.png", dpi=200, bbox_inches="tight")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Reorder ablation_results by speedup (baseline fixed)
# -------------------------------
ablation_results_sorted = list(ablation_results)  # shallow copy

baseline = ablation_results_sorted[0]
others = ablation_results_sorted[1:]

baseline_gpu_mean = (
    baseline["gpu_stats"].get("mean_ms", 1)
    if baseline.get("gpu_stats")
    else 1
)

others_sorted = sorted(
    others,
    key=lambda r: baseline_gpu_mean / r["gpu_stats"].get("mean_ms", baseline_gpu_mean)
    if r.get("gpu_stats") and r["gpu_stats"].get("mean_ms", 0) > 0
    else 1.0,
    reverse=True,
)

ablation_results_sorted = [baseline] + others_sorted

# -------------------------------
# Plotting
# -------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(12, 13))
fig.suptitle("MiniVLA Compilation Ablation Study", fontsize=18, fontweight="bold")

labels = [r["label"] for r in ablation_results_sorted]
x = np.arange(len(labels))
bar_width = 0.75

# Color scheme: baseline=gray, default=blue, max-autotune=green
colors = ["#595959", "#1f77b4", "#4a9fd4", "#7ec8e3", "#2ca02c", "#5fd35f", "#98e698"]

common_bar_kwargs = dict(width=bar_width, edgecolor="black", linewidth=0.5)

def format_axis(ax, title, ylabel, rotate=True):
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35 if rotate else 0, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

def add_value_labels(ax, bars, fmt="{:.1f}", offset=0.01):
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for bar in bars:
        val = bar.get_height()
        if val <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + span * offset,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )

# -------------------------------
# 1. GPU Throughput (Hz)
# -------------------------------
ax2 = axes[0, 0]
gpu_hz = [
    r["gpu_stats"].get("hz", 0) if r.get("gpu_stats") else 0
    for r in ablation_results_sorted
]
bars2 = ax2.bar(x, gpu_hz, color=colors[:len(x)], **common_bar_kwargs)
format_axis(ax2, "GPU Throughput (Hz)", "Frequency (Hz)")
add_value_labels(ax2, bars2, fmt="{:.1f}", offset=0.02)

# -------------------------------
# 2. Compilation Time
# -------------------------------
ax3 = axes[0, 1]
compile_times = [
    r.get("compilation_time_ms", 0) or 0
    for r in ablation_results_sorted
]
bars3 = ax3.bar(x, compile_times, color=colors[:len(x)], **common_bar_kwargs)
format_axis(ax3, "First Call Time (incl. compilation)", "Time (ms)")
add_value_labels(ax3, bars3, fmt="{:.0f}", offset=0.02)

# -------------------------------
# 3. GPU Mean Latency
# -------------------------------
ax4 = axes[1, 0]
gpu_mean = [
    r["gpu_stats"].get("mean_ms", 0) if r.get("gpu_stats") else 0
    for r in ablation_results_sorted
]
bars4 = ax4.bar(x, gpu_mean, color=colors[:len(x)], **common_bar_kwargs)
format_axis(ax4, "GPU Mean Latency (lower is better)", "Latency (ms)")
add_value_labels(ax4, bars4, fmt="{:.1f}", offset=0.02)

# -------------------------------
# 4. Speedup vs Baseline
# -------------------------------
ax6 = axes[1, 1]
gpu_speedup = [
    baseline_gpu_mean / r["gpu_stats"].get("mean_ms", baseline_gpu_mean)
    if r.get("gpu_stats") and r["gpu_stats"].get("mean_ms", 0) > 0
    else 1.0
    for r in ablation_results_sorted
]

bars6 = ax6.bar(x, gpu_speedup, color=colors[:len(x)], **common_bar_kwargs)
ax6.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline = 1×")
format_axis(ax6, "GPU Speedup vs Baseline (higher is better)", "Speedup (×)")
add_value_labels(ax6, bars6, fmt="{:.2f}×", offset=0.02)
ax6.legend(fontsize=9, loc="upper right", frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
filename = f"ablation_results_{date_str}.png"
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nsaved graph:\n   {filename}")


# %%
# Latency distribution plots (box plots)
fig2, ax_gpu = plt.subplots(figsize=(10, 6))
fig2.suptitle("GPU Latency Distributions Across Configurations", fontsize=14, fontweight="bold")

# GPU latency distributions
gpu_data = [r["all_policy_gpu_ms"] for r in ablation_results_sorted if r["all_policy_gpu_ms"]]
gpu_labels = [r["label"] for r in ablation_results_sorted if r["all_policy_gpu_ms"]]
if gpu_data:
    bp1 = ax_gpu.boxplot(gpu_data, labels=gpu_labels, patch_artist=True)
    for patch, color in zip(bp1["boxes"], colors[:len(gpu_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_gpu.set_ylabel("Latency (ms)")
    ax_gpu.set_title("GPU Latency Distribution (excludes warmup/compilation)")
    ax_gpu.tick_params(axis="x", rotation=45)
    for label in ax_gpu.get_xticklabels():
        label.set_ha("right")
        label.set_fontsize(8)

plt.tight_layout()
filename = f"ablation_latency_distributions_{date_str}.png"
plt.savefig(filename, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nsaved graph:\n   {filename}")

# %%
# Grouped comparison: Default vs Max-Autotune for each compile option
fig3, ax_c1 = plt.subplots(figsize=(8, 5))
fig3.suptitle("Compile Mode Comparison: Default vs Max-Autotune", fontsize=14, fontweight="bold")

compile_options = ["LLM only", "Vision only", "LLM+Vision"]
x_comp = np.arange(len(compile_options))
width = 0.35

# Extract data for comparison
default_gpu_hz = [ablation_results[1]["gpu_stats"].get("hz", 0),  # LLM only (default)
                  ablation_results[3]["gpu_stats"].get("hz", 0),  # Vision only (default)
                  ablation_results[6]["gpu_stats"].get("hz", 0)]  # LLM+Vision (default)

maxauto_gpu_hz = [ablation_results[2]["gpu_stats"].get("hz", 0),  # LLM only (max-autotune)
                  ablation_results[4]["gpu_stats"].get("hz", 0),  # Vision only (max-autotune)
                  ablation_results[7]["gpu_stats"].get("hz", 0)]  # LLM+Vision (max-autotune)

baseline_hz = ablation_results[0]["gpu_stats"].get("hz", 0) if ablation_results[0]["gpu_stats"] else 0

# GPU Hz comparison
bars_c1a = ax_c1.bar(x_comp - width/2, default_gpu_hz, width, label="Default", color="#1f77b4")
bars_c1b = ax_c1.bar(x_comp + width/2, maxauto_gpu_hz, width, label="Max-Autotune", color="#2ca02c")
ax_c1.axhline(y=baseline_hz, color="red", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_hz:.1f} Hz)")
ax_c1.set_ylabel("Frequency (Hz)")
ax_c1.set_title("GPU Throughput by Compile Mode")
ax_c1.set_xticks(x_comp)
ax_c1.set_xticklabels(compile_options)
ax_c1.legend()

plt.tight_layout()
filename = f"ablation_mode_comparison_{date_str}.png"
plt.savefig(filename, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nsaved graph:\n   {filename}")


# %%
# Print final summary table
summary_lines = []
summary_lines.append("\n" + "=" * 100)
summary_lines.append("FINAL ABLATION SUMMARY")
summary_lines.append("=" * 100)
summary_lines.append(f"\n{'Configuration':<35} {'Success Rate':>12} {'GPU Hz':>10} {'GPU Speedup':>12} {'Compile (ms)':>14}")
summary_lines.append("-" * 100)

for i, r in enumerate(ablation_results_sorted):
    gpu_hz_val = r["gpu_stats"].get("hz", 0) if r["gpu_stats"] else 0
    gpu_spd = gpu_speedup[i]
    compile_ms = r.get("compilation_time_ms", 0) or 0
    summary_lines.append(f"{r['label']:<35} {r['success_rate']*100:>11.1f}% {gpu_hz_val:>10.2f} {gpu_spd:>11.2f}x {compile_ms:>14.1f}")

summary_lines.append("=" * 100)

# Best configuration
best_gpu_idx = np.argmax(gpu_hz_arr := [r["gpu_stats"].get("hz", 0) if r["gpu_stats"] else 0 for r in ablation_results_sorted])

summary_lines.append(f"\nBest GPU throughput: {ablation_results_sorted[best_gpu_idx]['label']} ({gpu_hz_arr[best_gpu_idx]:.2f} Hz)")

# Print to console
for line in summary_lines:
    print(line)

# Save to text file
summary_filename = f"ablation_summary_{date_str}.txt"
with open(summary_filename, 'w') as f:
    for line in summary_lines:
        f.write(line + '\n')

print(f"\nsaved summary to:\n   {summary_filename}")

# %%



