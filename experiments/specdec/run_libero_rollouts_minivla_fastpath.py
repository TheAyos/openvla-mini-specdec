"""
LIBERO rollout SR+speed benchmark for MiniVLA + `MiniVLAFastPath`.

Run one rollout (default):
    python -m experiments.specdec.run_libero_rollouts_minivla_fastpath

Run 5 rollouts on task 0, with compiled LLM:
    python -m experiments.specdec.run_libero_rollouts_minivla_fastpath \\
      --num_trials_per_task 5 --compile_llm True

Evaluate first 8 tasks, 1 rollout each:
    python -m experiments.specdec.run_libero_rollouts_minivla_fastpath \\
      --task_id 0 --num_tasks 8 --num_trials_per_task 1
"""

import os

# Keep these as early as possible for offscreen mujoco + prismatic load.
os.environ.setdefault("PRISMATIC_DATA_ROOT", "")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import draccus
import gc
import numpy as np
import torch
import tqdm
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
    warmup_policy_calls: int = 1  # warmup calls per task before timing (helps compiled steady-state)

    # Logging / stats
    show_progress: bool = True
    save_video: bool = False  # Save a local MP4 replay using `save_rollout_video` (no wandb).


@draccus.wrap()
def run(cfg: EvalConfig) -> None:
    # Best-effort cleanup to reduce cross-run interference when benchmarking in the same process
    # (e.g., after a prior `torch.compile` run in a notebook).
    try:
        import torch._dynamo  # type: ignore

        torch._dynamo.reset()
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
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
    all_policy_wall_ms: List[float] = []

    # CUDA timing helpers
    use_cuda_events = torch.cuda.is_available()
    start_ev = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    end_ev = torch.cuda.Event(enable_timing=True) if use_cuda_events else None

    # Match `run_libero_eval.py` style: tqdm over tasks AND tqdm over trials-per-task.
    for task_id in tqdm.tqdm(task_ids, desc=f"Tasks({cfg.task_suite_name})", disable=not cfg.show_progress):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=env_resolution)
        initial_states = task_suite.get_task_init_states(task_id)

        # Per-task fastpath update (instruction usually constant throughout rollout).
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
            ep_policy_wall_ms: List[float] = []
            replay_images: List[np.ndarray] = []

            # Optional warmup (use first observation image after wait).
            warmed = False

            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Preprocess image (matches repo’s prismatic pathway).
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

                # Warmup calls (not timed) to stabilize compilation/allocations.
                if cfg.use_fastpath and (not warmed) and cfg.warmup_policy_calls > 0:
                    for _ in range(cfg.warmup_policy_calls):
                        _ = fast.predict_action_from_np(observation["full_image"], unnorm_key=cfg.unnorm_key)
                    if use_cuda_events:
                        torch.cuda.synchronize()
                    warmed = True

                # === Policy call timing ===
                t0 = time.perf_counter()
                if use_cuda_events:
                    torch.cuda.synchronize()
                    start_ev.record()

                if cfg.use_fastpath:
                    action = fast.predict_action_from_np(observation["full_image"], unnorm_key=cfg.unnorm_key)
                else:
                    # Baseline path (slower): uses `get_prismatic_vla_action` under the hood.
                    from experiments.robot.robot_utils import get_action

                    action = get_action(cfg, model, observation, task_description, processor=None)

                if use_cuda_events:
                    end_ev.record()
                    torch.cuda.synchronize()
                    ep_policy_gpu_ms.append(float(start_ev.elapsed_time(end_ev)))
                t1 = time.perf_counter()
                ep_policy_wall_ms.append((t1 - t0) * 1000.0)

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
            all_policy_wall_ms.extend(ep_policy_wall_ms)

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
    print("-" * 80)

    if torch.cuda.is_available():
        s_gpu = _summarize_ms(all_policy_gpu_ms)
        print("Policy GPU time (ms) stats:")
        print(f"  n={int(s_gpu['n'])} mean={s_gpu['mean_ms']:.2f} std={s_gpu['std_ms']:.2f} "
              f"p50={s_gpu['p50_ms']:.2f} p90={s_gpu['p90_ms']:.2f} p95={s_gpu['p95_ms']:.2f} "
              f"min={s_gpu['min_ms']:.2f} max={s_gpu['max_ms']:.2f}  => {s_gpu['hz']:.2f} Hz")
    else:
        print("Policy GPU time: CUDA not available, skipping.")

    s_wall = _summarize_ms(all_policy_wall_ms)
    print("Policy wall time (ms) stats:")
    print(f"  n={int(s_wall['n'])} mean={s_wall['mean_ms']:.2f} std={s_wall['std_ms']:.2f} "
          f"p50={s_wall['p50_ms']:.2f} p90={s_wall['p90_ms']:.2f} p95={s_wall['p95_ms']:.2f} "
          f"min={s_wall['min_ms']:.2f} max={s_wall['max_ms']:.2f}  => {s_wall['hz']:.2f} Hz")
    print("=" * 80)


if __name__ == "__main__":
    run()


