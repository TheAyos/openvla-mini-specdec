"""
Single-file, matching inference fastpath for MiniVLA (Prismatic `OpenVLA`) action prediction.

---------------
MiniVLA inference in this repo goes through `transformers.GenerationMixin.generate()` even though we only need
`action_dim` (typically 7) greedy tokens. For such short generations, Python-side generation overhead is noticeable.

This module provides:
- Cached prompt/tokenization for a **constant instruction** (common in robot rollouts).
- A tight greedy autoregressive loop using the same `past_key_values` mechanism as HF `generate()`.
- Optional, safe `torch.compile()` on the **LLM only** (largest observed win in existing benchmarks).
- Optional vision compile behind a flag (off by default).
- A built-in exactness verifier against `model.predict_action(...)`.

--------------------------------------------------------------

from experiments.specdec.minivla_fastpath import MiniVLAFastPath

draft_fast = MiniVLAFastPath(
    model=draft_model,
    instruction=task_description,
    unnorm_key=unnorm_key_draft,
    center_crop=cfg.center_crop,      # match your existing preprocessing
    compile_llm=True,                 # biggest safe win
    compile_mode="default",
    compile_vision=False,             # optional; can be brittle
)

def run_draft_inference():
    return draft_fast.predict_action_from_np(observation["full_image"])


=> If instruction changes, call `draft_fast.set_instruction(new_instruction)`

Run:
/mnt/scratch/aagouzoul/miniconda3/envs/mvla1311/bin/python -m experiments.specdec.minivla_fastpath \
  --compile-llm \
  --profile-dir /tmp/minivla_profile \
  --profile-steps 5

    python -m experiments.specdec.minivla_fastpath --help
    python -m experiments.specdec.minivla_fastpath --compile-llm
"""

from __future__ import annotations

import argparse
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# import logging

# torch._logging.set_logs(dynamo=logging.DEBUG)
# torch._logging.set_logs(graph=True)
# torch._logging.set_logs(fusion=True)

from prismatic.models.load import load_vla


def _read_hf_token(hf_token_path: Union[str, Path]) -> str:
    p = Path(hf_token_path)
    if p.exists():
        return p.read_text().strip()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        raise RuntimeError(f"No HF token found at `{p}` and no `HF_TOKEN`/`HUGGINGFACE_TOKEN` in env.")
    return tok


def load_minivla(checkpoint: Union[str, Path], hf_token: str, device: str = "cuda"):
    """Load MiniVLA (Prismatic `OpenVLA`) for inference in bf16."""
    vla = load_vla(str(checkpoint), hf_token=hf_token, load_for_training=False)

    # Verify full precision load.
    for p in vla.parameters():
        assert p.dtype == torch.float32, f"Loaded parameter not float32: {p.dtype}"

    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(device)
    vla.eval()
    return vla


def _apply_center_crop_np(im: np.ndarray, t_h: int, t_w: int) -> np.ndarray:
    # Matches `experiments/robot/openvla_utils.py::apply_center_crop`.
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]


def _maybe_center_crop(image: Image.Image, enabled: bool) -> Image.Image:
    """Match prismatic center-crop path in `experiments/robot/openvla_utils.py::get_prismatic_vla_action`."""
    if not enabled:
        return image
    crop_scale = 0.9
    sqrt_crop_scale = math.sqrt(crop_scale)
    temp = np.array(image)
    temp = _apply_center_crop_np(
        temp,
        t_h=int(sqrt_crop_scale * temp.shape[0]),
        t_w=int(sqrt_crop_scale * temp.shape[1]),
    )
    temp = Image.fromarray(temp)
    return temp.resize(image.size, Image.Resampling.BILINEAR)


def _to_device_pixel_values(
    pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values[None, ...].to(device)
    if isinstance(pixel_values, dict):
        return {k: v[None, ...].to(device) for k, v in pixel_values.items()}
    raise TypeError(f"Unsupported `pixel_values` type: {type(pixel_values)}")


def _maybe_compile(module: torch.nn.Module, *, mode: str, dynamic: bool, fullgraph: bool) -> Tuple[torch.nn.Module, bool]:
    if not hasattr(torch, "compile"):
        return module, False
    try:
        return torch.compile(module, mode=mode, dynamic=dynamic, fullgraph=fullgraph), True
    except Exception:
        return module, False


@dataclass
class ExactnessResult:
    ok: bool
    max_abs_diff: float


class MiniVLAFastPath:
    """
    Exact-match fastpath for `OpenVLA.predict_action()` for prismatic MiniVLA.
    Designed for the common case where the **instruction is constant** across many steps.
    """

    def __init__(
        self,
        model,
        instruction: str,
        unnorm_key: Optional[str],
        *,
        center_crop: bool = False,
        compile_llm: bool = True,
        compile_mode: str = "default",
        compile_vision: bool = False,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.autocast_dtype = model.llm_backbone.half_precision_dtype
        self.center_crop = center_crop
        self.unnorm_key = unnorm_key
        self.action_dim = model.get_action_dim(unnorm_key)

        self.image_transform = model.vision_backbone.get_image_transform()
        self.tokenizer = model.llm_backbone.tokenizer

        # Cache instruction => prompt => tokenization on device.
        self.set_instruction(instruction)

        # Optional compilation (best win: compile LLM only).
        self._compiled_llm = False
        self._compiled_vision = False
        if compile_llm:
            self.model.llm_backbone.llm, self._compiled_llm = _maybe_compile(
                self.model.llm_backbone.llm,
                mode=compile_mode,
                dynamic=True,   # KV cache shapes change with prompt length / decode steps # TODO: maybe what causes the Dynamo guard checks overhead
                fullgraph=False,
            )
        if compile_vision:
            vb = self.model.vision_backbone
            if hasattr(vb, "dino_featurizer") and hasattr(vb, "siglip_featurizer"):
                vb.dino_featurizer, ok1 = _maybe_compile(vb.dino_featurizer, mode=compile_mode, dynamic=False, fullgraph=False)
                vb.siglip_featurizer, ok2 = _maybe_compile(vb.siglip_featurizer, mode=compile_mode, dynamic=False, fullgraph=False)
                self._compiled_vision = ok1 and ok2
            elif hasattr(vb, "featurizer"):
                vb.featurizer, self._compiled_vision = _maybe_compile(vb.featurizer, mode=compile_mode, dynamic=False, fullgraph=False)

    def set_instruction(self, instruction: str) -> None:
        """Update the cached prompt/tokenization (call if instruction changes)."""
        self.instruction = instruction
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        input_ids = self.tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids

        # Match `prismatic/models/vlas/openvla.py` behavior.
        from transformers import LlamaTokenizerFast
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

        if isinstance(self.tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat((input_ids, torch.tensor([[29871]], dtype=torch.long)), dim=1)
        elif isinstance(self.tokenizer, Qwen2TokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported tokenizer type: {type(self.tokenizer)}")

        attention_mask = torch.ones_like(input_ids)
        self._input_ids = input_ids.to(self.device)
        self._attention_mask = attention_mask.to(self.device)

    @torch.inference_mode()
    def predict_action_from_pil(self, image: Union[Image.Image, list], *, unnorm_key: Optional[str] = None) -> np.ndarray:
        if isinstance(image, list):
            assert len(image) == 1, "Only single-image fastpath supported here (pass the first image)."
            image = image[0]

        image = image.convert("RGB")
        image = _maybe_center_crop(image, self.center_crop)

        pixel_values = self.image_transform(image)
        pixel_values = _to_device_pixel_values(pixel_values, self.device)

        unnorm_key = self.unnorm_key if unnorm_key is None else unnorm_key
        action_dim = self.model.get_action_dim(unnorm_key)

        # === Prefill (image + prompt) ===
        with torch.autocast("cuda", dtype=self.autocast_dtype, enabled=self.model.enable_mixed_precision_training):
            out = self.model(
                input_ids=self._input_ids,
                attention_mask=self._attention_mask,
                pixel_values=pixel_values,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

        # TODO: maybe don't predict for action_dim tokens, follow gamma
        # === Greedy decode (avoid per-step .item() syncs) ===
        generated = torch.empty((1, action_dim), device=self.device, dtype=torch.long)
        for i in range(action_dim):
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]
            generated[:, i] = next_token[:, 0]
            with torch.autocast("cuda", dtype=self.autocast_dtype, enabled=self.model.enable_mixed_precision_training):
                out = self.model.llm_backbone(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = out.past_key_values
            next_token_logits = out.logits[:, -1, :]

        token_ids = generated[0].cpu().numpy().astype(np.int64)
        normalized_actions = self.model.action_tokenizer.decode_token_ids_to_actions(token_ids)

        action_norm_stats = self.model.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        return np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)

    @torch.inference_mode()
    def predict_action_from_np(self, full_image_uint8: np.ndarray, *, unnorm_key: Optional[str] = None) -> np.ndarray:
        image = Image.fromarray(full_image_uint8).convert("RGB")
        return self.predict_action_from_pil(image, unnorm_key=unnorm_key)


@torch.inference_mode()
def verify_exact_match(
    model,
    fast: MiniVLAFastPath,
    image: Image.Image,
    instruction: str,
    unnorm_key: Optional[str],
    atol: float = 1e-6,
) -> ExactnessResult:
    base = model.predict_action(image, instruction, unnorm_key=unnorm_key)
    got = fast.predict_action_from_pil(image, unnorm_key=unnorm_key)
    max_abs = float(np.max(np.abs(base - got)))
    return ExactnessResult(ok=(max_abs <= atol), max_abs_diff=max_abs)


def _benchmark(fn, *, warmup: int, iters: int) -> Tuple[float, float]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        _ = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    ms = float(np.mean(times))
    hz = 1000.0 / ms
    return ms, hz


def _profile_to_dir(
    fn,
    *,
    profile_dir: Union[str, Path],
    steps: int,
    include_compile: bool,
) -> None:
    """
    Profile `fn()` and save traces to `profile_dir`.

    This is useful to inspect whether the compiled model is falling back to eager / recompiling (graph breaks).
    View exported chrome trace via https://ui.perfetto.dev/ or TensorBoard via `tensorboard --logdir <profile_dir>`.
    """
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    out_dir = Path(profile_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # By default, keep compilation (and its noise) out of the trace.
    if not include_compile:
        _ = fn()
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    handler = tensorboard_trace_handler(str(out_dir))
    sched = schedule(wait=0, warmup=0, active=max(1, steps), repeat=1)

    with profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(max(1, steps)):
            _ = fn()
            prof.step()

    # Also export a single chrome trace file for Perfetto.
    try:
        prof.export_chrome_trace(str(out_dir / "trace.json"))
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="Stanford-ILIAD/minivla-libero90-prismatic")
    ap.add_argument("--hf-token-path", type=str, default=".hf_token")
    ap.add_argument("--unnorm-key", type=str, default="libero_90")
    ap.add_argument("--instruction", type=str, default="pick up the red block")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--center-crop", action="store_true")
    ap.add_argument("--compile-llm", action="store_true")
    ap.add_argument("--compile-vision", action="store_true")
    ap.add_argument("--compile-mode", type=str, default="default")
    ap.add_argument("--profile-dir", type=str, default=None)
    ap.add_argument("--profile-steps", type=int, default=5)
    ap.add_argument("--profile-include-compile", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    assert torch.cuda.is_available() or args.device != "cuda", "CUDA not available."

    hf_token = _read_hf_token(args.hf_token_path)
    model = load_minivla(args.checkpoint, hf_token, device=args.device)

    # Fixed synthetic image for repeatability.
    rng = np.random.default_rng(0)
    img_np = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_np).convert("RGB")

    fast = MiniVLAFastPath(
        model=model,
        instruction=args.instruction,
        unnorm_key=args.unnorm_key,
        center_crop=args.center_crop,
        compile_llm=args.compile_llm,
        compile_mode=args.compile_mode,
        compile_vision=args.compile_vision,
    )

    # Verify exactness (if compile changes outputs, user can disable it).
    ex = verify_exact_match(model, fast, img, args.instruction, args.unnorm_key)
    print(f"[exactness] ok={ex.ok} max_abs_diff={ex.max_abs_diff:.3e}")
    if not ex.ok:
        raise SystemExit("Fastpath mismatch vs baseline. Disable compile flags or investigate numerical differences.")

    if args.profile_dir:
        _profile_to_dir(
            lambda: fast.predict_action_from_pil(img),
            profile_dir=args.profile_dir,
            steps=args.profile_steps,
            include_compile=args.profile_include_compile,
        )
        print(f"[profile] traces saved to: {args.profile_dir}")

    ms, hz = _benchmark(lambda: fast.predict_action_from_pil(img), warmup=args.warmup, iters=args.iters)
    print(f"[fastpath] {ms:.2f} ms  ({hz:.2f} Hz)  compiled_llm={fast._compiled_llm} compiled_vision={fast._compiled_vision}")


if __name__ == "__main__":
    main()


