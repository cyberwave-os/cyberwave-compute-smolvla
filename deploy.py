#!/usr/bin/env python3
"""SmolVLA inference with Cyberwave integration.

Entry point for cloud deployment. Loads the SmolVLA model once, then uses
CwProcessor to handle all Cyberwave I/O and data transformations.

This file is intentionally minimal - it only handles:
- Model loading (with PEFT/LoRA support)
- Building a predict_fn that takes inputs and returns raw action tensors

All Cyberwave-related concerns (camera mapping, joint names, action conversion)
are handled by CwProcessor and the SmolVLAResolver.

Usage:
    python deploy.py '<json_payload>' or /path/to/params.json

Environment variables:
    SMOLVLA_CHECKPOINT: Local path to SmolVLA checkpoint directory
    SMOLVLA_BASE_MODEL: Optional base model for PEFT checkpoints
    HF_TOKEN: Hugging Face token (fewer rate limits; faster hub metadata)
    SMOLVLA_HF_OFFLINE: If 1/true, use HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE (requires warm cache)
    SMOLVLA_SKIP_VLM_PREFETCH: If 1/true, skip snapshot_download of vlm_model_name before load
    SMOLVLA_SKIP_GPU_WARMUP: If 1/true, skip the post-load CUDA warmup forward (first real cycle pays JIT cost)
    CUDA_MODULE_LOADING: Set to LAZY for faster CUDA init (export CUDA_MODULE_LOADING=LAZY)

Note on PEFT vs full finetune: merged PEFT and a full saved policy are the same architecture for inference.
The long first _get_action_chunk is almost always one-time CUDA/cuDNN/attention kernel setup on that graph.
Training never shows it as a separate "inference" line—the first optimizer step pays the same cost inside step 0.
Deploy had no prior forward, so cycle 1 looked slow until the optional GPU warmup runs after load.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.utils import build_inference_frame
from lerobot.utils.constants import ACTION, OBS_STATE

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from cw_processor import (
    CwProcessor,
    PredictFn,
    download_weights,
    load_json_argument,
    parse_request_payload,
)

logger = logging.getLogger(__name__)

MODEL_SLUG = "smolvla"


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _configure_cuda_for_inference() -> None:
    """Configure CUDA/cuDNN/dynamo for faster cold start (less autotuning overhead)."""
    if not torch.cuda.is_available():
        return
    # Disable cuDNN benchmark — skips algorithm search on first forward.
    # Slight steady-state perf hit but much faster warmup.
    torch.backends.cudnn.benchmark = False
    # Deterministic algorithms (no random kernel selection).
    torch.backends.cudnn.deterministic = True
    # TF32 on Ampere+ for speed (slightly less precision, fine for inference).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Disable torch.compile / dynamo / inductor entirely — avoids JIT overhead.
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.suppress_errors = True
        _dynamo.disable()
    except Exception:
        pass


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dataset_features_from_policy(cfg: SmolVLAConfig) -> dict[str, dict]:
    """LeRobot dataset-style feature dict for build_inference_frame / make_robot_action."""
    out: dict[str, dict] = {}
    if OBS_STATE not in cfg.input_features:
        raise ValueError(
            "Policy config has no observation.state; SmolVLA cloud inference expects proprioception."
        )
    pf_state = cfg.input_features[OBS_STATE]
    n = int(pf_state.shape[0])
    out[OBS_STATE] = {
        "dtype": "float32",
        "shape": (n,),
        "names": [f"state_{i}" for i in range(n)],
    }
    for key, pf in cfg.input_features.items():
        if pf.type != FeatureType.VISUAL:
            continue
        c, h, w = pf.shape
        out[key] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }
    for _key, pf in cfg.output_features.items():
        if pf.type == FeatureType.ACTION:
            na = int(pf.shape[0])
            out[ACTION] = {
                "dtype": "float32",
                "shape": (na,),
                "names": [f"joint_{i}" for i in range(na)],
            }
            break
    if ACTION not in out:
        raise ValueError("Policy config has no ACTION output feature.")
    return out


def _warmup_inputs_for_policy(cfg: SmolVLAConfig, state_dim: int) -> dict[str, Any]:
    """Synthetic batch matching training cameras (zeros) for one CUDA warmup forward."""
    images: dict[str, np.ndarray] = {}
    for key, pf in cfg.input_features.items():
        if pf.type != FeatureType.VISUAL:
            continue
        if "empty_camera" in key:
            continue
        short = key.split("observation.images.")[-1]
        c, h, w = pf.shape
        images[short] = np.zeros((h, w, c), dtype=np.uint8)
    return {
        "images": images,
        "state": np.zeros(state_dim, dtype=np.float32),
        "instruction": "warmup",
    }


def _align_state(state: np.ndarray, expected_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < expected_dim:
        state = np.pad(state, (0, expected_dim - state.shape[0]))
    elif state.shape[0] > expected_dim:
        logger.warning("Truncating state from %s to %s", state.shape[0], expected_dim)
        state = state[:expected_dim]
    return state


def raw_observation_from_tensors(
    cfg: SmolVLAConfig,
    ds_features: dict[str, dict],
    images: dict[str, np.ndarray],
    state: np.ndarray,
) -> dict:
    """Build a robot-like observation dict for `build_inference_frame` (keys match dataset features)."""
    raw: dict = {}
    st_spec = ds_features[OBS_STATE]
    dim = st_spec["shape"][0]
    state = _align_state(state, dim)
    for i, name in enumerate(st_spec["names"]):
        raw[name] = float(state[i])

    for key, pf in cfg.input_features.items():
        if pf.type != FeatureType.VISUAL:
            continue
        short = key.split("observation.images.")[-1]
        if "empty_camera" in key:
            c, h, w = pf.shape
            raw[short] = np.zeros((h, w, c), dtype=np.uint8)
            continue
        if short not in images:
            raise KeyError(
                f"Missing image array for camera {short!r} (policy expects {key}). "
                f"Got keys: {sorted(images)}"
            )
        img = images[short]
        if img.ndim != 3:
            raise ValueError(f"Image {short} must be HWC (uint8 RGB); got shape {img.shape}")
        # Writable copy avoids torch.from_numpy warnings in LeRobot (read-only buffer views).
        raw[short] = np.asarray(img, dtype=np.uint8).copy()

    return raw


def _is_peft_checkpoint(checkpoint: str) -> bool:
    """Check if checkpoint is a PEFT (LoRA) adapter checkpoint."""
    checkpoint_path = Path(checkpoint)
    adapter_config = checkpoint_path / "adapter_config.json"
    adapter_model = checkpoint_path / "adapter_model.safetensors"
    model_safetensors = checkpoint_path / "model.safetensors"

    # PEFT checkpoint has adapter files but no full model
    return adapter_config.exists() and adapter_model.exists() and not model_safetensors.exists()


def _get_base_model_from_peft_config(checkpoint: str) -> str | None:
    """Extract base model path from PEFT adapter_config.json.

    Only returns the base_model_name_or_path if it's:
    - A HuggingFace hub model (not a local path), or
    - A local path with model.safetensors (full model, not another PEFT checkpoint)

    Returns None if the base model is another PEFT checkpoint (we'll use default).
    """
    adapter_config_path = Path(checkpoint) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    try:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path")

        if not base_model:
            return None

        # If it's a local path, check if it has a full model (not another PEFT checkpoint)
        base_path = Path(base_model)
        if base_path.exists():
            model_file = base_path / "model.safetensors"
            if not model_file.exists():
                logger.warning(
                    "Base model from adapter_config (%s) is another PEFT checkpoint. "
                    "Using default base model instead.",
                    base_model,
                )
                return None

        return base_model
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read adapter_config.json: %s", e)
        return None


def _apply_hf_offline_from_env() -> bool:
    """If SMOLVLA_HF_OFFLINE is set, force hub clients to use local cache only."""
    raw = os.environ.get("SMOLVLA_HF_OFFLINE", "").strip().lower()
    if raw not in ("1", "true", "yes"):
        return False
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return True


def _prefetch_vlm_snapshot(checkpoint: str) -> None:
    """Download SmolVLM (and tokenizer/processor files) into HF cache before policy load.

    Without this, `from_pretrained` + tokenizer processors issue many sequential hub
    HEAD/GET calls — often the dominant cost before the first inference (more than compile).
    """
    if _apply_hf_offline_from_env():
        logger.info("HF offline mode: skipping VLM prefetch (using local cache only)")
        return
    if os.environ.get("SMOLVLA_SKIP_VLM_PREFETCH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return

    cfg_path = Path(checkpoint) / "config.json"
    if not cfg_path.is_file():
        return
    try:
        with open(cfg_path, encoding="utf-8") as f:
            raw = json.load(f)
        vlm_id = raw.get("vlm_model_name")
        if not vlm_id or not isinstance(vlm_id, str):
            return
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read vlm_model_name from %s: %s", cfg_path, e)
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not available; VLM prefetch skipped")
        return

    logger.info(
        "Prefetching VLM into HF cache: %s (one-time network; speeds later hub access)",
        vlm_id,
    )
    try:
        snapshot_download(repo_id=vlm_id, repo_type="model")
    except Exception as e:
        logger.warning("VLM prefetch failed (non-fatal, load may hit hub again): %s", e)


def build_predict_fn(
    checkpoint: str,
    robot_type: str = "",
    device: torch.device | None = None,
    base_model: str | None = None,
) -> PredictFn:
    """Load SmolVLA model and return a predict_fn for per-step inference.

    The predict_fn takes a simple inputs dict and returns raw action tensors.
    All camera/joint mapping is handled by CwProcessor.

    Args:
        checkpoint: HuggingFace hub ID or local path to SmolVLA checkpoint
        robot_type: Optional robot_type string for multi-embodiment models
        device: torch device (defaults to cuda/mps/cpu auto-detect)
        base_model: For PEFT checkpoints, the base model to load

    Returns:
        predict_fn(inputs) -> raw action tensor [1, chunk_size, action_dim]
    """
    device = device or _device()
    if device.type == "cuda":
        _configure_cuda_for_inference()
        logger.info("CUDA configured: benchmark=off, deterministic=on, dynamo=disabled")
    logger.info("Loading SmolVLA policy from %s on device %s", checkpoint, device)

    _prefetch_vlm_snapshot(checkpoint)

    # Check if this is a PEFT (LoRA) checkpoint
    if _is_peft_checkpoint(checkpoint):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT checkpoint detected but peft library not installed. "
                "Install with: pip install peft"
            )

        # Determine base model (priority: argument > env var > adapter_config > default)
        env_base_model = os.environ.get("SMOLVLA_BASE_MODEL")
        logger.info("SMOLVLA_BASE_MODEL env var: %s", env_base_model)

        if base_model is None:
            base_model = env_base_model
        if base_model is None:
            base_model = _get_base_model_from_peft_config(checkpoint)
            logger.info("Base model from adapter_config: %s", base_model)
        if base_model is None:
            base_model = "lerobot/smolvla_base"

        logger.info("PEFT checkpoint detected. Using base model: %s", base_model)
        policy = SmolVLAPolicy.from_pretrained(base_model)

        logger.info("Applying PEFT adapter from: %s", checkpoint)
        peft_policy = PeftModel.from_pretrained(policy, checkpoint)
        policy = peft_policy.merge_and_unload()
        logger.info("PEFT adapter merged successfully")
    else:
        policy = SmolVLAPolicy.from_pretrained(checkpoint)

    # Cloud deploy: never torch.compile — checkpoints often have compile_model=True from
    # training, which adds a large one-time compile cost before the first prediction.
    policy.config.compile_model = False
    logger.info("Inference: compile_model forced off for faster cold start")

    policy.reset()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    cfg = policy.config
    assert isinstance(cfg, SmolVLAConfig)
    ds_features = dataset_features_from_policy(cfg)
    expected_state_dim = ds_features[OBS_STATE]["shape"][0]
    logger.info("SmolVLA policy loaded (state_dim=%d)", expected_state_dim)

    _infer_calls: dict[str, Any] = {"n": 0, "quiet": False}

    def predict(inputs: dict[str, Any]) -> torch.Tensor:
        """Run policy inference on prepared inputs.

        Args:
            inputs: Dict with keys:
                - "images": {training_camera_name: np.ndarray HWC uint8}
                - "state": np.ndarray of joint positions
                - "instruction": str

        Returns:
            Raw action chunk tensor [1, chunk_size, action_dim] (postprocessed/denormalized)
        """
        _infer_calls["n"] += 1
        call_id = _infer_calls["n"]
        quiet = bool(_infer_calls.get("quiet"))
        t_all = time.perf_counter()

        images = inputs["images"]
        state = inputs["state"]
        instruction = inputs["instruction"]

        t0 = time.perf_counter()
        state = _align_state(state, expected_state_dim)
        t_align = time.perf_counter() - t0

        t0 = time.perf_counter()
        obs = raw_observation_from_tensors(cfg, ds_features, images, state)
        t_obs = time.perf_counter() - t0
        if not quiet:
            logger.info(
                "Inference #%d: raw_observation_from_tensors done in %.4fs",
                call_id,
                t_obs,
            )

        t0 = time.perf_counter()
        frame = build_inference_frame(
            obs, device, ds_features, task=instruction, robot_type=robot_type
        )
        t_frame = time.perf_counter() - t0
        if not quiet:
            logger.info(
                "Inference #%d: build_inference_frame done in %.4fs",
                call_id,
                t_frame,
            )

        t0 = time.perf_counter()
        frame = preprocess(frame)
        _cuda_sync(device)
        t_pre = time.perf_counter() - t0
        if not quiet:
            logger.info(
                "Inference #%d: preprocess done in %.4fs (waiting on GPU if slow)",
                call_id,
                t_pre,
            )

        t0 = time.perf_counter()
        policy.reset()
        t_reset = time.perf_counter() - t0

        if not quiet:
            logger.info(
                "Inference #%d: starting _get_action_chunk (first call can be slow: CUDA graphs, kernels)...",
                call_id,
            )
        t0 = time.perf_counter()
        with torch.no_grad():
            action_chunk = policy._get_action_chunk(frame, noise=None)
        _cuda_sync(device)
        t_forward = time.perf_counter() - t0
        if not quiet:
            logger.info(
                "Inference #%d: _get_action_chunk done in %.4fs",
                call_id,
                t_forward,
            )

        t0 = time.perf_counter()
        action_chunk = postprocess(action_chunk)
        _cuda_sync(device)
        t_post = time.perf_counter() - t0

        total = time.perf_counter() - t_all
        if not quiet:
            logger.info(
                "Inference #%d timings (s): align=%.4f obs=%.4f build_frame=%.4f "
                "preprocess=%.4f reset=%.4f forward=%.4f postprocess=%.4f total=%.4f | device=%s",
                call_id,
                t_align,
                t_obs,
                t_frame,
                t_pre,
                t_reset,
                t_forward,
                t_post,
                total,
                device,
            )
            logger.info(
                "Predicted action chunk shape %s for instruction: %s",
                action_chunk.shape,
                instruction,
            )
        return action_chunk

    skip_warmup = os.environ.get("SMOLVLA_SKIP_GPU_WARMUP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if device.type == "cuda" and not skip_warmup:
        w0 = time.perf_counter()
        logger.info(
            "GPU warmup: one forward before control loop (first CUDA pass can take ~1–2 min; "
            "disable with SMOLVLA_SKIP_GPU_WARMUP=1)..."
        )
        _infer_calls["quiet"] = True
        try:
            predict(_warmup_inputs_for_policy(cfg, expected_state_dim))
        except Exception as e:
            logger.warning(
                "GPU warmup failed (non-fatal; first live cycle may be slow): %s", e
            )
        finally:
            _infer_calls["quiet"] = False
            _infer_calls["n"] = 0
        logger.info("GPU warmup finished in %.2fs", time.perf_counter() - w0)
    elif device.type == "cuda" and skip_warmup:
        logger.info(
            "SMOLVLA_SKIP_GPU_WARMUP set: skipping GPU warmup (first inference may be very slow)"
        )

    # Expose the camera short-names the policy was trained with so callers
    # (e.g. CwProcessorPreview) can build the images dict with the correct keys
    # without needing a separate resolver pass.
    predict.training_camera_names = [  # type: ignore[attr-defined]
        key.split("observation.images.")[-1]
        for key, pf in cfg.input_features.items()
        if pf.type == FeatureType.VISUAL and "empty_camera" not in key
    ]
    predict.expected_state_dim = expected_state_dim  # type: ignore[attr-defined]

    # action_out_dim = int(ds_features[ACTION]["shape"][0])
    # predict.action_dim = action_out_dim  # type: ignore[attr-defined]
    # _chunk_raw = getattr(cfg, "chunk_size", None)
    # if _chunk_raw is None:
    #     _chunk_raw = getattr(cfg, "n_action_steps", None)
    # try:
    #     chunk_size_int = int(_chunk_raw) if _chunk_raw is not None else 0
    # except (TypeError, ValueError):
    #     chunk_size_int = 0
    # predict.chunk_size = chunk_size_int  # type: ignore[attr-defined]

    # # CHW training layout from policy; preview clients send HWC uint8 (see cw_processor_preview).
    # predict.image_input_specs = {  # type: ignore[attr-defined]
    #     key.split("observation.images.")[-1]: {
    #         "layout": "CHW",
    #         "channels": int(pf.shape[0]),
    #         "height": int(pf.shape[1]),
    #         "width": int(pf.shape[2]),
    #     }
    #     for key, pf in cfg.input_features.items()
    #     if pf.type == FeatureType.VISUAL and "empty_camera" not in key
    # }

    return predict


class ColorFormatter(logging.Formatter):
    """Colorized logging formatter for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        level_short = {
            "DEBUG": "DBG",
            "INFO": "INF",
            "WARNING": "WRN",
            "ERROR": "ERR",
            "CRITICAL": "CRT",
        }.get(record.levelname, record.levelname[:3])

        # Simplified format: [LEVEL] message
        msg = record.getMessage()
        return f"{color}{self.BOLD}[{level_short}]{self.RESET} {msg}"


def setup_logging():
    """Configure colorized logging."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter())

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    _apply_hf_offline_from_env()
    setup_logging()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(
            "Usage: python deploy.py '<json_payload>' or /path/to/params.json",
            file=sys.stderr,
        )
        return 1

    try:
        request = parse_request_payload(load_json_argument(arguments[0]))

        checkpoint = os.environ.get("SMOLVLA_CHECKPOINT", "")
        if not checkpoint and request.weights_url:
            checkpoint = download_weights(request.weights_url)
        if not checkpoint:
            raise ValueError(
                "No checkpoint available. Set SMOLVLA_CHECKPOINT env var "
                "or provide 'weights_url' in the request payload."
            )

        robot_type = request.calibration_robot_type or ""

        # Build predict_fn (model loading happens here)
        predict_fn = build_predict_fn(
            checkpoint,
            robot_type=robot_type,
            base_model=os.environ.get("SMOLVLA_BASE_MODEL") or request.policy_repo_id,
        )

        # CwProcessor handles all Cyberwave I/O and data transformations
        # It builds the resolver internally based on model_slug
        processor = CwProcessor(
            request,
            model_slug=MODEL_SLUG,
            checkpoint=checkpoint,
            predict_fn=predict_fn,
        )
        processor.setup()
        try:
            result = processor.run()
        finally:
            processor.disconnect()

        # Add resolver info to result
        result["model_slug"] = MODEL_SLUG
        result["training_cameras"] = processor.resolver.training_camera_names
        result["camera_mapping"] = processor.camera_mapping

        print(json.dumps(result))
        return 0

    except Exception as exc:
        logger.exception("SmolVLA inference failed")
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
