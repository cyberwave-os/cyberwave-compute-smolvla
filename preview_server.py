#!/usr/bin/env python3
"""SmolVLA hot inference server for playground preview runs.

Loads the model ONCE at startup, then serves inference requests over a local
HTTP API.  preview.py connects to this server instead of cold-loading the model
on every workload — eliminating the 30-60 s model load time for subsequent
playground requests.

Architecture:
                                         GPU node
   ┌───────────────────────────────────────────────────────┐
   │                                                       │
   │  preview_server.py (long-running, started once)       │
   │  ┌─────────────────────────────────────────────────┐  │
   │  │  FastAPI  :8765  /health /config /predict              │  │
   │  │  model loaded in memory (predict_fn in closure) │  │
   │  └─────────────────────────────────────────────────┘  │
   │              ↑  HTTP POST                              │
   │  preview.py  │  (one per workload, thin client)        │
   │  └──── POST  /predict  → result JSON ────────────────► │
   │                                                       │
   └───────────────────────────────────────────────────────┘

Startup:
    The cloud-node operator starts this server once (e.g. via the
    server_start command in cyberwave.yml) before any preview workloads
    are dispatched:

        python preview_server.py

    preview.py checks whether the server is up (GET /health) before
    each workload.  If it is, it POSTs to /predict.  If not, it falls
    back to cold-loading the model inline (same behaviour as before).

Endpoints:
    GET  /health   → {"status": "ok"|"loading", "ready": bool, "checkpoint", ...}
    GET  /config   → input/output JSON shapes and field documentation (usable while loading)
    POST /predict  → VLAPreviewParams JSON → result JSON (same as preview.py)

Environment variables:
    SMOLVLA_CHECKPOINT          Local checkpoint directory or HuggingFace repo id (required).
    SMOLVLA_BASE_MODEL          Optional base model for PEFT checkpoints.
    PREVIEW_SERVER_PORT         Port to bind (default: 8765).
    PREVIEW_SERVER_HOST         Host to bind (default: 127.0.0.1).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Stable identifier — ``preview.py`` uses this to ignore unrelated HTTP services
# bound to the same port (which would otherwise look like ``ready: false`` and
# block spawning for the full load timeout).
PREVIEW_SERVER_SERVICE_ID = "smolvla-preview-server"

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

_CHECKPOINT = os.environ.get("SMOLVLA_CHECKPOINT", "").strip()
_BASE_MODEL = os.environ.get("SMOLVLA_BASE_MODEL", "").strip() or None
_PORT = int(os.environ.get("PREVIEW_SERVER_PORT", "8765"))
_HOST = os.environ.get("PREVIEW_SERVER_HOST", "127.0.0.1")


# ---------------------------------------------------------------------------
# Module-level model state (loaded once at startup)
# ---------------------------------------------------------------------------

_predict_fn: Any = None
_training_camera_names: list[str] = []
_state_dim: int = 0
_action_dim: int = 0
_chunk_size: int = 0
_image_input_specs: dict[str, Any] = {}
_ready: bool = False
_load_error: str | None = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SmolVLA Preview Server", version="1.0.0")


class PredictRequest(BaseModel):
    """Thin wrapper — we forward the full VLAPreviewParams dict as-is."""

    payload: dict[str, Any]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "service_id": PREVIEW_SERVER_SERVICE_ID,
        "status": "ok" if _ready else "loading",
        "ready": _ready,
        "checkpoint": _CHECKPOINT,
        "port": _PORT,
        "training_camera_names": _training_camera_names,
        "state_dim": _state_dim,
        "action_dim": _action_dim,
        "chunk_size": _chunk_size,
        "load_error": _load_error,
    }


@app.get("/config")
def config() -> dict[str, Any]:
    """Describe request/response JSON contracts and model tensor layouts."""
    return _build_public_config()


def _build_public_config() -> dict[str, Any]:
    """Static + runtime I/O documentation for operators and API clients."""
    return {
        "service": {
            "name": "smolvla-preview",
            "title": "SmolVLA Preview Server",
            "version": app.version,
        },
        "model": {
            "checkpoint": _CHECKPOINT,
            "base_model": _BASE_MODEL,
            "ready": _ready,
            "state_dim": _state_dim,
            "action_dim": _action_dim,
            "chunk_size": _chunk_size,
            "training_camera_names": list(_training_camera_names),
            "image_input_specs": dict(_image_input_specs),
        },
        "inputs": {
            "content_type": "application/json",
            "post_body": {
                "description": "POST /predict with { \"payload\": <VLAPreviewParams dict> }.",
                "wrapper_key": "payload",
                "vlapreviewparams_fields": {
                    "mlmodel_uuid": "string (required for playground routing)",
                    "weights_url": "string | null",
                    "family": "string (e.g. smolvla)",
                    "prompt": "natural-language instruction",
                    "image_base64": "base64-encoded RGB image bytes (no data-URL prefix required)",
                    "checkpoint": "optional HF id / path hint",
                    "workload_uuid": "optional cloud workload id",
                    "cyberwave_token": "optional API token for result upload",
                    "robot": {
                        "dof": "int",
                        "joint_names": "list[str]",
                        "joint_limits": "dict",
                        "output_mode": "joint_action | cartesian_ee_delta (forwarded to result)",
                        "registry_id": "string | null",
                        "gripper_max_width_m": "float | null",
                    },
                },
            },
            "internal_predict_fn": {
                "description": "CwProcessorPreview builds this from the payload; exposed for debugging.",
                "images": {
                    "type": "dict[str, ndarray]",
                    "keys": "training_camera_names (single playground image is replicated per key)",
                    "dtype": "uint8",
                    "layout": "HWC RGB",
                },
                "state": {
                    "type": "ndarray float32",
                    "shape": "(state_dim,) — preview uses zeros; live deploy uses proprio",
                },
                "instruction": "string",
            },
        },
        "outputs": {
            "predict_response": {
                "description": "JSON from CwProcessorPreview.run() plus inference_ms on hot server.",
                "top_level_fields": [
                    "status",
                    "family",
                    "prompt",
                    "mlmodel_uuid",
                    "workload_uuid",
                    "robot",
                    "result",
                    "inference_ms",
                ],
                "result_object": {
                    "output_format": "joint_action_chunk | ee_delta_chunk",
                    "output": "list[list[float]] — one row per timestep, length num_steps",
                    "num_steps": "int (typically matches policy chunk_size after squeeze)",
                    "action_dim": "int (columns per row after truncation/padding)",
                    "output_mode": "string (from robot.output_mode)",
                    "joint_names": "optional list[str] trimmed to action_dim",
                },
            },
            "raw_tensor": {
                "description": "Internal predict_fn return before formatting (for reference).",
                "shape": "[1, chunk_size, action_dim] or [chunk_size, action_dim]",
            },
        },
    }
@app.post("/predict")
def predict(request: PredictRequest) -> JSONResponse:
    if not _ready or _predict_fn is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    from cw_processor_preview import CwProcessorPreview

    processor = CwProcessorPreview(
        request.payload,
        _predict_fn,
        training_camera_names=_training_camera_names,
        state_dim=_state_dim,
    )

    t0 = time.perf_counter()
    try:
        result = processor.run()
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Inference done in %.1f ms", elapsed_ms)
    result["inference_ms"] = round(elapsed_ms, 1)

    # Coerce any numpy/torch values to plain Python before JSON serialisation.
    def _default(x: Any) -> Any:
        if hasattr(x, "tolist"):
            return x.tolist()
        return str(x)

    return JSONResponse(content=json_safe(result))


def json_safe(obj: Any) -> Any:
    """Recursively coerce numpy/torch scalars and arrays to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _checkpoint_must_exist_on_disk(checkpoint: str) -> bool:
    """Mirror ``preview.py`` cold-start: HF hub ids like ``org/model`` skip ``exists()``."""
    if not checkpoint.strip():
        return False
    path = Path(checkpoint)
    if "/" in checkpoint and not path.is_absolute():
        return False
    return True


def load_model() -> None:
    """Load the SmolVLA model into module-level globals. Called at startup."""
    global _predict_fn, _training_camera_names, _state_dim, _action_dim, _chunk_size
    global _image_input_specs, _ready, _load_error

    _load_error = None

    checkpoint = _CHECKPOINT
    if not checkpoint:
        _load_error = (
            "SMOLVLA_CHECKPOINT is not set. "
            "Export a local path or a HuggingFace repo id (e.g. lerobot/smolvla_base)."
        )
        logger.error(
            "SMOLVLA_CHECKPOINT env var not set. "
            "Set it to the path of your SmolVLA checkpoint or a HF hub id before starting the server."
        )
        return

    if _checkpoint_must_exist_on_disk(checkpoint) and not Path(checkpoint).exists():
        _load_error = f"Checkpoint path does not exist: {checkpoint!r}"
        logger.error(
            "Checkpoint not found at %r. "
            "Set SMOLVLA_CHECKPOINT to a valid local path.",
            checkpoint,
        )
        return

    logger.info("Loading SmolVLA model from %s …", checkpoint)
    t0 = time.perf_counter()

    try:
        # Load camera names and state dim from resolver before loading the model.
        try:
            from smolvla_resolver import SmolVLAResolver

            resolver = SmolVLAResolver(checkpoint)
            _training_camera_names = resolver.training_camera_names
            _state_dim = resolver.get_expected_state_dim()
            _action_dim = resolver.get_expected_action_dim()
            _chunk_size = resolver.get_chunk_size()
            logger.info(
                "Resolver: cameras=%s state_dim=%d action_dim=%d chunk_size=%d",
                _training_camera_names,
                _state_dim,
                _action_dim,
                _chunk_size,
            )
        except Exception as e:
            logger.warning("Could not load resolver metadata: %s — using defaults", e)

        from deploy import build_predict_fn

        _predict_fn = build_predict_fn(
            checkpoint=checkpoint,
            robot_type="",
            base_model=_BASE_MODEL,
        )

        # Authoritative camera keys and state dim come from the loaded policy
        # (see deploy.build_predict_fn). The resolver often has no metadata for
        # HuggingFace hub ids — without this sync, CwProcessorPreview falls back
        # to {"primary": ...} while raw_observation_from_tensors expects e.g. camera1.
        policy_cams = list(getattr(_predict_fn, "training_camera_names", []) or [])
        if policy_cams:
            _training_camera_names = policy_cams
            logger.info("Training camera keys from policy: %s", policy_cams)
        policy_state_dim = int(getattr(_predict_fn, "expected_state_dim", 0) or 0)
        if policy_state_dim > 0:
            _state_dim = policy_state_dim

        ad = int(getattr(_predict_fn, "action_dim", 0) or 0)
        if ad > 0:
            _action_dim = ad
        cs = int(getattr(_predict_fn, "chunk_size", 0) or 0)
        if cs > 0:
            _chunk_size = cs
        specs = getattr(_predict_fn, "image_input_specs", None)
        if isinstance(specs, dict) and specs:
            _image_input_specs = dict(specs)

        elapsed = time.perf_counter() - t0
        logger.info(
            "SmolVLA model loaded in %.1f s — server ready on %s:%d", elapsed, _HOST, _PORT
        )
        _ready = True
    except Exception as exc:
        _load_error = str(exc) or type(exc).__name__
        logger.exception("SmolVLA model load failed")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    # Load the model in a background thread so uvicorn can accept connections
    # immediately.  GET /health then returns ``ready: false`` while loading,
    # which lets preview.py poll instead of spawning duplicate servers.
    threading.Thread(
        target=load_model,
        name="smolvla-preview-load",
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    uvicorn.run(
        "preview_server:app",
        host=_HOST,
        port=_PORT,
        log_level="info",
        # Single worker — the SmolVLA model is not thread-safe; requests are
        # serialised by uvicorn's single-threaded event loop.
        workers=1,
    )


if __name__ == "__main__":
    # Ensure the smolvla directory is on sys.path when run directly.
    _here = str(Path(__file__).parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    main()
