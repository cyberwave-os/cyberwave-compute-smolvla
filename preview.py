#!/usr/bin/env python3
"""SmolVLA preview entrypoint — runs as the `inference` command.

This script is invoked by the Cloud Node for every inference workload.  It
detects whether the payload is a **preview** request (from the playground) or a
**live** inference request (from a robot twin), and routes accordingly:

  Live inference  → delegates to deploy.py / CwProcessor (MQTT loop).
  Preview         → routes through the hot inference server for fast, stateless
                    one-shot inference.

Preview server lifecycle (self-managed):
  The hot inference server (preview_server.py) loads the model once and stays
  alive between workloads to eliminate cold-start latency.  This script manages
  the server transparently:

    1. On every preview request, check GET /health (fast, 2 s timeout).
    2. If the server IS running and ready, POST to /predict and return.
    3. If NOT running, spawn preview_server.py as a detached background daemon
       and poll /health until the model is loaded (up to PREVIEW_SERVER_LOAD_TIMEOUT,
       default 300 s).
    4. On timeout, fall back to cold-start inference inline.

Payload detection:
  A payload is treated as a preview request when it contains the key
  "mlmodel_uuid" AND does NOT contain "robot_twin_uuid".  Live inference
  payloads always contain "robot_twin_uuid".

Usage (Cloud Node invokes this automatically):
    python preview.py /path/to/params.json
    python preview.py '<json_payload>'

Environment variables:
    SMOLVLA_CHECKPOINT          Checkpoint: local directory or HuggingFace repo id
                                (e.g. lerobot/smolvla_base). If unset, preview uses
                                payload["checkpoint"] or the same default HF id.
    PREVIEW_SERVER_PORT         Port for the hot inference server (default: 8765).
    PREVIEW_SERVER_HOST         Host for the hot inference server (default: 127.0.0.1).
    PREVIEW_SERVER_LOAD_TIMEOUT Seconds to wait for the model to load after HTTP is up
                                (default: 300).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl  # Unix: serialize daemon spawn across concurrent preview.py runs
except ImportError:
    fcntl = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_SERVER_HOST = os.environ.get("PREVIEW_SERVER_HOST", "127.0.0.1")
_SERVER_PORT = int(os.environ.get("PREVIEW_SERVER_PORT", "8765"))
_SERVER_BASE_URL = f"http://{_SERVER_HOST}:{_SERVER_PORT}"
_SERVER_CONNECT_TIMEOUT = 2.0   # seconds — health-check connect timeout
_SERVER_READ_TIMEOUT = 120.0    # seconds — inference request timeout
# Max time to wait for the model to finish loading after the HTTP server is up.
_SERVER_MODEL_LOAD_TIMEOUT = float(
    os.environ.get("PREVIEW_SERVER_LOAD_TIMEOUT", "300")
)
_STATE_DIR = Path(os.environ.get("HOME", "/tmp")) / ".cyberwave"
_SERVER_STATE_JSON = _STATE_DIR / "smolvla_preview_server_state.json"
_SPAWN_LOCK_PATH = _STATE_DIR / "smolvla_preview_server_spawn.lock"
_SMOLVLA_DIR = Path(__file__).resolve().parent

# Must match ``PREVIEW_SERVER_SERVICE_ID`` in ``preview_server.py`` (avoid import cycle).
_PREVIEW_SERVER_SERVICE_ID = "smolvla-preview-server"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("PIL", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def load_json_argument(argument: str) -> dict[str, Any]:
    """Load the payload from a file path or inline JSON string."""
    if os.path.isfile(argument):
        with open(argument, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(argument)


def _is_preview_payload(payload: dict[str, Any]) -> bool:
    """Return True if this is a playground preview payload.

    Preview payloads contain "mlmodel_uuid" but NOT "robot_twin_uuid".
    Live inference payloads always contain "robot_twin_uuid".
    """
    return bool(payload.get("mlmodel_uuid")) and not payload.get("robot_twin_uuid")


# ---------------------------------------------------------------------------
# Hot-server helpers
# ---------------------------------------------------------------------------


def _urlopen_hot_server(req: Any, *, timeout: float) -> Any:
    """Open an HTTP connection to the local preview server **without** HTTP(S)_PROXY.

    Corporate proxies often break or stall ``http://127.0.0.1/...``, which makes
    health polling look like a full ``PREVIEW_SERVER_LOAD_TIMEOUT`` wait.
    """
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


def _http_get_json(url: str, timeout: float) -> dict[str, Any] | None:
    try:
        import urllib.request

        req = urllib.request.Request(url)
        with _urlopen_hot_server(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _is_our_preview_health(data: dict[str, Any]) -> bool:
    """True if JSON came from *our* FastAPI ``/health`` (not another process on the port)."""
    if not isinstance(data, dict):
        return False
    if data.get("service_id") == _PREVIEW_SERVER_SERVICE_ID:
        return True
    # Legacy preview_server builds before ``service_id`` was added.
    return (
        data.get("status") in ("ok", "loading")
        and isinstance(data.get("ready"), bool)
        and "checkpoint" in data
        and isinstance(data.get("training_camera_names"), list)
    )


def _fetch_health(timeout: float | None = None) -> dict[str, Any] | None:
    """GET /health from our preview server, or None if unreachable / wrong service."""
    raw = _http_get_json(
        f"{_SERVER_BASE_URL}/health",
        _SERVER_CONNECT_TIMEOUT if timeout is None else timeout,
    )
    if raw is None:
        return None
    if not _is_our_preview_health(raw):
        keys = sorted(raw.keys()) if isinstance(raw, dict) else []
        logger.warning(
            "Ignoring /health on %s — not our SmolVLA preview server "
            "(expected service_id=%r or legacy shape; got keys=%s). "
            "Another process may be bound to this port.",
            _SERVER_BASE_URL,
            _PREVIEW_SERVER_SERVICE_ID,
            keys[:20],
        )
        return None
    return raw


def _write_server_state(pid: int) -> None:
    """Record daemon metadata for operators and concurrent preview.py processes."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "host": _SERVER_HOST,
        "port": _SERVER_PORT,
        "base_url": _SERVER_BASE_URL,
    }
    with open(_SERVER_STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _read_state_pid() -> int | None:
    """Return daemon PID from ``smolvla_preview_server_state.json`` if readable."""
    try:
        with open(_SERVER_STATE_JSON, encoding="utf-8") as sf:
            data = json.load(sf)
        pid_raw = data.get("pid")
        pid = int(pid_raw) if pid_raw is not None else 0
        return pid if pid > 0 else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _daemon_pid_claimed_alive() -> bool:
    """True if state file lists a PID that still exists (daemon likely starting or up)."""
    pid = _read_state_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def _spawn_lock() -> Iterator[None]:
    """Exclusive lock so only one preview.py spawns preview_server at a time."""
    if fcntl is None:
        yield
        return
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _SPAWN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SPAWN_LOCK_PATH, "a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _start_server_daemon() -> int:
    """Spawn preview_server.py as a detached background daemon.

    Uses the same Python interpreter as the current process.  Stdout/stderr
    are appended to a log file so the operator can inspect startup progress.
    The process is detached (start_new_session=True) so it survives after
    this workload subprocess exits.

    Returns the child PID.
    """
    server_script = _SMOLVLA_DIR / "preview_server.py"
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _STATE_DIR / "smolvla_preview_server.log"

    logger.info("Spawning preview_server.py daemon (log: %s) …", log_file)
    with open(log_file, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(server_script)],
            cwd=str(_SMOLVLA_DIR),
            stdin=subprocess.DEVNULL,
            env=os.environ.copy(),
            stdout=lf,
            stderr=lf,
            start_new_session=True,
        )
    pid = int(proc.pid)
    _write_server_state(pid)
    logger.info(
        "preview_server.py daemon started (pid=%s, state=%s)",
        pid,
        _SERVER_STATE_JSON,
    )
    return pid


def _spawn_daemon_if_absent() -> None:
    """If nothing is listening on /health, start exactly one daemon (flock)."""

    def _maybe_spawn() -> None:
        h = _fetch_health()
        if h is not None:
            logger.info(
                "Preview HTTP server already reachable — not spawning "
                "(another worker likely started it)."
            )
            return
        # HTTP not up yet, but a daemon we spawned may still be booting (uvicorn
        # was blocking on model load in older preview_server versions, or GPU init).
        if _daemon_pid_claimed_alive():
            logger.info(
                "Preview daemon pid %s still alive — waiting for HTTP rather than "
                "spawning a second server.",
                _read_state_pid(),
            )
            return
        _start_server_daemon()

    with _spawn_lock():
        _maybe_spawn()


def _wait_until_model_ready(deadline_monotonic: float) -> bool:
    """Poll GET /health until ``ready`` or deadline. Returns True if ready."""
    fast_poll_until = time.monotonic() + 45.0
    saw_http = False
    wait_started = time.monotonic()
    last_stall_log = 0.0
    while time.monotonic() < deadline_monotonic:
        h = _fetch_health()
        if h:
            err = h.get("load_error")
            if isinstance(err, str) and err.strip():
                logger.warning(
                    "Preview server will not become ready (load_error=%s) — cold-start fallback.",
                    err.strip()[:300],
                )
                return False
            if h.get("ready"):
                return True
            if not saw_http:
                saw_http = True
                logger.info(
                    "Preview server HTTP is up; waiting for model (ready=false, timeout %.0f s) …",
                    _SERVER_MODEL_LOAD_TIMEOUT,
                )
        else:
            elapsed = time.monotonic() - wait_started
            if elapsed >= 15.0 and (time.monotonic() - last_stall_log) >= 30.0:
                last_stall_log = time.monotonic()
                log_path = _STATE_DIR / "smolvla_preview_server.log"
                logger.warning(
                    "Still cannot reach %s/health after %.0f s — "
                    "check whether preview_server is listening and see %s",
                    _SERVER_BASE_URL,
                    elapsed,
                    log_path,
                )
        # Tighter polling right after spawn so a 10–20 s model load is noticed quickly.
        interval = 0.4 if time.monotonic() < fast_poll_until else 1.0
        time.sleep(min(interval, max(0.0, deadline_monotonic - time.monotonic())))
    h = _fetch_health()
    if h:
        err = h.get("load_error")
        if isinstance(err, str) and err.strip():
            logger.warning(
                "Preview server load_error at deadline — cold-start fallback: %s",
                err.strip()[:300],
            )
            return False
    ok = bool(h and h.get("ready"))
    if not ok:
        logger.warning(
            "Hot inference server did not become ready within %.0f s — cold-start fallback.",
            _SERVER_MODEL_LOAD_TIMEOUT,
        )
    return ok


def _ensure_server_ready() -> bool:
    """Ensure the hot inference server is running and the model is loaded.

    - If /health returns ``ready: true``, return immediately.
    - If /health responds with ``ready: false``, the FastAPI process is already
      up and the model is still loading — **do not spawn another server**;
      poll until ``ready`` or ``PREVIEW_SERVER_LOAD_TIMEOUT``.
    - If the TCP connection fails, take the spawn lock, spawn the daemon if
      still down, then poll until ready or timeout.

    Returns True if ``/predict`` can be used; False if timed out (caller may
    cold-start inline).
    """
    h = _fetch_health()
    if h and h.get("ready"):
        logger.info("Hot inference server already ready.")
        return True
    if h:
        err = h.get("load_error")
        if isinstance(err, str) and err.strip():
            logger.warning(
                "Hot inference server reported load_error — skipping wait: %s",
                err.strip()[:300],
            )
            return False

    deadline = time.monotonic() + _SERVER_MODEL_LOAD_TIMEOUT

    if h is not None:
        # IMPORTANT: ``ready: false`` means uvicorn is up and the model is
        # still loading.  Spawning another preview_server here binds the same
        # port and produces duplicate daemons — see issue reports from the
        # playground preview path.
        logger.info(
            "Hot inference server HTTP up but model still loading — waiting "
            "(timeout %.0f s) …",
            _SERVER_MODEL_LOAD_TIMEOUT,
        )
        return _wait_until_model_ready(deadline)

    logger.info(
        "Hot inference server not reachable — spawning daemon (timeout %.0f s) …",
        _SERVER_MODEL_LOAD_TIMEOUT,
    )
    _spawn_daemon_if_absent()
    return _wait_until_model_ready(deadline)


def _run_via_hot_server(payload: dict[str, Any]) -> dict[str, Any] | None:
    """POST the payload to the hot server. Returns result dict or None on failure."""
    try:
        import urllib.request

        body = json.dumps({"payload": payload}).encode()
        req = urllib.request.Request(
            f"{_SERVER_BASE_URL}/predict",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urlopen_hot_server(req, timeout=_SERVER_READ_TIMEOUT) as resp:
            result = json.loads(resp.read().decode())
            logger.info("Hot-server inference complete (%.0f ms)", result.get("inference_ms", 0))
            return result
    except Exception as exc:
        logger.warning("Hot-server /predict failed (%s) — falling back to cold start", exc)
        return None


# ---------------------------------------------------------------------------
# Preview cold-start path
# ---------------------------------------------------------------------------


_PREVIEW_BASE_MODEL = "lerobot/smolvla_base"


def _resolve_checkpoint(payload: dict[str, Any]) -> str:
    """Return the checkpoint for a preview run.

    Preview nodes load weights from a fixed path baked into the image / VM
    (``SMOLVLA_CHECKPOINT``). The playground payload may carry an optional HF
    hub id hint in ``checkpoint`` when the node was not pre-configured.

    Priority:
      1. ``SMOLVLA_CHECKPOINT`` env (production / Docker default)
      2. ``payload["checkpoint"]`` — HuggingFace hub id hint from model metadata
      3. default base model id
    """
    checkpoint = os.environ.get("SMOLVLA_CHECKPOINT", "").strip()
    if checkpoint:
        return checkpoint
    hint = payload.get("checkpoint")
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    return _PREVIEW_BASE_MODEL


def _run_preview_cold_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Cold-start preview: load the model inline and run one forward pass."""
    checkpoint = _resolve_checkpoint(payload)

    # Only check the filesystem if the checkpoint looks like a local path.
    # HuggingFace Hub IDs (e.g. "lerobot/smolvla_base") are fetched on demand
    # by the model loading code and do not need to exist on disk first.
    if "/" not in checkpoint or Path(checkpoint).is_absolute():
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint!r}. "
                "Set SMOLVLA_CHECKPOINT env var to a valid path."
            )

    from deploy import build_predict_fn

    predict_fn = build_predict_fn(
        checkpoint=checkpoint,
        robot_type="",
        base_model=_PREVIEW_BASE_MODEL,
    )

    # build_predict_fn attaches the authoritative training camera names and
    # expected state dim directly to the predict function after loading the
    # policy config — use those instead of the resolver (which only works for
    # local checkpoint paths, not HuggingFace Hub IDs like lerobot/smolvla_base).
    training_camera_names: list[str] = getattr(predict_fn, "training_camera_names", [])
    state_dim: int = int(getattr(predict_fn, "expected_state_dim", 0))

    if training_camera_names:
        logger.info("Camera names from policy config: %s", training_camera_names)
    else:
        logger.warning("No camera names resolved from policy config — falling back to positional slot")

    from cw_processor_preview import CwProcessorPreview

    processor = CwProcessorPreview(
        payload,
        predict_fn,
        training_camera_names=training_camera_names,
        state_dim=state_dim,
    )
    return processor.run()


# ---------------------------------------------------------------------------
# Live inference path
# ---------------------------------------------------------------------------


def _run_live_inference(payload: dict[str, Any]) -> int:
    """Delegate to deploy.py main() for live robot inference via MQTT."""
    import importlib.util as _ilu
    import tempfile

    deploy = Path(__file__).parent / "deploy.py"
    spec = _ilu.spec_from_file_location("_smolvla_deploy_top", deploy)
    mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(payload, tf)
        tmp_path = tf.name

    original_argv = sys.argv[:]
    sys.argv = [str(deploy), tmp_path]
    try:
        return mod.main()  # type: ignore[no-any-return]
    finally:
        sys.argv = original_argv
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Backend result upload
# ---------------------------------------------------------------------------


def _complete_workload_with_result(
    workload_uuid: str,
    result: dict[str, Any],
    api_base_url: str,
    api_token: str,
) -> bool:
    """POST result data to the backend's complete endpoint.

    Calls ``POST /api/v1/cloud-node-workloads/{workload_uuid}/complete`` with
    an inline ``result`` payload so the backend persists it in the workload's
    ``results`` field before this process exits.

    Returns True on success, False on any error (failures are non-fatal —
    the result is also printed to stdout for the cloud-node log stream).
    """
    import urllib.error
    import urllib.request

    url = f"{api_base_url.rstrip('/')}/api/v1/cloud-node-workloads/{workload_uuid}/complete"
    body = json.dumps({"result": result, "success": True}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {api_token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            logger.info(
                "Uploaded result to backend for workload %s (HTTP %d)",
                workload_uuid,
                resp.status,
            )
            return True
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace") if exc.fp else ""
        logger.warning(
            "Failed to upload result to backend (HTTP %d): %s",
            exc.code,
            body_text[:200],
        )
    except Exception as exc:
        logger.warning("Failed to upload result to backend: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    arguments = list(sys.argv[1:] if argv is None else argv)

    if not arguments:
        print("Usage: python preview.py '<json_payload>' or /path/to/params.json", file=sys.stderr)
        return 1

    try:
        payload = load_json_argument(arguments[0])
    except Exception as e:
        logger.exception("Failed to parse payload")
        print(json.dumps({"status": "error", "error": f"Invalid payload: {e}"}))
        return 1

    # ------------------------------------------------------------------
    # Route: live inference vs. preview
    # ------------------------------------------------------------------
    if not _is_preview_payload(payload):
        logger.info("Routing to live inference (deploy.py)")
        try:
            rc = _run_live_inference(payload)
        except Exception as e:
            logger.exception("Live inference failed")
            print(json.dumps({"status": "error", "error": str(e)}))
            return 1
        return rc

    # ------------------------------------------------------------------
    # Preview path: ensure hot server, fall back to cold start
    # ------------------------------------------------------------------
    logger.info("Routing to preview inference")

    # Hot server and cold-start must agree on weights (fixed node checkpoint or HF hint).
    if not os.environ.get("SMOLVLA_CHECKPOINT", "").strip():
        ckpt = _resolve_checkpoint(payload)
        os.environ["SMOLVLA_CHECKPOINT"] = ckpt
        logger.info(
            "SMOLVLA_CHECKPOINT was unset — resolved preview checkpoint %r "
            "for hot server + cold start.",
            ckpt,
        )

    server_ready = _ensure_server_ready()
    if server_ready:
        result = _run_via_hot_server(payload)
        if result is not None:
            _maybe_upload_result(payload, result)
            print(json.dumps(result))
            return 0

    logger.info("Cold-start preview inference …")
    try:
        result = _run_preview_cold_start(payload)
    except Exception as e:
        logger.exception("Preview inference failed")
        print(json.dumps({"status": "error", "error": str(e)}))
        return 1

    _maybe_upload_result(payload, result)
    print(json.dumps(result, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    return 0


def _maybe_upload_result(payload: dict[str, Any], result: dict[str, Any]) -> None:
    """Upload result to backend if we have the necessary credentials in the payload."""
    workload_uuid = payload.get("workload_uuid") or (result or {}).get("workload_uuid")
    api_token = payload.get("cyberwave_token") or os.environ.get("CYBERWAVE_API_KEY", "")
    api_base_url = (
        os.environ.get("CYBERWAVE_API_URL")
        or os.environ.get("CYBERWAVE_BASE_URL")
        or "http://localhost:8000"
    )

    if not workload_uuid:
        logger.warning("No workload_uuid in payload — skipping backend result upload")
        return
    if not api_token:
        logger.warning("No API token available — skipping backend result upload")
        return
    if not api_base_url:
        logger.warning("CYBERWAVE_BASE_URL not set — skipping backend result upload")
        return

    _complete_workload_with_result(
        workload_uuid=workload_uuid,
        result=result,
        api_base_url=api_base_url,
        api_token=api_token,
    )


if __name__ == "__main__":
    raise SystemExit(main())
