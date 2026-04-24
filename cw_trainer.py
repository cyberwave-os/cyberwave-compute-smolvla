"""Cyberwave training orchestrator for VLA models.

Handles all Cyberwave-specific training concerns:
- Dataset download from Cyberwave API
- Base model download (optional, for custom weights)
- WandB→Cyberwave logger monkey-patch
- Training execution via lerobot
- Status/ETA/completion updates to Cyberwave API
- Artifact compression and placement in results_folder

Usage:
    from cw_trainer import CwTrainer

    trainer = CwTrainer(params, model_slug="smolvla")
    trainer.setup()
    result = trainer.run()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import tarfile
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

import requests

from cw_processor import C, cprint

if TYPE_CHECKING:
    from base_trainer import BaseVLATrainer
    from lerobot.configs.train import TrainPipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_ROOT = Path("/data/cyberwave_runtime")
FALLBACK_RUNTIME_ROOT = Path("/tmp/cyberwave_runtime")


def _resolve_runtime_root() -> Path:
    """Prefer /data disk for downloads and cache files."""
    candidates = [
        os.environ.get("CYBERWAVE_RUNTIME_ROOT"),
        str(DEFAULT_RUNTIME_ROOT),
        str(FALLBACK_RUNTIME_ROOT),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    raise RuntimeError("Unable to create a runtime directory under /data or /tmp")


def _get_api_base_url(environment: str) -> str:
    """Get Cyberwave API base URL for the given environment."""
    for key in ("CYBERWAVE_API_URL", "CYBERWAVE_BASE_URL"):
        url = os.environ.get(key)
        if url:
            return url.rstrip("/")
    env_l = (environment or "").strip().lower()
    if env_l in ("local", "localhost", "dev", "development"):
        return "http://localhost:8000"
    if environment == "production":
        return "https://api.cyberwave.com"
    return "https://api-dev.cyberwave.com"


def _is_local_api_url(url: str) -> bool:
    u = (url or "").lower()
    return "localhost" in u or "127.0.0.1" in u


def _requests_get_with_auth(url: str, token: str, **kwargs: Any) -> requests.Response:
    """GET with Bearer, then Token if 401/403 (matches backend CustomTokenAuthentication)."""
    tok = (token or "").strip()
    if not tok:
        return requests.get(url, **kwargs)
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {tok}"
    resp = requests.get(url, headers=headers, **kwargs)
    if resp.status_code in (401, 403):
        headers["Authorization"] = f"Token {tok}"
        resp = requests.get(url, headers=headers, **kwargs)
    return resp


def _requests_put_with_auth(url: str, token: str, **kwargs: Any) -> requests.Response:
    """PUT with Bearer, then Token if 401/403."""
    tok = (token or "").strip()
    if not tok:
        return requests.put(url, **kwargs)
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {tok}"
    resp = requests.put(url, headers=headers, **kwargs)
    if resp.status_code in (401, 403):
        headers["Authorization"] = f"Token {tok}"
        resp = requests.put(url, headers=headers, **kwargs)
    return resp


def _get_trainer_registry() -> dict[str, type[BaseVLATrainer]]:
    """Lazy import trainer classes to avoid circular imports."""
    from smolvla_trainer import SmolVLATrainer

    return {
        SmolVLATrainer.MODEL_SLUG: SmolVLATrainer,
    }


@dataclass
class LogEvent:
    """Event structure for the logging queue."""

    event_type: str  # "metrics", "estimate", "checkpoint", "completed", "failed"
    payload: dict[str, Any]
    step: int | None = None


LogCallback = Callable[[LogEvent], None]


class CyberwaveLogger:
    """Drop-in replacement for lerobot's WandBLogger.

    This class only handles format conversion from lerobot to Cyberwave format.
    It does NOT make any API calls - instead it puts events into a queue.
    The CwTrainer class owns the queue and handles all Cyberwave API communication.

    Implements the same public interface as lerobot.rl.wandb_utils.WandBLogger:
    - __init__(cfg: TrainPipelineConfig)
    - log_dict(d, step, mode, custom_step_key)
    - log_policy(checkpoint_dir)
    - log_video(video_path, step, mode)
    """

    def __init__(
        self,
        cfg: "TrainPipelineConfig",
        *,
        on_log: LogCallback | None = None,
        max_steps: int | None = None,
    ):
        """Initialize the logger.

        Args:
            cfg: TrainPipelineConfig from lerobot
            on_log: Callback invoked when a log event is ready to send
            max_steps: Total training steps (for ETA calculation)
        """
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self._on_log = on_log
        self._max_steps = max_steps or cfg.steps
        self._start_time: float | None = None

        cprint("  CyberwaveLogger initialized (queue mode)", C.CYAN)

    def log_dict(
        self,
        d: dict[str, Any],
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        """Convert metrics to Cyberwave format and queue for sending.

        Payload follows MLTrainingMetrics dataclass structure:
        - step: int
        - log: dict[str, Any]
        - eta: datetime ISO string
        - loss: float
        """
        if mode not in {"train", "eval"}:
            raise ValueError(f"Invalid mode: {mode}")
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        effective_step: int
        if step is not None:
            effective_step = step
        elif custom_step_key is not None:
            effective_step = int(d.get(custom_step_key, 0))
        else:
            effective_step = 0

        if self._start_time is None:
            self._start_time = time.time()

        log_dict = {}
        loss_value: float = 0.0
        for name, value in d.items():
            if not isinstance(value, (int, float, str)):
                continue
            formatted_name = f"{mode}/{name.replace('_', ' ').title()}"
            log_dict[formatted_name] = value
            if "loss" in name.lower() and isinstance(value, (int, float)):
                loss_value = float(value)

        eta_iso = self._compute_eta_iso(effective_step)

        payload: dict[str, Any] = {
            "step": effective_step,
            "log": log_dict,
            "loss": loss_value,
        }
        if eta_iso:
            payload["eta"] = eta_iso

        self._emit(LogEvent(event_type="metrics", payload=payload, step=effective_step))

    def _compute_eta_iso(self, current_step: int) -> str | None:
        """Compute ETA as ISO string for the current step."""
        if self._start_time is None or self._max_steps is None or current_step <= 0:
            return None

        elapsed = time.time() - self._start_time
        steps_remaining = self._max_steps - current_step
        seconds_per_step = elapsed / current_step
        estimated_remaining = steps_remaining * seconds_per_step

        eta = datetime.now(timezone.utc) + timedelta(seconds=estimated_remaining)
        return eta.isoformat()

    def log_policy(self, checkpoint_dir: Path) -> None:
        """Log policy checkpoint event."""
        if self.cfg.disable_artifact:
            return
        step_id = checkpoint_dir.name
        payload = {
            "metadata": {"checkpoint_dir": str(checkpoint_dir), "step_id": step_id},
            "update_type": "checkpoint",
        }
        self._emit(LogEvent(event_type="checkpoint", payload=payload))
        logger.info(f"[CyberwaveLogger] Checkpoint saved at step {step_id}: {checkpoint_dir}")

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        """Log video (no-op for Cyberwave)."""
        pass

    def _emit(self, event: LogEvent) -> None:
        """Emit an event to the callback if registered."""
        if self._on_log is not None:
            try:
                self._on_log(event)
            except Exception as e:
                logger.error(f"[CyberwaveLogger] Error in log callback: {e}")


class CwTrainer:
    """Cyberwave training orchestrator.

    Handles:
    - Dataset download from Cyberwave endpoint
    - Base model download (optional, for custom weights)
    - WandBLogger monkey-patch to redirect metrics to Cyberwave
    - Training execution via lerobot
    - Artifact compression to results_folder
    - Status/completion updates
    """

    def __init__(self, params: dict[str, Any], *, model_slug: str):
        """Initialize the trainer.

        Args:
            params: Training parameters from Cyberwave JSON payload
            model_slug: Model identifier (e.g. "smolvla") for trainer registry lookup
        """
        self.params = params
        self.model_slug = model_slug
        self.training_uuid = params.get("cyberwave_training_uuid") or params.get("training_uuid", "")
        self.token = (
            params.get("cyberwave_token")
            or os.environ.get("CYBERWAVE_API_KEY")
            or os.environ.get("CYBERWAVE_API_TOKEN")
            or ""
        )
        env_param = params.get("environment")
        self.environment: str = (
            env_param if isinstance(env_param, str) else os.environ.get("CYBERWAVE_ENVIRONMENT", "production")
        )
        self.results_folder = Path(params.get("results_folder", "./runs/artifacts"))
        self._api_base_url = _get_api_base_url(self.environment)
        # Workload JSON often carries an API token for the deployed environment; for local Docker
        # use ~/.cyberwave/credentials.json (merged into env by train.py) instead.
        if _is_local_api_url(self._api_base_url):
            env_tok = (
                os.environ.get("CYBERWAVE_API_KEY") or os.environ.get("CYBERWAVE_API_TOKEN") or ""
            ).strip()
            if env_tok:
                self.token = env_tok
        self._runtime_root = _resolve_runtime_root()
        self._tmp_dir = self._runtime_root / "tmpfiles"
        self._weights_cache = self._runtime_root / "cyberwave_vla_weights"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        registry = _get_trainer_registry()
        trainer_cls = registry.get(model_slug)
        if trainer_cls is None:
            available = ", ".join(registry.keys())
            raise ValueError(f"Unknown model_slug '{model_slug}'. Available: {available}")
        self.trainer: BaseVLATrainer = trainer_cls()

        self.dataset_root: Path | None = None
        self.dataset_repo_id: str | None = None
        self.base_model_path: str | None = None
        self.output_dir: Path | None = None
        self.training_cfg: TrainPipelineConfig | None = None
        self._cyberwave_logger: CyberwaveLogger | None = None

        self._log_queue: queue.Queue[LogEvent | None] = queue.Queue()
        self._log_thread: threading.Thread | None = None
        self._log_thread_running = False

        cprint("\n=== CwTrainer Initialized ===", C.BOLD + C.CYAN)
        cprint(f"  Model: {model_slug}", C.WHITE)
        cprint(f"  Training UUID: {self.training_uuid or '(not set)'}", C.WHITE)
        cprint(f"  Environment: {self.environment}", C.WHITE)
        cprint(f"  API endpoint: {self._api_base_url}", C.DIM)

    def setup(self) -> None:
        """Prepare for training: download dataset/model, build config, patch logger."""
        cprint("\n--- Setup Phase ---", C.BOLD + C.YELLOW)

        self._download_dataset()
        self._resolve_base_model()
        self._build_training_config()
        self._patch_wandb_logger()

        cprint("  Setup complete", C.GREEN)

    def run(self) -> dict[str, Any]:
        """Execute training and handle completion.

        Returns:
            Result dictionary with status and artifact path.
        """
        if self.training_cfg is None:
            raise RuntimeError("Must call setup() before run()")

        cprint("\n--- Training Phase ---", C.BOLD + C.YELLOW)
        self._start_log_thread()
        self._send_status("training")

        try:
            from lerobot.scripts.lerobot_train import train

            cprint("  Starting lerobot training...", C.CYAN)
            train(self.training_cfg)

            cprint("\n--- Compression Phase ---", C.BOLD + C.YELLOW)
            self._send_status("compressing")
            artifact_path = self._compress_results()

            cprint("\n--- Completion ---", C.BOLD + C.GREEN)
            self._send_completion(str(artifact_path))

            result = {
                "status": "completed",
                "artifact": str(artifact_path),
                "training_uuid": self.training_uuid,
            }
            cprint("  Training completed successfully", C.GREEN)
            cprint(f"  Artifact: {artifact_path}", C.WHITE)
            return result

        except Exception as e:
            cprint(f"\n  Training failed: {e}", C.RED)
            self._send_status("failed", error=str(e))
            raise
        finally:
            self._stop_log_thread()

    # -------------------------------------------------------------------------
    # Log Queue & Cyberwave API Communication
    # -------------------------------------------------------------------------

    def _on_log_event(self, event: LogEvent) -> None:
        """Callback invoked by CyberwaveLogger when a log event is ready.

        This is called from the training thread; we just put the event in the
        queue for the background thread to send.
        """
        self._log_queue.put(event)

    def _start_log_thread(self) -> None:
        """Start background thread for processing log events."""
        if not self.training_uuid:
            cprint("  No training_uuid, log events will not be sent", C.YELLOW)
            return

        self._log_thread_running = True
        self._log_thread = threading.Thread(
            target=self._log_thread_worker,
            name="CwTrainer-LogWorker",
            daemon=True,
        )
        self._log_thread.start()
        cprint("  Log worker thread started", C.DIM)

    def _stop_log_thread(self) -> None:
        """Stop the background log processing thread."""
        if self._log_thread is None:
            return

        self._log_thread_running = False
        self._log_queue.put(None)

        self._log_thread.join(timeout=5.0)
        if self._log_thread.is_alive():
            logger.warning("[CwTrainer] Log thread did not stop gracefully")
        self._log_thread = None
        cprint("  Log worker thread stopped", C.DIM)

    def _log_thread_worker(self) -> None:
        """Background worker that processes log events and sends to Cyberwave API."""
        while self._log_thread_running:
            try:
                event = self._log_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event is None:
                break

            self._send_log_event(event)

    def _send_log_event(self, event: LogEvent) -> None:
        """Send a single log event to Cyberwave API.

        Routes to the appropriate endpoint based on event type:
        - metrics: PUT /mltrainings/{uuid}/metrics
        - others: PUT /mltrainings/{uuid}
        """
        if not self.training_uuid:
            return

        base_url = f"{self._api_base_url}/api/v1/mltrainings/{self.training_uuid}"

        if event.event_type == "metrics":
            url = f"{base_url}/metrics"
        else:
            url = base_url

        try:
            response = _requests_put_with_auth(
                url,
                self.token,
                json=event.payload,
                timeout=30,
            )
            if response.status_code == 200:
                if event.event_type == "metrics" and event.step is not None:
                    if event.step % 100 == 0:
                        logger.info(f"[Cyberwave] Logged metrics at step {event.step}")
            else:
                logger.warning(
                    f"[Cyberwave] Failed to send {event.event_type}: status {response.status_code}"
                )
        except Exception as e:
            logger.error(f"[Cyberwave] Error sending {event.event_type}: {e}")

    def _send_status(self, status: str, error: str | None = None) -> None:
        """Send status update to Cyberwave API."""
        if not self.training_uuid:
            return

        payload: dict[str, Any] = {"status": status}
        if error:
            payload["metadata"] = {"error": error}

        url = f"{self._api_base_url}/api/v1/mltrainings/{self.training_uuid}"
        try:
            _requests_put_with_auth(
                url,
                self.token,
                json=payload,
                timeout=30,
            )
            cprint(f"  Status: {status}", C.CYAN if status != "failed" else C.RED)
        except Exception as e:
            logger.warning(f"Failed to send status update: {e}")

    def _send_completion(self, checkpoint_path: str | None = None) -> None:
        """Send completion notification to Cyberwave API."""
        if not self.training_uuid:
            return

        payload: dict[str, Any] = {"status": "completed"}
        metadata: dict[str, Any] = {"completion_source": "cw_trainer.py"}
        if checkpoint_path:
            metadata["local_checkpoint_path"] = checkpoint_path
        payload["metadata"] = metadata

        url = f"{self._api_base_url}/api/v1/mltrainings/{self.training_uuid}"
        try:
            response = _requests_put_with_auth(
                url,
                self.token,
                json=payload,
                timeout=30,
            )
            if response.status_code == 200:
                cprint("  Completion status sent to Cyberwave", C.GREEN)
            else:
                logger.warning(f"[Cyberwave] Failed to send completion: status {response.status_code}")
        except Exception as e:
            logger.error(f"[Cyberwave] Error sending completion: {e}")

    # -------------------------------------------------------------------------
    # Dataset & Model Downloads
    # -------------------------------------------------------------------------

    def _resolved_dataset_repo_id(self, dataset_uuid: str) -> str:
        """LeRobot dataset repo_id from workload (repo_id) or default local/{uuid}."""
        for key in ("repo_id", "dataset_repo_id"):
            value = self.params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"local/{dataset_uuid}"

    def _download_dataset(self) -> None:
        """Download and extract dataset from Cyberwave API."""
        dataset_uuid = self.params.get("dataset_uuid")
        dataset_name = self.params.get("dataset_name")
        data_root_dir = self.params.get("data_root_dir", "./datasets")

        if not dataset_uuid:
            cprint("  No dataset_uuid provided, skipping download", C.YELLOW)
            self.dataset_root = Path(data_root_dir)
            self.dataset_repo_id = self.params.get("dataset_repo_id", "local/dataset")
            return

        cprint(f"  Downloading dataset: {dataset_name} ({dataset_uuid})", C.CYAN)

        data_root_path = Path(data_root_dir)
        data_root_path.mkdir(parents=True, exist_ok=True)
        target_dir = data_root_path / (dataset_name or dataset_uuid)

        if target_dir.exists() and any(target_dir.iterdir()):
            cprint(f"  Dataset already exists at {target_dir}", C.GREEN)
            self.dataset_root = target_dir
            self.dataset_repo_id = self._resolved_dataset_repo_id(str(dataset_uuid))
            return

        zip_url: str | None = None
        dataset_zip_url = self.params.get("dataset_zip_url")
        if isinstance(dataset_zip_url, str) and dataset_zip_url.startswith(
            ("http://", "https://")
        ):
            zip_url = dataset_zip_url
            cprint("  Using dataset_zip_url (HTTPS) from workload params", C.DIM)

        zip_url_endpoint = f"{self._api_base_url}/api/v1/datasets/{dataset_uuid}/zip?format=lerobot"
        try:
            if zip_url is None:
                response = _requests_get_with_auth(
                    zip_url_endpoint,
                    self.token,
                    timeout=30,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to get zip URL: status {response.status_code}, {response.text[:200]}"
                    )

                zip_data = response.json()
                zip_url = zip_data.get("zip_url")
                if not zip_url or zip_url == "No zip file found":
                    raise RuntimeError(f"No zip file found for dataset {dataset_uuid}")
                if not zip_url.startswith(("http://", "https://")):
                    raise RuntimeError(f"Invalid zip URL format: {zip_url[:100]}")

            cprint("  Downloading zip file...", C.DIM)
            zip_response = requests.get(zip_url, timeout=300, stream=True)
            if zip_response.status_code != 200:
                raise RuntimeError(f"Failed to download zip: status {zip_response.status_code}")

            target_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                suffix=".zip", delete=False, dir=self._tmp_dir
            ) as tmp_zip:
                tmp_zip_path = tmp_zip.name
                total = int(zip_response.headers.get("content-length", 0))
                downloaded = 0
                for chunk in zip_response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_zip.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (1024 * 1024 * 10) < 8192:
                            pct = (downloaded / total) * 100
                            print(f"\r    Progress: {pct:.1f}%", end="", flush=True)
                print()

            try:
                with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                    zip_ref.testzip()
                    file_list = zip_ref.namelist()
                    cprint(f"  Extracting {len(file_list)} files...", C.DIM)
                    zip_ref.extractall(target_dir)
                cprint(f"  Dataset extracted to {target_dir}", C.GREEN)
            finally:
                try:
                    os.unlink(tmp_zip_path)
                except Exception:
                    pass

            self.dataset_root = target_dir
            self.dataset_repo_id = self._resolved_dataset_repo_id(str(dataset_uuid))

        except Exception as e:
            cprint(f"  Dataset download failed: {e}", C.RED)
            self.dataset_root = Path(data_root_dir)
            self.dataset_repo_id = self._resolved_dataset_repo_id(str(dataset_uuid))

    def _resolve_base_model(self) -> None:
        """Resolve base model path (HF hub ID or downloaded weights)."""
        weights_url = self.params.get("weights_url")
        base_model = (
            self.params.get("base_model")
            or self.params.get("policy_repo_id")
            or "lerobot/smolvla_base"
        )

        if not weights_url:
            cprint(f"  Using base model: {base_model}", C.CYAN)
            self.base_model_path = base_model
            return

        cprint(f"  Downloading custom weights from {weights_url[:60]}...", C.CYAN)
        try:
            checkpoint_path = self._download_weights(weights_url)
            self.base_model_path = str(checkpoint_path)
            cprint(f"  Weights resolved to: {checkpoint_path}", C.GREEN)
        except Exception as e:
            cprint(f"  Weights download failed: {e}", C.RED)
            cprint(f"  Falling back to: {base_model}", C.YELLOW)
            self.base_model_path = base_model

    def _download_weights(self, weights_url: str) -> Path:
        """Download and extract weights archive from URL."""
        download_dir = self._weights_cache
        download_dir.mkdir(parents=True, exist_ok=True)

        def _get_with_auth(url: str, stream: bool = True):
            token_candidates = [self.token, os.environ.get("CYBERWAVE_API_KEY")]
            token = next(
                (t.strip() for t in token_candidates if t and t.strip()), None
            )
            headers = {}
            if token:
                headers = {"Authorization": f"Bearer {token}"}
            resp = requests.get(url, headers=headers, stream=stream, timeout=300)
            if resp.status_code in (401, 403) and token:
                headers = {"Authorization": f"Token {token}"}
                resp = requests.get(url, headers=headers, stream=stream, timeout=300)
            return resp

        response = _get_with_auth(weights_url, stream=True)
        response.raise_for_status()

        resolved_download_url = weights_url
        resolved_checkpoint_path = None
        content_type = (response.headers.get("content-type") or "").lower()

        if "application/json" in content_type:
            payload = response.json()
            signed_url = payload.get("signed_url")
            resolved_checkpoint_path = payload.get("checkpoint_path")
            if not signed_url:
                raise ValueError("weights_url returned JSON but missing 'signed_url'")
            resolved_download_url = signed_url
            response = requests.get(signed_url, stream=True, timeout=300)
            response.raise_for_status()

        default_display = Path(urlparse(weights_url).path).name or "checkpoint.tar"
        if resolved_checkpoint_path:
            display_name = Path(resolved_checkpoint_path).name or default_display
        else:
            display_name = Path(urlparse(resolved_download_url).path).name or default_display

        cache_key_src = resolved_checkpoint_path or resolved_download_url or weights_url
        cache_key = hashlib.sha256(cache_key_src.encode("utf-8")).hexdigest()[:16]
        cache_dir = download_dir / "cache" / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        download_path = cache_dir / display_name
        extract_dir = cache_dir / "checkpoint"

        def _resolve_checkpoint(extracted_root: Path) -> Path | None:
            if (extracted_root / "config.json").exists():
                return extracted_root
            config_candidates = sorted(
                {p.parent for p in extracted_root.rglob("config.json") if p.is_file()},
                key=lambda p: len(p.parts),
            )
            if config_candidates:
                return config_candidates[0]
            adapter_candidates = sorted(
                {
                    p.parent
                    for p in extracted_root.rglob("*")
                    if p.is_file() and p.name.startswith("adapter_")
                },
                key=lambda p: len(p.parts),
            )
            if adapter_candidates:
                return adapter_candidates[0]
            return None

        reused_cached = False
        if download_path.exists() and download_path.stat().st_size > 0:
            reused_cached = True
            cprint(f"    Using cached download: {download_path}", C.DIM)
        else:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 50) < 8192 * 1024:
                        pct = (downloaded / total_size) * 100
                        print(f"\r    Progress: {pct:.1f}%", end="", flush=True)
            print()

        try:
            is_tar = tarfile.is_tarfile(download_path)
        except Exception:
            is_tar = False

        if is_tar:
            if reused_cached and extract_dir.exists():
                cached = _resolve_checkpoint(extract_dir)
                if cached:
                    return cached

            extract_dir.mkdir(exist_ok=True)
            with tarfile.open(download_path, "r:*") as tar:
                tar.extractall(extract_dir)

            checkpoint = _resolve_checkpoint(extract_dir)
            if checkpoint:
                return checkpoint
            raise ValueError(f"No valid checkpoint found in {extract_dir}")

        if download_path.is_dir():
            return download_path

        raise ValueError(f"Downloaded weights are not a tar archive: {download_path}")

    def _build_training_config(self) -> None:
        """Build TrainPipelineConfig using the model-specific trainer."""
        if self.dataset_root is None or self.base_model_path is None:
            raise RuntimeError("Dataset and base model must be resolved first")

        self.output_dir = Path(
            self.params.get("output_dir", f"./outputs/train/{self.training_uuid or 'run'}")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cprint("  Building training config...", C.CYAN)
        self.training_cfg = self.trainer.build_pipeline_config(
            self.params,
            dataset_root=self.dataset_root,
            dataset_repo_id=self.dataset_repo_id or "local/dataset",
            base_model_path=self.base_model_path,
            output_dir=self.output_dir,
        )
        cprint(f"  Config built: {self.training_cfg.steps} steps, batch_size={self.training_cfg.batch_size}", C.GREEN)

    def _patch_wandb_logger(self) -> None:
        """Monkey-patch lerobot's WandBLogger with CyberwaveLogger."""
        if not self.training_uuid:
            cprint("  No training_uuid, skipping logger patch", C.YELLOW)
            return

        cprint("  Patching WandBLogger -> CyberwaveLogger...", C.CYAN)

        on_log_callback = self._on_log_event

        try:
            import lerobot.rl.wandb_utils as wandb_module

            class PatchedWandBLogger(CyberwaveLogger):
                """Patched logger that uses queue-based logging via CwTrainer."""

                def __init__(inner_self, cfg: "TrainPipelineConfig"):
                    super().__init__(
                        cfg,
                        on_log=on_log_callback,
                        max_steps=cfg.steps,
                    )
                    self._cyberwave_logger = inner_self

            wandb_module.WandBLogger = PatchedWandBLogger
            cprint("  Patched lerobot.rl.wandb_utils.WandBLogger", C.GREEN)
        except ImportError as e:
            cprint(f"  Warning: Could not patch wandb_utils: {e}", C.YELLOW)

        try:
            import lerobot.scripts.lerobot_train as train_module
            import lerobot.rl.wandb_utils as wandb_module_ref

            if hasattr(train_module, "WandBLogger"):
                train_module.WandBLogger = wandb_module_ref.WandBLogger
                cprint("  Patched lerobot.scripts.lerobot_train.WandBLogger", C.GREEN)
        except (ImportError, NameError) as e:
            cprint(f"  Warning: Could not patch train module: {e}", C.YELLOW)

        os.environ["WANDB_MODE"] = "disabled"
        cprint("  Set WANDB_MODE=disabled", C.DIM)

    def _compress_results(self) -> Path:
        """Compress training results into a single tarball in results_folder."""
        if self.output_dir is None:
            raise RuntimeError("No output_dir set")

        checkpoint_dir = self.output_dir / "checkpoints" / "last"
        if not checkpoint_dir.exists():
            candidates = list((self.output_dir / "checkpoints").glob("*"))
            if candidates:
                checkpoint_dir = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                checkpoint_dir = self.output_dir

        self.results_folder.mkdir(parents=True, exist_ok=True)
        artifact_name = f"{self.training_uuid or 'checkpoint'}.tar.gz"
        artifact_path = self.results_folder / artifact_name

        cprint(f"  Compressing {checkpoint_dir} -> {artifact_path}", C.CYAN)

        with tarfile.open(artifact_path, "w:gz") as tar:
            tar.add(checkpoint_dir, arcname="checkpoint")

        size_mb = artifact_path.stat().st_size / (1024 * 1024)
        cprint(f"  Artifact created: {artifact_path} ({size_mb:.1f}MB)", C.GREEN)

        return artifact_path


def load_json_argument(json_or_path: str) -> dict[str, Any]:
    """Load JSON from a file path or inline string.

    Cloud Node typically passes a path to a params file.
    """
    candidate = Path(json_or_path)
    if candidate.exists() and candidate.is_file():
        try:
            raw = candidate.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"Error reading params file {candidate}: {e}")

    try:
        return json.loads(json_or_path)
    except json.JSONDecodeError:
        raise FileNotFoundError(
            f"Expected a path to a JSON params file or inline JSON, got: {json_or_path!r}"
        )
