from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from base_resolver import BaseVLAResolver

logger = logging.getLogger(__name__)


# Terminal colors
class C:
    """Terminal color codes for readable output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    # Bright colors
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


def cprint(msg: str, color: str = C.RESET, bold: bool = False) -> None:
    """Print colored message to stdout."""
    prefix = C.BOLD if bold else ""
    print(f"{prefix}{color}{msg}{C.RESET}", flush=True)


def apply_cyberwave_credentials_env() -> None:
    """Load Cyberwave credentials from ~/.cyberwave/credentials.json into environment.

    The Cyberwave CLI stores credentials at ~/.cyberwave/credentials.json after login.
    This function reads that file and sets environment variables if they are not already set:
    - CYBERWAVE_API_KEY / CYBERWAVE_API_TOKEN
    - CYBERWAVE_BASE_URL / CYBERWAVE_API_URL
    - CYBERWAVE_ENVIRONMENT

    This enables local Docker development against localhost:8000 without manually
    setting env vars.
    """
    from pathlib import Path

    credentials_path = Path.home() / ".cyberwave" / "credentials.json"
    if not credentials_path.exists():
        return

    try:
        with open(credentials_path, "r", encoding="utf-8") as f:
            creds = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load credentials from %s: %s", credentials_path, e)
        return

    # Map credential keys to environment variables
    mappings = [
        ("token", "CYBERWAVE_API_KEY"),
        ("api_key", "CYBERWAVE_API_KEY"),
        ("base_url", "CYBERWAVE_BASE_URL"),
        ("api_url", "CYBERWAVE_API_URL"),
        ("environment", "CYBERWAVE_ENVIRONMENT"),
    ]

    for cred_key, env_key in mappings:
        if cred_key in creds and not os.environ.get(env_key):
            os.environ[env_key] = str(creds[cred_key])
            logger.debug("Set %s from credentials.json", env_key)


# New signature: predict_fn(inputs_dict) -> raw action tensor/array
# inputs_dict: {"images": {training_name: np.uint8 HWC}, "state": np.ndarray, "instruction": str}
# returns: raw action chunk (torch.Tensor or np.ndarray) of shape [1, chunk_size, action_dim] or [chunk_size, action_dim]
PredictFn = Callable[[dict[str, Any]], Any]

# Legacy signature for backwards compatibility (will be removed)
LegacyPredictFn = Callable[
    [dict[str, bytes], dict[str, float], str], dict[str, float] | list[dict[str, float]]
]


def _get_resolver_registry() -> dict[str, type[BaseVLAResolver]]:
    """Lazy import resolver classes to avoid circular imports."""
    from smolvla_resolver import SmolVLAResolver

    return {
        "smolvla": SmolVLAResolver,
    }


@dataclass
class CameraBinding:
    """Holds state for a camera twin used by the inference loop."""

    role: str  # training_name (e.g. "front")
    twin_uuid: str
    twin: Any  # SDK Twin handle
    latest_bytes: bytes = b""
    latest_image: np.ndarray | None = None  # HWC uint8 RGB
    last_ts: float = 0.0


@dataclass
class InferenceRequest:
    robot_twin_uuid: str
    instruction: str = "perform the requested task"
    camera_twin_uuids: list[str] = field(default_factory=list)
    camera_endpoints_by_role: dict[str, str] = field(default_factory=dict)
    camera_sensor_ids: list[str] = field(default_factory=list)
    twin_calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration_robot_type: str | None = None
    seed: int | None = None
    joint_delta: float = 0.1
    source_type: str | None = None
    mode: str | None = None
    max_steps: int = 1
    wait_for_joint_update_seconds: float = 1.0
    workload_uuid: str | None = None
    # Execution control
    action_sleep_seconds: float = 0.1  # Sleep between publishing each action
    actions_per_cycle: int = (
        25  # How many actions to execute per inference cycle (from chunk of 50)
    )
    inference_loop: bool = (
        True  # If True, run multiple inference cycles; if False, single chunk
    )
    # Camera frame fetch: retry when API returns tiny payloads (errors, not ready yet)
    frame_fetch_max_attempts: int = 8
    frame_fetch_retry_delay_seconds: float = 0.08
    frame_min_valid_bytes: int = 512
    # Background camera polling interval (seconds)
    camera_poll_interval_seconds: float = 0.05
    # Model weights configuration
    weights_url: str | None = None  # URL to download model weights from
    policy_repo_id: str | None = None  # HuggingFace repo ID (e.g., "lerobot/smolvla_base")
    policy_revision: str | None = None  # HuggingFace revision/branch


def _get_param(
    params: dict[str, Any], snake_key: str, camel_key: str, default: Any = None
) -> Any:
    if snake_key in params:
        return params[snake_key]
    if camel_key in params:
        return params[camel_key]
    return default


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    raise ValueError(f"Expected list or string, got {type(value).__name__}")


def load_json_argument(argument: str) -> str:
    if os.path.isfile(argument):
        with open(argument, "r", encoding="utf-8") as handle:
            return handle.read()
    return argument


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected object, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


_LATEST_FRAME_TWIN_URL_RE = re.compile(r"/twins/([^/]+)/latest-frame(?:$|[?#])")


def _camera_reference_to_twin_uuid(value: Any) -> str:
    reference = str(value)
    match = _LATEST_FRAME_TWIN_URL_RE.search(reference)
    if match:
        return match.group(1)
    return reference


def _normalize_joint_limits(limits: Any) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for limit_name, limit_value in _coerce_mapping(limits).items():
        if isinstance(limit_value, (int, float)):
            normalized[str(limit_name)] = float(limit_value)
    return normalized


def _normalize_sampling_calibration(
    calibration: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for joint_name, joint_limits in _coerce_mapping(calibration).items():
        limits = _coerce_mapping(joint_limits)
        lower = limits.get("min", limits.get("lower"))
        upper = limits.get("max", limits.get("upper"))
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            normalized[str(joint_name)] = {
                "min": float(lower),
                "max": float(upper),
            }
    return normalized


def normalize_twin_calibration(
    calibration: Any,
    *,
    robot_type: str | None = None,
) -> dict[str, dict[str, float]]:
    if calibration is None:
        return {}

    if hasattr(calibration, "joint_calibration"):
        return {
            str(joint_name): {
                key: float(value)
                for key, value in {
                    "lower": getattr(joint_limits, "lower", None),
                    "upper": getattr(joint_limits, "upper", None),
                    "range_min": getattr(joint_limits, "range_min", None),
                    "range_max": getattr(joint_limits, "range_max", None),
                    "homing_offset": getattr(joint_limits, "homing_offset", None),
                }.items()
                if isinstance(value, (int, float))
            }
            for joint_name, joint_limits in _coerce_mapping(
                getattr(calibration, "joint_calibration", {})
            ).items()
        }

    calibration_mapping = _coerce_mapping(calibration)
    if calibration_mapping and all(
        key in {"leader", "follower"} for key in calibration_mapping
    ):
        selected_robot_type = robot_type or "leader"
        return {
            str(joint_name): _normalize_joint_limits(joint_limits)
            for joint_name, joint_limits in _coerce_mapping(
                calibration_mapping.get(selected_robot_type, {})
            ).items()
        }

    return {
        str(joint_name): _normalize_joint_limits(joint_limits)
        for joint_name, joint_limits in calibration_mapping.items()
    }


def parse_request_payload(raw_payload: str) -> InferenceRequest:
    params = json.loads(raw_payload)

    robot_twin_uuid = _get_param(params, "robot_twin_uuid", "robotTwinUuid")
    if not robot_twin_uuid:
        robot_twin_uuid = _get_param(params, "twin_uuid", "twinUuid")
    if not robot_twin_uuid:
        raise ValueError("Missing required parameter: 'robot_twin_uuid'")

    camera_endpoints_by_role = _coerce_mapping(
        _get_param(params, "camera_endpoints_by_role", "cameraEndpointsByRole", {})
    )
    camera_twin_uuids = _coerce_list(
        _get_param(params, "camera_twin_uuids", "cameraTwinUuids", [])
    )
    for endpoint in camera_endpoints_by_role.values():
        camera_twin_uuid = _camera_reference_to_twin_uuid(endpoint)
        if camera_twin_uuid not in camera_twin_uuids:
            camera_twin_uuids.append(camera_twin_uuid)

    calibration_robot_type = _get_param(
        params,
        "calibration_robot_type",
        "calibrationRobotType",
    )

    return InferenceRequest(
        robot_twin_uuid=str(robot_twin_uuid),
        instruction=str(
            _get_param(
                params,
                "instruction",
                "instruction",
                "perform the requested task",
            )
        ),
        camera_twin_uuids=camera_twin_uuids,
        camera_endpoints_by_role={
            key: str(value) for key, value in camera_endpoints_by_role.items()
        },
        camera_sensor_ids=_coerce_list(
            _get_param(params, "camera_sensor_ids", "cameraSensorIds", [])
        ),
        twin_calibration=normalize_twin_calibration(
            _get_param(params, "twin_calibration", "twinCalibration", {}),
            robot_type=str(calibration_robot_type) if calibration_robot_type else None,
        ),
        calibration_robot_type=str(calibration_robot_type)
        if calibration_robot_type
        else None,
        seed=_get_param(params, "seed", "seed"),
        joint_delta=float(_get_param(params, "joint_delta", "jointDelta", 0.1)),
        source_type=_get_param(params, "source_type", "sourceType"),
        mode=_get_param(params, "mode", "mode"),
        max_steps=max(1, int(_get_param(params, "max_steps", "maxSteps", 1))),
        wait_for_joint_update_seconds=float(
            _get_param(
                params,
                "wait_for_joint_update_seconds",
                "waitForJointUpdateSeconds",
                1.0,
            )
        ),
        workload_uuid=_get_param(params, "workload_uuid", "workloadUuid"),
        action_sleep_seconds=float(
            _get_param(params, "action_sleep_seconds", "actionSleepSeconds", 0.1)
        ),
        actions_per_cycle=int(
            _get_param(params, "actions_per_cycle", "actionsPerCycle", 25)
        ),
        inference_loop=bool(
            _get_param(params, "inference_loop", "inferenceLoop", True)
        ),
        frame_fetch_max_attempts=max(
            1,
            int(
                _get_param(
                    params,
                    "frame_fetch_max_attempts",
                    "frameFetchMaxAttempts",
                    8,
                )
            ),
        ),
        frame_fetch_retry_delay_seconds=float(
            _get_param(
                params,
                "frame_fetch_retry_delay_seconds",
                "frameFetchRetryDelaySeconds",
                0.08,
            )
        ),
        frame_min_valid_bytes=max(
            64,
            int(
                _get_param(
                    params,
                    "frame_min_valid_bytes",
                    "frameMinValidBytes",
                    512,
                )
            ),
        ),
        camera_poll_interval_seconds=float(
            _get_param(
                params,
                "camera_poll_interval_seconds",
                "cameraPollIntervalSeconds",
                0.05,
            )
        ),
        weights_url=_get_param(params, "weights_url", "weightsUrl"),
        policy_repo_id=_get_param(params, "policy_repo_id", "policyRepoId"),
        policy_revision=_get_param(params, "policy_revision", "policyRevision"),
    )


def _extract_joint_positions(payload: dict[str, Any]) -> dict[str, float]:
    """Extract joint positions from MQTT payload.

    MQTT joint state payloads look like:
    {
        'source_type': 'edge_follower',
        'positions': {'_1': 0.01, '_2': -1.54, '_3': 1.48, ...},
        'velocities': {...},
        'efforts': {...},
        ...
    }
    """
    joint_positions: dict[str, float] = {}

    # Primary format: 'positions' dict (actual MQTT format)
    if "positions" in payload and isinstance(payload["positions"], dict):
        for joint_name, position in payload["positions"].items():
            if isinstance(position, (int, float)):
                joint_positions[str(joint_name)] = float(position)
        return joint_positions

    # Legacy format: 'joint_states' dict
    if "joint_states" in payload and isinstance(payload["joint_states"], dict):
        for joint_name, joint_state in payload["joint_states"].items():
            if isinstance(joint_state, dict):
                position = joint_state.get("position")
            else:
                position = joint_state
            if isinstance(position, (int, float)):
                joint_positions[str(joint_name)] = float(position)

    # Single joint format
    joint_name = payload.get("joint_name")
    joint_state = payload.get("joint_state")
    if joint_name is not None:
        if isinstance(joint_state, dict):
            position = joint_state.get("position")
        else:
            position = joint_state
        if isinstance(position, (int, float)):
            joint_positions[str(joint_name)] = float(position)

    return joint_positions


def download_weights(
    weights_url: str,
    *,
    api_key: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Download model weights from a URL and return the local checkpoint path.

    Args:
        weights_url: URL to download weights from (e.g., Cyberwave MLModel weights endpoint)
        api_key: API key for authentication (defaults to CYBERWAVE_API_KEY env var)
        cache_dir: Directory to cache downloaded weights (defaults to ~/.cache/cyberwave/weights)

    Returns:
        Path to the extracted checkpoint directory
    """
    import hashlib
    import shutil
    import tarfile
    import tempfile
    import zipfile
    from pathlib import Path
    from urllib.parse import urlparse

    import requests

    if not api_key:
        api_key = os.environ.get("CYBERWAVE_API_KEY") or os.environ.get(
            "CYBERWAVE_API_TOKEN"
        )
    if not api_key:
        raise ValueError(
            "API key required for weights download. Set CYBERWAVE_API_KEY env var."
        )

    if not cache_dir:
        cache_dir = os.path.join(Path.home(), ".cache", "cyberwave", "weights")
    os.makedirs(cache_dir, exist_ok=True)

    url_hash = hashlib.sha256(weights_url.encode()).hexdigest()[:16]
    parsed = urlparse(weights_url)
    url_basename = os.path.basename(parsed.path) or "weights"
    checkpoint_dir = os.path.join(cache_dir, f"{url_basename}_{url_hash}")

    # Helper to find actual model directory within extracted archive
    model_markers = ["config.json", "train_config.json", "adapter_config.json"]

    def find_model_dir(base: str, max_depth: int = 3) -> str:
        """Recursively find directory containing model config files."""
        for marker in model_markers:
            if os.path.exists(os.path.join(base, marker)):
                return base

        pretrained = os.path.join(base, "pretrained_model")
        if os.path.isdir(pretrained):
            for marker in model_markers:
                if os.path.exists(os.path.join(pretrained, marker)):
                    return pretrained

        if max_depth > 0:
            subdirs = [
                d for d in os.listdir(base)
                if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
            ]
            if len(subdirs) == 1:
                return find_model_dir(os.path.join(base, subdirs[0]), max_depth - 1)

        return base

    marker_file = os.path.join(checkpoint_dir, ".download_complete")
    if os.path.exists(marker_file):
        resolved = find_model_dir(checkpoint_dir)
        cprint(f"  Using cached weights: {resolved}", C.GREEN)
        return resolved

    cprint(f"  Fetching weights from {weights_url[:60]}...", C.CYAN)
    headers = {"Authorization": f"Bearer {api_key}"}

    # Step 1: Call the weights endpoint to get signed URL
    response = requests.get(weights_url, headers=headers, timeout=60)
    response.raise_for_status()

    # Check if response is JSON with signed_url (Cyberwave MLModel weights endpoint)
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        data = response.json()
        signed_url = data.get("signed_url")
        if not signed_url:
            raise ValueError(
                f"Weights endpoint returned JSON but no signed_url: {list(data.keys())}"
            )
        cprint(f"  Got signed URL (expires: {data.get('expires_at', 'unknown')})", C.DIM)

        # Step 2: Download from the signed URL (no auth needed, it's pre-signed)
        cprint("  Downloading from signed URL...", C.CYAN)
        response = requests.get(signed_url, stream=True, timeout=300)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")

    content_disp = response.headers.get("Content-Disposition", "")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        with open(tmp_path, "rb") as f:
            magic = f.read(4)

        is_zip = (
            magic[:2] == b"PK"
            or "zip" in content_type.lower()
            or ".zip" in content_disp.lower()
        )
        is_zstd = (
            magic == b"\x28\xb5\x2f\xfd"  # zstd magic
            or "zstd" in content_type.lower()
            or ".zst" in content_disp.lower()
        )
        is_tar = (
            magic[:2] == b"\x1f\x8b"  # gzip
            or "gzip" in content_type.lower()
            or "tar" in content_type.lower()
            or any(
                ext in content_disp.lower()
                for ext in [".tar.gz", ".tgz", ".tar"]
            )
        )

        if is_zip:
            cprint("  Extracting ZIP archive...", C.CYAN)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(checkpoint_dir)
        elif is_zstd:
            cprint("  Extracting TAR.ZST archive...", C.CYAN)
            try:
                import zstandard as zstd
            except ImportError:
                raise ImportError(
                    "zstandard package required for .tar.zst files. "
                    "Install with: pip install zstandard"
                )
            with open(tmp_path, "rb") as compressed:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(compressed) as reader:
                    with tarfile.open(fileobj=reader, mode="r|") as tf:
                        tf.extractall(checkpoint_dir)
        elif is_tar:
            cprint("  Extracting TAR archive...", C.CYAN)
            with tarfile.open(tmp_path, "r:*") as tf:
                tf.extractall(checkpoint_dir)
        else:
            cprint("  Weights file is not an archive, copying directly...", C.YELLOW)
            shutil.copy(tmp_path, os.path.join(checkpoint_dir, "weights"))

        # Resolve to actual model directory
        resolved_dir = find_model_dir(checkpoint_dir)
        if resolved_dir != checkpoint_dir:
            cprint(f"  Resolved model directory: {resolved_dir}", C.DIM)
            checkpoint_dir = resolved_dir

        with open(marker_file, "w") as f:
            f.write(weights_url)

        cprint(f"  ✓ Weights downloaded to {checkpoint_dir}", C.GREEN)
        return checkpoint_dir

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


class CwProcessor:
    """Process inference requests via Cyberwave SDK.

    Handles all Cyberwave I/O: SDK client, MQTT, frames, joint states, publishing.
    Also handles model-agnostic data preparation via the resolver pattern.

    Control loop:
    1. Connect MQTT and subscribe to joint states
    2. Wait for initial joint states (required before proceeding)
    3. Loop: observe (get_inputs) -> predict -> convert actions -> execute via MQTT -> repeat
    """

    def __init__(
        self,
        request: InferenceRequest,
        *,
        model_slug: str,
        checkpoint: str,
        predict_fn: PredictFn,
        source_subtype: str | None = None,
        cw: Any | None = None,
        now_fn: Callable[[], float] = time.time,
    ):
        """Initialize processor with model metadata and predict function.

        Args:
            request: Parsed inference request with robot/camera config
            model_slug: Model identifier (e.g., "smolvla") used to lookup resolver
            checkpoint: Path to model checkpoint (passed to resolver for config loading)
            predict_fn: Function that takes inputs dict and returns raw action tensor
            source_subtype: Optional override for MQTT source_subtype (defaults to model_slug)
            cw: Optional preconfigured Cyberwave client (mostly for tests)
            now_fn: Time function (for testing)
        """
        self.request = request
        self.model_slug = model_slug
        self.checkpoint = checkpoint
        self.predict_fn = predict_fn
        self.source_subtype = source_subtype or model_slug
        self.cw = cw
        self.now_fn = now_fn

        # Build resolver from registry
        resolver_registry = _get_resolver_registry()
        resolver_cls = resolver_registry.get(model_slug)
        if resolver_cls is None:
            raise ValueError(
                f"Unknown model_slug: {model_slug!r}. "
                f"Available: {list(resolver_registry.keys())}"
            )
        self.resolver: BaseVLAResolver = resolver_cls(checkpoint)

        # Cyberwave state (initialized in setup())
        self.client: Any | None = None
        self.robot_twin: Any | None = None
        self.twin_calibration: dict[str, dict[str, float]] = {}
        self.initial_joints: dict[str, float] = {}
        self.joint_state_source: str = "unknown"

        # Joint configuration (derived in setup())
        self.joint_names: list[str] = []
        self.num_joints: int = 0

        # Camera mapping: training_name -> runtime_key (built in setup())
        self.camera_mapping: dict[str, str] = {}

        # Continuous joint state tracking
        self._current_joints: dict[str, float] = {}
        self._joint_lock = threading.Lock()
        self._joint_update_event = threading.Event()

        # Camera bindings with background frame fetchers
        self.cameras: dict[str, CameraBinding] = {}
        self._camera_threads: list[threading.Thread] = []
        self._camera_stop = threading.Event()
        self._camera_lock = threading.Lock()

    def _cyberwave_kwargs_from_env(self) -> dict[str, Any]:
        """Build Cyberwave client kwargs from environment variables."""
        kwargs: dict[str, Any] = {}

        api_key = os.environ.get("CYBERWAVE_API_KEY") or os.environ.get(
            "CYBERWAVE_API_TOKEN"
        )
        if api_key:
            kwargs["api_key"] = api_key
            cprint(f"  API Key: {api_key[:8]}...{api_key[-4:]}", C.GREEN)
        else:
            cprint("  ✗ CYBERWAVE_API_KEY not set!", C.RED, bold=True)

        mqtt_host = os.environ.get("CYBERWAVE_MQTT_HOST")
        mqtt_port = os.environ.get("CYBERWAVE_MQTT_PORT")
        mqtt_username = os.environ.get("CYBERWAVE_MQTT_USERNAME")

        if mqtt_host:
            kwargs["mqtt_host"] = mqtt_host
        if mqtt_port:
            kwargs["mqtt_port"] = int(mqtt_port)
        if mqtt_username:
            kwargs["mqtt_username"] = mqtt_username

        base_url = os.environ.get("CYBERWAVE_BASE_URL") or os.environ.get(
            "CYBERWAVE_API_URL"
        )
        if base_url:
            kwargs["base_url"] = base_url

        topic_prefix = os.environ.get("CYBERWAVE_MQTT_TOPIC_PREFIX") or os.environ.get(
            "CYBERWAVE_ENVIRONMENT"
        )
        if topic_prefix and topic_prefix.lower() != "production":
            kwargs["topic_prefix"] = topic_prefix

        return kwargs

    def setup(self) -> None:
        """Create SDK client, fetch twin, resolve calibration, subscribe to joints."""
        cprint("\n" + "=" * 50, C.MAGENTA, bold=True)
        cprint("  CYBERWAVE SETUP", C.MAGENTA, bold=True)
        cprint("=" * 50, C.MAGENTA, bold=True)

        if self.cw is None:
            from cyberwave import Cyberwave

            cprint("  Creating Cyberwave client...", C.CYAN)
            self.cw = Cyberwave(**self._cyberwave_kwargs_from_env())
        else:
            cprint("  Using injected Cyberwave client", C.CYAN)

        self.client = self.cw
        cprint("  ✓ Client created", C.GREEN)

        if self.request.mode:
            self.client = self.client.affect(self.request.mode)

        self.robot_twin = self.client.twins.get(self.request.robot_twin_uuid)

        self.twin_calibration = {
            joint_name: dict(joint_limits)
            for joint_name, joint_limits in self.request.twin_calibration.items()
        }

        if hasattr(self.robot_twin, "get_calibration"):
            try:
                if self.request.calibration_robot_type:
                    sdk_calibration = normalize_twin_calibration(
                        self.robot_twin.get_calibration(
                            robot_type=self.request.calibration_robot_type
                        )
                    )
                else:
                    sdk_calibration = normalize_twin_calibration(
                        self.robot_twin.get_calibration()
                    )
                self.twin_calibration.update(sdk_calibration)
            except Exception as exc:
                logger.warning("Failed to fetch twin calibration: %s", exc)

        # Connect MQTT explicitly
        cprint("  Connecting to MQTT...", C.CYAN)
        try:
            self.client.mqtt.connect()
            cprint("  ✓ MQTT connected", C.GREEN)
        except Exception as exc:
            cprint(f"  ✗ MQTT connection failed: {exc}", C.RED)
            raise RuntimeError(f"MQTT connection failed: {exc}")

        # Subscribe to joint states
        cprint("  Subscribing to joint states...", C.CYAN)
        try:
            self.client.mqtt.subscribe_joint_states(
                self.request.robot_twin_uuid,
                self._on_mqtt_joint_update,
            )
        except Exception as exc:
            cprint(f"  ✗ Subscribe failed: {exc}", C.RED)

        # Wait for initial joint states
        wait_timeout = max(self.request.wait_for_joint_update_seconds, 5.0)
        cprint(f"  Waiting for joint states ({wait_timeout}s timeout)...", C.CYAN)
        deadline = time.time() + wait_timeout
        poll_interval = 0.2
        while time.time() < deadline:
            with self._joint_lock:
                if self._current_joints:
                    joint_vals = [
                        f"{v:.2f}" for v in list(self._current_joints.values())[:6]
                    ]
                    cprint(f"  ✓ Joints received: [{', '.join(joint_vals)}]", C.GREEN)
                    self.joint_state_source = "mqtt_subscription"
                    break
            time.sleep(poll_interval)
        else:
            cprint("  ✗ MQTT timeout - trying REST API...", C.YELLOW)
            try:
                fallback_joints = self.robot_twin.joints.get_all()
                with self._joint_lock:
                    self._current_joints = dict(fallback_joints)
                self.joint_state_source = "rest_poll"
                if fallback_joints:
                    cprint(f"  ✓ REST returned {len(fallback_joints)} joints", C.YELLOW)
                else:
                    cprint("  ✗ REST returned empty joints", C.RED)
            except Exception as exc:
                cprint(f"  ✗ REST failed: {exc}", C.RED)
                raise RuntimeError(
                    "Cannot get initial joint states via MQTT or REST API. "
                    "Check network connectivity and CYBERWAVE_API_KEY."
                )

        self.initial_joints = self.get_current_joints()
        logger.info("Initial joints acquired: %s", self.initial_joints)

        # Derive joint names from twin (hardcoded for SO-101 for now)
        self._derive_joint_names_from_twin()
        cprint(f"  Joint names: {self.joint_names}", C.CYAN)

        # Build camera mapping using resolver
        self._build_camera_mapping()
        if self.camera_mapping:
            cprint(f"  Camera mapping: {len(self.camera_mapping)} cameras", C.CYAN)
            for train_name, runtime_key in self.camera_mapping.items():
                cprint(f"    {train_name} <- {runtime_key}", C.DIM)

        # Create CameraBinding for each mapped camera and start background fetchers
        self._setup_camera_bindings()
        if self.cameras:
            cprint(
                f"  Started {len(self._camera_threads)} background camera fetchers",
                C.GREEN,
            )

        # Verify MQTT is actually working by checking connection state
        if hasattr(self.client.mqtt, "_connected") and not self.client.mqtt._connected:
            logger.error(
                "MQTT client reports not connected - actions will NOT reach the robot!"
            )
        elif self.joint_state_source == "rest_poll":
            logger.warning(
                "MQTT subscription did not work - proceeding but actions may not reach robot. "
                "Check CYBERWAVE_API_KEY is valid for MQTT."
            )

    def _on_mqtt_joint_update(self, payload: dict[str, Any]) -> None:
        """Callback for MQTT joint state updates."""
        positions = _extract_joint_positions(payload)
        if positions:
            with self._joint_lock:
                self._current_joints.update(positions)
            self._joint_update_event.set()
            logger.debug("MQTT joint update: %s", positions)

    def _derive_joint_names_from_twin(self) -> None:
        """Derive joint names from twin's universal schema.

        Uses the SDK's get_controllable_joint_names() method which returns
        joint names for revolute, prismatic, and continuous joints from the
        twin's schema.

        Raises:
            RuntimeError: If robot_twin is None or joint names cannot be derived.
        """
        if self.robot_twin is None:
            raise RuntimeError(
                "robot_twin is None - setup() must be called before _derive_joint_names_from_twin()"
            )

        if not hasattr(self.robot_twin, "get_controllable_joint_names"):
            raise RuntimeError(
                "robot_twin does not have get_controllable_joint_names() method. "
                "Ensure you are using cyberwave SDK >= 0.3.46"
            )

        joint_names = self.robot_twin.get_controllable_joint_names()
        if not joint_names:
            raise RuntimeError(
                f"Twin {self.request.robot_twin_uuid} has no controllable joints. "
                "Check that the twin's asset has a valid schema with revolute/prismatic/continuous joints."
            )

        self.joint_names = joint_names
        self.num_joints = len(self.joint_names)
        logger.info(
            "Derived joint names from twin schema: %s (num_joints=%d)",
            self.joint_names,
            self.num_joints,
        )

        # Validate that twin joints match model's expected dimensions.
        # Older resolvers may not implement these getters; skip validation gracefully.
        get_state_dim = getattr(self.resolver, "get_expected_state_dim", None)
        get_action_dim = getattr(self.resolver, "get_expected_action_dim", None)
        if not callable(get_state_dim) or not callable(get_action_dim):
            logger.warning(
                "Resolver %s does not expose get_expected_state_dim/get_expected_action_dim; "
                "skipping joint-count validation.",
                type(self.resolver).__name__,
            )
            return

        expected_state_dim = int(get_state_dim() or 0)
        expected_action_dim = int(get_action_dim() or 0)

        if expected_state_dim > 0 and self.num_joints != expected_state_dim:
            raise RuntimeError(
                f"Joint count mismatch: twin has {self.num_joints} controllable joints "
                f"({self.joint_names}), but model expects {expected_state_dim} state inputs. "
                f"Ensure the twin's asset matches the model training configuration."
            )

        if expected_action_dim > 0 and self.num_joints != expected_action_dim:
            raise RuntimeError(
                f"Joint count mismatch: twin has {self.num_joints} controllable joints "
                f"({self.joint_names}), but model expects {expected_action_dim} action outputs. "
                f"Ensure the twin's asset matches the model training configuration."
            )

        if expected_state_dim > 0:
            cprint(f"  ✓ Joint count matches model: {self.num_joints} joints", C.GREEN)

    def _build_camera_mapping(self) -> None:
        """Build camera mapping using resolver and runtime camera config."""
        if not self.resolver.training_camera_names:
            logger.warning(
                "Resolver has no training camera names - skipping camera mapping"
            )
            return

        # Determine runtime cameras from request
        if self.request.camera_endpoints_by_role:
            runtime_cameras = self.request.camera_endpoints_by_role
        elif self.request.camera_twin_uuids:
            runtime_cameras = self.request.camera_twin_uuids
        else:
            logger.warning(
                "Training expects %d cameras but no runtime cameras provided",
                len(self.resolver.training_camera_names),
            )
            return

        self.camera_mapping = self.resolver.build_camera_mapping(runtime_cameras)

    def _setup_camera_bindings(self) -> None:
        """Create CameraBinding for each mapped camera and start background fetchers."""
        if not self.camera_mapping or self.client is None:
            return

        for training_name, runtime_key in self.camera_mapping.items():
            # runtime_key is the role name (e.g., "camera_wrist")
            # Look up the full endpoint URL from camera_endpoints_by_role to extract UUID
            endpoint_or_uuid = self.request.camera_endpoints_by_role.get(
                runtime_key, runtime_key
            )
            twin_uuid = _camera_reference_to_twin_uuid(endpoint_or_uuid)
            try:
                camera_twin = self.client.twins.get(twin_uuid)
            except Exception as exc:
                logger.warning(
                    "Failed to fetch camera twin %s for role %s: %s",
                    twin_uuid,
                    training_name,
                    exc,
                )
                continue

            binding = CameraBinding(
                role=training_name,
                twin_uuid=twin_uuid,
                twin=camera_twin,
            )
            self.cameras[training_name] = binding

            thread = threading.Thread(
                target=self._camera_loop,
                args=(binding,),
                daemon=True,
                name=f"camera-{training_name}",
            )
            thread.start()
            self._camera_threads.append(thread)
            logger.info("Started background fetcher for camera %s", training_name)

        # Wait for all cameras to fetch their first frame
        self._wait_for_camera_frames()

    def _wait_for_camera_frames(self, timeout: float = 10.0) -> None:
        """Wait for all camera bindings to have at least one frame.

        Args:
            timeout: Maximum time to wait in seconds.
        """
        if not self.cameras:
            return

        start = self.now_fn()
        pending = set(self.cameras.keys())
        poll_interval = 0.1

        while pending and (self.now_fn() - start) < timeout:
            with self._camera_lock:
                for role in list(pending):
                    if self.cameras[role].latest_image is not None:
                        pending.discard(role)
                        cprint(f"  ✓ Camera {role} ready", C.GREEN)

            if pending:
                time.sleep(poll_interval)

        if pending:
            logger.warning(
                "Timeout waiting for camera frames: %s still pending after %.1fs",
                list(pending),
                timeout,
            )
            cprint(
                f"  ⚠ Cameras not ready after {timeout}s: {list(pending)}",
                C.YELLOW,
            )

    def _camera_loop(self, binding: CameraBinding) -> None:
        """Background loop that continuously fetches and decodes frames for a camera.

        TODO: Replace REST polling with direct WebRTC subscription once available.
        """
        while not self._camera_stop.is_set():
            try:
                raw = self._fetch_latest_frame_bytes(
                    f"camera:{binding.role}", binding.twin.get_latest_frame
                )
                if len(raw) >= self.request.frame_min_valid_bytes:
                    img = np.asarray(
                        Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8
                    )
                    with self._camera_lock:
                        binding.latest_bytes = raw
                        binding.latest_image = img
                        binding.last_ts = self.now_fn()
            except Exception as exc:
                logger.warning("camera %s fetch failed: %s", binding.role, exc)
            self._camera_stop.wait(self.request.camera_poll_interval_seconds)

    def get_inputs(self) -> dict[str, Any]:
        """Prepare model inputs from current observation.

        Reads cached frames from background fetchers (if available) or falls back
        to legacy REST fetch for sensor_ids / default camera.

        Returns:
            Dict with keys:
            - "images": {training_camera_name: np.ndarray (HWC uint8)}
            - "state": np.ndarray of joint positions
            - "instruction": str
        """
        joints = self.get_current_joints()
        images: dict[str, np.ndarray] = {}

        if self.cameras:
            with self._camera_lock:
                for role, binding in self.cameras.items():
                    if binding.latest_image is not None:
                        images[role] = binding.latest_image
        else:
            frames_bytes = self.get_frames()
            runtime_to_training = {v: k for k, v in self.camera_mapping.items()}
            min_frame = self.request.frame_min_valid_bytes

            for runtime_key, frame_bytes in frames_bytes.items():
                if not frame_bytes or len(frame_bytes) < min_frame:
                    logger.warning(
                        "Skipping invalid frame for camera %s (len=%d, min=%d)",
                        runtime_key,
                        len(frame_bytes) if frame_bytes else 0,
                        min_frame,
                    )
                    continue
                try:
                    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                    img_arr = np.asarray(img, dtype=np.uint8)
                except Exception as exc:
                    logger.warning(
                        "Failed to decode frame for camera %s: %s", runtime_key, exc
                    )
                    continue

                training_name = runtime_to_training.get(runtime_key, runtime_key)
                images[training_name] = img_arr
                if training_name != runtime_key:
                    logger.debug("Remapped camera '%s' -> '%s'", runtime_key, training_name)

        if not images:
            raise RuntimeError("No valid camera frames available from background fetchers")

        state = np.array(
            [joints.get(name, 0.0) for name in self.joint_names],
            dtype=np.float32,
        )

        return {
            "images": images,
            "state": state,
            "instruction": self.request.instruction,
        }

    def _convert_raw_actions(self, raw: Any) -> list[dict[str, float]]:
        """Convert raw action tensor to list of joint dicts.

        Args:
            raw: Raw action chunk from predict_fn. Expected shapes:
                - [1, chunk_size, action_dim] (batched)
                - [chunk_size, action_dim] (unbatched)

        Returns:
            List of dicts mapping joint name -> position value
        """
        # Handle torch tensors
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()

        arr = np.asarray(raw)

        # Strip batch dimension if present
        if arr.ndim == 3:
            arr = arr[0]  # [chunk_size, action_dim]

        chunk: list[dict[str, float]] = []
        for i in range(arr.shape[0]):
            # Take first num_joints values (action_dim may be larger)
            vals = arr[i, : self.num_joints]
            action_dict = {
                self.joint_names[j]: float(vals[j]) for j in range(self.num_joints)
            }
            chunk.append(action_dict)

        return chunk

    def get_current_joints(self) -> dict[str, float]:
        """Get current joint positions (thread-safe)."""
        with self._joint_lock:
            return dict(self._current_joints)

    def wait_for_joint_update(self, timeout: float = 1.0) -> bool:
        """Wait for a joint state update via MQTT."""
        self._joint_update_event.clear()
        return self._joint_update_event.wait(timeout)

    def get_joint_states(self) -> tuple[dict[str, float], str]:
        """Return current joint positions (MQTT-first, REST fallback)."""
        if self.robot_twin is None:
            raise RuntimeError(
                "CwProcessor.setup() must be called before get_joint_states()"
            )

        latest_from_mqtt: dict[str, float] = {}
        received_update = threading.Event()

        def on_update(payload: dict[str, Any]) -> None:
            positions = _extract_joint_positions(payload)
            if positions:
                latest_from_mqtt.update(positions)
                received_update.set()

        try:
            self.robot_twin.subscribe_joints(on_update)
            if self.request.wait_for_joint_update_seconds > 0:
                received_update.wait(self.request.wait_for_joint_update_seconds)
        except Exception as exc:
            logger.warning("Failed to subscribe to joint updates: %s", exc)

        fallback_joints = self.robot_twin.joints.get_all()
        if latest_from_mqtt:
            merged = dict(fallback_joints)
            merged.update(latest_from_mqtt)
            return merged, "mqtt_subscription"
        return fallback_joints, "rest_poll"

    def _fetch_latest_frame_bytes(self, label: str, fetch: Callable[[], Any]) -> bytes:
        """Call ``fetch`` until bytes look like a real image or attempts exhausted."""
        min_len = self.request.frame_min_valid_bytes
        max_attempts = self.request.frame_fetch_max_attempts
        delay = self.request.frame_fetch_retry_delay_seconds

        last: bytes = b""
        for attempt in range(max_attempts):
            raw = fetch()
            if raw is None:
                b = b""
            elif isinstance(raw, memoryview):
                b = raw.tobytes()
            elif isinstance(raw, (bytes, bytearray)):
                b = bytes(raw)
            else:
                try:
                    b = bytes(raw)
                except Exception:
                    b = b""
            last = b
            if len(b) >= min_len:
                if attempt > 0:
                    cprint(
                        f"  Frame {label}: OK after {attempt + 1} attempt(s), len={len(b)}",
                        C.GREEN,
                    )
                return b
            if attempt + 1 < max_attempts:
                logger.debug(
                    "Frame %s too short (len=%d), retry %d/%d",
                    label,
                    len(b),
                    attempt + 1,
                    max_attempts,
                )
                if delay > 0:
                    time.sleep(delay)

        logger.warning(
            "Frame %s still short after %d attempt(s) (len=%d, min=%d)",
            label,
            max_attempts,
            len(last),
            min_len,
        )
        return last

    def get_frames(self) -> dict[str, bytes]:
        """Return latest camera frames as {role_or_uuid: raw_bytes}.

        If background camera fetchers are running, returns cached bytes.
        Otherwise falls back to REST fetch for legacy sensor_ids / default camera.
        """
        if self.client is None or self.robot_twin is None:
            raise RuntimeError("CwProcessor.setup() must be called before get_frames()")

        if self.cameras:
            frames: dict[str, bytes] = {}
            with self._camera_lock:
                for role, binding in self.cameras.items():
                    frames[role] = binding.latest_bytes
            return frames

        frames = {}
        camera_role_by_uuid = {
            camera_twin_uuid: role
            for role, camera_twin_uuid in self.request.camera_endpoints_by_role.items()
        }

        for camera_twin_uuid in self.request.camera_twin_uuids:
            camera_twin = self.client.twins.get(camera_twin_uuid)
            role = camera_role_by_uuid.get(camera_twin_uuid)
            key = role if role else camera_twin_uuid
            frame_bytes = self._fetch_latest_frame_bytes(
                f"camera:{key}",
                camera_twin.get_latest_frame,
            )
            frames[key] = frame_bytes

        for sensor_id in self.request.camera_sensor_ids:
            sid = str(sensor_id)
            frame_bytes = self._fetch_latest_frame_bytes(
                f"sensor:{sid}",
                lambda s=sid: self.robot_twin.get_latest_frame(sensor_id=s),
            )
            frames[sid] = frame_bytes

        if not frames:
            frame_bytes = self._fetch_latest_frame_bytes(
                "robot_twin:default",
                self.robot_twin.get_latest_frame,
            )
            frames["default"] = frame_bytes

        return frames

    def publish_actions(self, actions: dict[str, float]) -> None:
        """Publish predicted joint targets via MQTT.

        Sends a single MQTT message with all joint positions.
        """
        if self.client is None:
            raise RuntimeError(
                "CwProcessor.setup() must be called before publish_actions()"
            )

        # Build publish kwargs
        publish_kwargs: dict[str, Any] = {
            "twin_uuid": self.request.robot_twin_uuid,
            "joint_positions": actions,
            "source_type": self.request.source_type or "tele",
            "source_subtype": self.source_subtype,
            "velocities": {name: 0.0 for name in actions},
            "efforts": {name: 0.0 for name in actions},
            "timestamp": self.now_fn(),
        }
        if self.request.workload_uuid:
            publish_kwargs["workload_uuid"] = self.request.workload_uuid

        # Log the action being published
        action_str = ", ".join(f"{k}={v:.4f}" for k, v in actions.items())
        logger.debug(
            "MQTT publish to %s: %s", self.request.robot_twin_uuid[:8], action_str
        )

        try:
            result = self.client.mqtt.update_joints_state(**publish_kwargs)
            # Log result on first call to verify it's working
            if not hasattr(self, "_first_publish_done"):
                print(f"First MQTT publish result: {result}", flush=True)
                self._first_publish_done = True
        except Exception as exc:
            logger.error("MQTT publish failed: %s", exc)
            raise

    def _execute_action_chunk(
        self,
        action_chunk: list[dict[str, float]],
        num_actions: int,
    ) -> tuple[int, dict[str, float]]:
        """Execute actions from a chunk sequentially with sleep between each.

        Args:
            action_chunk: List of action dicts from model prediction
            num_actions: How many actions to execute from the chunk

        Returns:
            (actions_executed, last_target_joints)
        """
        actions_to_execute = action_chunk[:num_actions]
        last_target: dict[str, float] = {}
        executed = 0

        # Gripper close verification settings
        gripper_key = "_6"  # Last joint is gripper
        gripper_close_threshold = 0.3  # Values below this = closing
        position_match_threshold = 0.01  # Must match within this tolerance
        max_gripper_retries = 10  # Max retries for gripper close

        total = len(actions_to_execute)
        for i, action_dict in enumerate(actions_to_execute):
            step_num = i + 1

            # Progress bar visualization
            if step_num == 1 or step_num == total or step_num % 5 == 0:
                bar_width = 20
                filled = int(bar_width * step_num / total)
                bar = "█" * filled + "░" * (bar_width - filled)
                joint_vals = [f"{v:+.2f}" for v in list(action_dict.values())[:6]]
                gripper = action_dict.get(gripper_key, 0)
                gripper_icon = "✊" if gripper < 0.3 else "✋"
                cprint(
                    f"  [{bar}] {step_num:2d}/{total}  {gripper_icon} [{', '.join(joint_vals)}]",
                    C.WHITE,
                )

            # Publish single action (all joints in one MQTT message)
            self.publish_actions(action_dict)
            last_target = action_dict
            executed += 1

            # Update internal state
            with self._joint_lock:
                self._current_joints.update(action_dict)

            # Sleep between actions (let robot move)
            if self.request.action_sleep_seconds > 0:
                time.sleep(self.request.action_sleep_seconds)

            # Gripper close verification: if gripper is closing, ensure it reaches target
            gripper_target = action_dict.get(gripper_key, 1.0)
            if gripper_target < gripper_close_threshold:
                retries = 0
                while retries < max_gripper_retries:
                    with self._joint_lock:
                        actual_gripper = self._current_joints.get(
                            gripper_key, gripper_target
                        )

                    error = abs(actual_gripper - gripper_target)
                    if error <= position_match_threshold:
                        break

                    retries += 1
                    self.publish_actions(action_dict)
                    time.sleep(self.request.action_sleep_seconds)

                if retries > 0:
                    cprint(
                        f"  ✊ Gripper close: {retries} retries (target={gripper_target:.2f})",
                        C.YELLOW,
                    )

        cprint(f"  ✓ Executed {executed} actions", C.GREEN)
        return executed, last_target

    def run(self) -> dict[str, Any]:
        """Execute the control loop: get observation -> predict -> execute N actions -> repeat.

        Control loop:
        1. Get current joint states (MQTT or REST)
        2. Fetch camera frames
        3. Call predict_fn to get action chunk (e.g., 50 actions)
        4. Execute first N actions (actions_per_cycle, default 25) one by one via MQTT
        5. If inference_loop=True and more steps needed, goto step 1

        Each action is sent as a single MQTT message with all joint positions.
        """
        if self.robot_twin is None:
            raise RuntimeError("CwProcessor.setup() must be called before run()")

        total_actions_executed = 0
        inference_cycles = 0
        target_joints: dict[str, float] = {}
        frame_summaries: list[dict[str, Any]] = []

        cprint("\n" + "=" * 50, C.CYAN, bold=True)
        cprint("  SMOLVLA CONTROL LOOP", C.CYAN, bold=True)
        cprint("=" * 50, C.CYAN, bold=True)
        cprint(f"  Max steps:        {self.request.max_steps}", C.WHITE)
        cprint(f"  Actions/cycle:    {self.request.actions_per_cycle}", C.WHITE)
        cprint(f"  Action sleep:     {self.request.action_sleep_seconds}s", C.WHITE)
        cprint(f"  Inference loop:   {self.request.inference_loop}", C.WHITE)
        mqtt_color = (
            C.GREEN if self.joint_state_source == "mqtt_subscription" else C.YELLOW
        )
        cprint(f"  Joint source:     {self.joint_state_source}", mqtt_color)
        print()

        # Warn if MQTT isn't working (actions won't reach robot)
        if self.joint_state_source == "rest_poll":
            cprint("WARNING: MQTT subscription failed!", C.RED, bold=True)
            cprint("         Actions may not reach the robot.", C.YELLOW)
            cprint("         Check CYBERWAVE_API_KEY is set and valid.\n", C.YELLOW)

        while total_actions_executed < self.request.max_steps:
            inference_cycles += 1
            remaining_steps = self.request.max_steps - total_actions_executed

            # How many actions to execute this cycle
            actions_this_cycle = min(self.request.actions_per_cycle, remaining_steps)

            # Cycle header
            progress = total_actions_executed / self.request.max_steps * 100
            cprint(f"\n{'─' * 50}", C.BLUE)
            cprint(
                f"  CYCLE {inference_cycles}  │  {total_actions_executed}/{self.request.max_steps} steps ({progress:.0f}%)",
                C.BLUE,
                bold=True,
            )
            cprint(f"{'─' * 50}", C.BLUE)

            # Get model inputs (frames + state + instruction)
            t_inputs0 = time.perf_counter()
            inputs = self.get_inputs()
            t_get_inputs = time.perf_counter() - t_inputs0
            logger.info(
                "Cycle %d: get_inputs done in %.4fs (cameras=%d)",
                inference_cycles,
                t_get_inputs,
                len(inputs.get("images") or {}),
            )

            # Record frame info and log initial joints (only on first cycle)
            if inference_cycles == 1:
                frame_summaries = [
                    {"key": key, "shape": img.shape}
                    for key, img in inputs["images"].items()
                ]
                init_vals = [
                    f"{v:.2f}" for v in list(self.initial_joints.values())[:6]
                ]
                cprint(f"  Initial: [{', '.join(init_vals)}]", C.DIM)

            # Show observation status
            frames_ok = len(inputs["images"])
            state = inputs["state"]
            cprint(
                f"  Cameras: {frames_ok} frames  │  State dim: {len(state)}",
                C.GREEN if frames_ok >= 3 and len(state) >= 6 else C.YELLOW,
            )

            state_vals = [f"{v:.2f}" for v in state[:6]]
            cprint(f"  State: [{', '.join(state_vals)}]", C.DIM)

            # Run inference - predict_fn returns raw action tensor
            cprint("  Running inference...", C.CYAN)
            logger.info("Cycle %d: calling predict_fn...", inference_cycles)
            t_pred0 = time.perf_counter()
            raw_actions = self.predict_fn(inputs)
            t_predict = time.perf_counter() - t_pred0
            logger.info(
                "Cycle %d: predict_fn returned in %.4fs | tensor shape=%s dtype=%s",
                inference_cycles,
                t_predict,
                getattr(raw_actions, "shape", None),
                getattr(raw_actions, "dtype", None),
            )

            # Convert raw tensor to list of joint dicts
            t_conv0 = time.perf_counter()
            action_chunk = self._convert_raw_actions(raw_actions)
            t_convert = time.perf_counter() - t_conv0
            if t_convert > 0.01:
                logger.info(
                    "Cycle %d: _convert_raw_actions took %.4fs (%d actions)",
                    inference_cycles,
                    t_convert,
                    len(action_chunk),
                )
            chunk_size = len(action_chunk)
            cprint(
                f"  Predicted {chunk_size} actions -> executing {actions_this_cycle}",
                C.GREEN,
            )

            # Execute first N actions from the chunk
            executed, target_joints = self._execute_action_chunk(
                action_chunk, actions_this_cycle
            )
            total_actions_executed += executed

            # If not in inference loop mode, we're done after one chunk
            if not self.request.inference_loop:
                logger.info("Single inference mode - completed after one chunk")
                break

            # Brief pause between inference cycles to let robot settle
            if (
                total_actions_executed < self.request.max_steps
                and self.request.inference_loop
            ):
                logger.info("Waiting before next inference cycle...")
                time.sleep(0.1)

        logger.info(
            "Control loop completed: %d actions executed in %d inference cycles",
            total_actions_executed,
            inference_cycles,
        )

        return {
            "status": "ok",
            "robot_twin_uuid": self.request.robot_twin_uuid,
            "instruction": self.request.instruction,
            "joint_state_source": self.joint_state_source,
            "frames": frame_summaries,
            "initial_joints": self.initial_joints,
            "current_joints": self.get_current_joints(),
            "target_joints": target_joints,
            "twin_calibration": self.twin_calibration,
            "calibration_robot_type": self.request.calibration_robot_type,
            "source_type": self.request.source_type,
            "mode": self.request.mode,
            "max_steps": self.request.max_steps,
            "actions_per_cycle": self.request.actions_per_cycle,
            "seed": self.request.seed,
            "steps_executed": total_actions_executed,
            "inference_cycles": inference_cycles,
        }

    def disconnect(self) -> None:
        """Stop camera fetchers and tear down SDK client."""
        self._camera_stop.set()
        for thread in self._camera_threads:
            thread.join(timeout=1.0)
        self._camera_threads.clear()
        self.cameras.clear()

        if self.client is not None:
            disconnect = getattr(self.client, "disconnect", None)
            if callable(disconnect):
                disconnect()
            self.client = None
            self.cw = None
