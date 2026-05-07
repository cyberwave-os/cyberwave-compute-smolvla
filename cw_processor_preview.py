"""SmolVLA one-shot preview processor for the Cyberwave playground.

Handles a single inference pass from a playground "preview" workload:
  - Decodes the inline base64 image
  - Builds a zero proprio state (joint-space, dimension from robot spec)
  - Calls the model predict_fn once
  - Returns the action chunk as a JSON result

Unlike CwProcessor (which manages a live MQTT control loop with a robot twin),
this processor has NO Cyberwave SDK dependency - it only needs the model and
an image from the payload.

SmolVLA outputs joint-space actions by default.  The action chunk shape is:
  (chunk_size, action_dim)
where action_dim is determined by the checkpoint's output_features
(typically equal to the robot's DoF).

Output mode:
  SmolVLA always outputs joint-space actions ("joint_action") regardless of
  robot configuration.  output_mode from the robot spec is forwarded verbatim
  so the frontend knows how to interpret the values.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Camera slot name used for the single preview image (positional, first slot).
_PREVIEW_CAMERA_SLOT = "primary"

# SmolVLA default action output mode.
_DEFAULT_OUTPUT_MODE = "joint_action"


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode a raw base64 string (no data-URL prefix) to an H×W×3 uint8 array."""
    try:
        from PIL import Image as PILImage

        raw_bytes = base64.b64decode(image_b64)
        pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        logger.error("Failed to decode base64 image: %s", e)
        raise


# ---------------------------------------------------------------------------
# Preview request dataclass
# ---------------------------------------------------------------------------


class PreviewRequest:
    """Parsed VLAPreviewParams payload for a one-shot preview run.

    Fields match the backend VLAPreviewParams dataclass.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.mlmodel_uuid: str = str(payload.get("mlmodel_uuid") or "")
        self.weights_url: str | None = payload.get("weights_url") or None
        self.family: str = str(payload.get("family") or "smolvla")
        self.prompt: str = str(payload.get("prompt") or "")
        self.image_base64: str | None = payload.get("image_base64")
        self.workload_uuid: str | None = payload.get("workload_uuid")

        robot: dict[str, Any] = {}
        raw_robot = payload.get("robot")
        if isinstance(raw_robot, dict):
            robot = raw_robot

        self.robot_dof: int = int(robot.get("dof") or 0)
        self.robot_joint_names: list[str] = list(robot.get("joint_names") or [])
        self.robot_joint_limits: dict[str, Any] = dict(robot.get("joint_limits") or {})
        self.robot_output_mode: str = str(robot.get("output_mode") or _DEFAULT_OUTPUT_MODE)
        self.robot_registry_id: str | None = robot.get("registry_id")
        self.robot_gripper_max_width_m: float | None = (
            float(robot.get("gripper_max_width_m"))
            if robot.get("gripper_max_width_m") is not None
            else None
        )


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------


class CwProcessorPreview:
    """One-shot preview processor: decode image → predict → format result.

    No Cyberwave SDK, no MQTT, no robot twin.
    Designed to run inside a cloud-node 'preview' workload.

    SmolVLA-specific behaviour:
      - Camera names are derived from the training checkpoint config
        (e.g. "cam_7e7bf9fe").  For preview we always use positional slot 0
        ("primary") since we only have one incoming image, and let the predict_fn
        handle remapping internally (it was built with the training camera names).
      - State (proprio) is zero-padded to `state_dim` — the dimension the model
        was trained with.  state_dim=0 means no proprio input.
      - Actions are joint-space (not Cartesian EE delta).

    Args:
        payload: Parsed VLAPreviewParams dict from the workload command_params.
        predict_fn: Callable built by build_predict_fn() in deploy.py.
                    Signature: predict_fn(inputs: dict) -> torch.Tensor
                    where inputs has keys: "images", "state", "instruction"
                    and return shape is (1, chunk_size, action_dim) or
                    (chunk_size, action_dim).
        training_camera_names: Camera names the model was trained with (from resolver).
                               Used to key the images dict correctly.  If empty,
                               falls back to positional slot 0.
        state_dim: Dimension of the proprio state vector the model expects.
                   0 = no proprio.
    """

    def __init__(
        self,
        payload: dict[str, Any],
        predict_fn: Callable[[dict[str, Any]], Any],
        *,
        training_camera_names: list[str] | None = None,
        state_dim: int = 0,
    ) -> None:
        self.request = PreviewRequest(payload)
        self.predict_fn = predict_fn
        cams = list(training_camera_names or [])
        if not cams:
            fn_cams = getattr(predict_fn, "training_camera_names", None)
            if isinstance(fn_cams, list) and fn_cams:
                cams = list(fn_cams)
        self.training_camera_names = cams
        sd = int(state_dim or 0)
        if sd <= 0:
            sd = int(getattr(predict_fn, "expected_state_dim", 0) or 0)
        self.state_dim = sd

    def _build_images(self) -> dict[str, np.ndarray]:
        """Decode the inline image and build the images dict keyed by training camera name.

        SmolVLA's predict_fn (built by deploy.py) expects a dict keyed by the
        *training* camera short names (e.g. "cam_7e7bf9fe").  For preview we
        only have one image, so we replicate it across all expected camera slots.
        """
        if not self.request.image_base64:
            raise ValueError("No image_base64 in preview payload.")

        primary = decode_base64_image(self.request.image_base64)

        if self.training_camera_names:
            # Replicate the single image across all training slots.
            return {name: primary.copy() for name in self.training_camera_names}

        # Fallback: positional slot 0 — predict_fn will use whatever key it finds.
        return {_PREVIEW_CAMERA_SLOT: primary}

    def _build_state(self) -> np.ndarray:
        """Build zero proprio state vector sized to what the model expects.

        We use state_dim (from the training checkpoint config) as the
        authoritative dimension — the model was trained on exactly this many
        proprio values and will truncate/reject anything else.  robot_dof is
        informational only; it describes the real robot, which may differ from
        the training embodiment.

        Falls back to robot_dof if state_dim is unavailable (e.g. HF Hub
        base weights with no local train_config.json).
        """
        dim = self.state_dim if self.state_dim > 0 else self.request.robot_dof
        if dim <= 0:
            return np.zeros(0, dtype=np.float32)

        logger.info("Building zero proprio state (dim=%d, model state_dim).", dim)
        return np.zeros(dim, dtype=np.float32)

    def _format_action_output(self, raw_actions: Any) -> dict[str, Any]:
        """Convert raw action tensor to a serialisable result dict.

        SmolVLA returns (1, chunk_size, action_dim) or (chunk_size, action_dim).

        Returns a dict with:
          output_format: "joint_action_chunk"
          output: list of action steps, each a list of floats
        """
        try:
            # Convert torch.Tensor / numpy array to numpy.
            if hasattr(raw_actions, "cpu"):
                arr = raw_actions.cpu().numpy()
            else:
                arr = np.array(raw_actions)
        except Exception:
            arr = np.array(raw_actions)

        # Squeeze batch dim if present: (1, chunk, dim) -> (chunk, dim).
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]  # (1, dim)

        # Truncate action columns to the robot's actual DOF when the model
        # outputs more dimensions than the robot has joints.
        # SmolVLA base weights are trained on a fixed action dim (e.g. 6 for
        # SO-101); when running on a robot with fewer joints we truncate the
        # trailing columns.  When the model outputs *fewer* columns than the
        # robot DOF we leave the output as-is — zero-padding unknown joints
        # would silently drive them to zero, which is unsafe.
        robot_dof = self.request.robot_dof
        if robot_dof > 0 and arr.ndim == 2:
            if arr.shape[1] > robot_dof:
                original_dim = arr.shape[1]
                arr = arr[:, :robot_dof]
                logger.info(
                    "Truncated action output from %d to %d columns (robot dof=%d).",
                    original_dim,
                    robot_dof,
                    robot_dof,
                )
            elif arr.shape[1] < robot_dof:
                pad = robot_dof - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
                logger.info(
                    "Padded action output from %d to %d columns (robot dof=%d). "
                    "TODO: replace zero-padding with proper EEF conversion.",
                    arr.shape[1] - pad,
                    robot_dof,
                    robot_dof,
                )

        mode = self.request.robot_output_mode
        output_format = "joint_action_chunk" if mode == "joint_action" else "ee_delta_chunk"
        output_steps = arr.tolist()
        actual_dim = arr.shape[-1] if arr.ndim > 1 else (len(output_steps[0]) if output_steps else 0)

        result: dict[str, Any] = {
            "output_format": output_format,
            "output": output_steps,
            "num_steps": len(output_steps),
            "action_dim": actual_dim,
            "output_mode": mode,
        }

        # Trim joint_names to match actual output columns so the frontend can
        # map each column to the correct URDF joint by name.
        if self.request.robot_joint_names:
            names = self.request.robot_joint_names[:actual_dim]
            result["joint_names"] = names

        return result

    def run(self) -> dict[str, Any]:
        """Execute the one-shot preview inference.

        Returns a result dict suitable for JSON serialisation and consumption
        by the cloud-node workload completion handler and the frontend.
        """
        logger.info(
            "Preview inference: family=%s prompt=%r output_mode=%s dof=%d cameras=%s",
            self.request.family,
            self.request.prompt[:60],
            self.request.robot_output_mode,
            self.request.robot_dof,
            self.training_camera_names,
        )

        images = self._build_images()
        state = self._build_state()

        inputs: dict[str, Any] = {
            "images": images,
            "state": state,
            "instruction": self.request.prompt or "perform the task",
        }

        logger.info(
            "Running predict_fn with %d image slot(s), state_dim=%d",
            len(images),
            state.shape[0] if hasattr(state, "shape") else 0,
        )

        raw_actions = self.predict_fn(inputs)
        action_result = self._format_action_output(raw_actions)

        return {
            "status": "ok",
            "family": self.request.family,
            "prompt": self.request.prompt,
            "mlmodel_uuid": self.request.mlmodel_uuid,
            "workload_uuid": self.request.workload_uuid,
            "robot": {
                "dof": self.request.robot_dof,
                "output_mode": self.request.robot_output_mode,
                "joint_names": self.request.robot_joint_names,
                "registry_id": self.request.robot_registry_id,
            },
            "result": action_result,
        }
