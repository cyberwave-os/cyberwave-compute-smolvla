"""SmolVLA-specific metadata resolver.

Handles training config loading and camera mapping for SmolVLA models.
No torch, no Cyberwave imports. Does not load the model itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from base_resolver import BaseVLAResolver

logger = logging.getLogger(__name__)


class SmolVLAResolver(BaseVLAResolver):
    """Resolver for SmolVLA model-specific metadata.

    Responsible for:
    - Loading and parsing train_config.json from checkpoint
    - Extracting camera names used during training
    - Building mappings between training camera names and runtime identifiers
    """

    MODEL_SLUG = "smolvla"

    def __init__(self, checkpoint: str) -> None:
        """Initialize resolver with checkpoint path.

        Args:
            checkpoint: Local path to SmolVLA checkpoint directory.
                       Should contain pretrained_model/train_config.json or train_config.json.
        """
        self.checkpoint = checkpoint
        self.training_config = self._load_training_config()
        self.training_camera_names = self._extract_camera_names()
        self.expected_state_dim = self._extract_state_dim()
        self.expected_action_dim = self._extract_action_dim()

    def _load_training_config(self) -> dict[str, Any] | None:
        """Load train_config.json from checkpoint directory.

        Searches for config in common locations:
        - {checkpoint}/train_config.json
        - {checkpoint}/pretrained_model/train_config.json
        """
        checkpoint_path = Path(self.checkpoint)
        candidates = [
            checkpoint_path / "train_config.json",
            checkpoint_path / "pretrained_model" / "train_config.json",
        ]

        for config_path in candidates:
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    logger.info("Loaded training config from %s", config_path)
                    return config
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to load training config from %s: %s", config_path, e)

        logger.warning("No train_config.json found in checkpoint: %s", self.checkpoint)
        return None

    def _extract_camera_names(self) -> list[str]:
        """Extract camera names from training config's input_features.

        Returns list of camera short names (without 'observation.images.' prefix),
        in the order they appear in the config.

        Example: ["cam_7e7bf9fe", "cam_9fcace87", "cam_a6f944f4"]
        """
        if not self.training_config:
            return []

        input_features = self.training_config.get("policy", {}).get("input_features", {})
        camera_names = []

        for key in input_features:
            if key.startswith("observation.images."):
                short_name = key.split("observation.images.")[-1]
                camera_names.append(short_name)

        logger.info("Extracted %d camera names from training config: %s", len(camera_names), camera_names)
        return camera_names

    def _extract_state_dim(self) -> int:
        """Extract expected state dimension from training config.

        The state dimension is the number of joint positions the model expects
        as input (observation.state shape).

        Returns:
            Expected state dimension, or 0 if not found.
        """
        if not self.training_config:
            return 0

        input_features = self.training_config.get("policy", {}).get("input_features", {})
        state_feature = input_features.get("observation.state", {})

        # Shape can be a list [dim] or tuple (dim,)
        shape = state_feature.get("shape", [])
        if isinstance(shape, (list, tuple)) and len(shape) > 0:
            dim = int(shape[0])
            logger.info("Extracted expected_state_dim=%d from training config", dim)
            return dim

        return 0

    def _extract_action_dim(self) -> int:
        """Extract expected action dimension from training config.

        The action dimension is the number of joint positions the model outputs.

        Returns:
            Expected action dimension, or 0 if not found.
        """
        if not self.training_config:
            return 0

        output_features = self.training_config.get("policy", {}).get("output_features", {})
        action_feature = output_features.get("action", {})

        # Shape can be a list [dim] or tuple (dim,)
        shape = action_feature.get("shape", [])
        if isinstance(shape, (list, tuple)) and len(shape) > 0:
            dim = int(shape[0])
            logger.info("Extracted expected_action_dim=%d from training config", dim)
            return dim

        return 0

    def get_expected_state_dim(self) -> int:
        """Return expected state dimension (number of input joints)."""
        return self.expected_state_dim

    def get_expected_action_dim(self) -> int:
        """Return expected action dimension (number of output joints)."""
        return self.expected_action_dim

    def build_camera_mapping(
        self,
        runtime_cameras: dict[str, str] | list[str],
    ) -> dict[str, str]:
        """Map training camera names to runtime camera identifiers.

        Since training camera names are typically UUIDs or non-semantic IDs,
        we map by position:
        - If runtime provides a dict (role -> uuid), we use the KEYS (roles) since
          CwProcessor.get_frames() returns frames keyed by role, not UUID
        - If runtime provides a list, we use them directly in order

        Args:
            runtime_cameras: Either dict[role, uuid] or list[uuid] from runtime request

        Returns:
            Dict mapping training camera name -> runtime frame key
            (the key that will appear in the frames dict from CwProcessor.get_frames())
        """
        mapping: dict[str, str] = {}

        if isinstance(runtime_cameras, dict):
            runtime_list = list(runtime_cameras.keys())
        else:
            runtime_list = list(runtime_cameras)

        for i, training_name in enumerate(self.training_camera_names):
            if i < len(runtime_list):
                mapping[training_name] = runtime_list[i]
                logger.info("Camera mapping: training '%s' <- runtime '%s'", training_name, runtime_list[i])
            else:
                logger.warning(
                    "No runtime camera for training camera '%s' (index %d). "
                    "Expected %d cameras, got %d.",
                    training_name, i, len(self.training_camera_names), len(runtime_list)
                )

        return mapping

    def get_expected_camera_count(self) -> int:
        """Return number of cameras expected by the model."""
        return len(self.training_camera_names)
