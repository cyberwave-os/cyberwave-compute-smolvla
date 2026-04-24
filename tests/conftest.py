"""Shared test fixtures for SmolVLA tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_inference_payload() -> dict[str, Any]:
    """Sample inference request payload."""
    return {
        "robot_twin_uuid": "b10e8ffa-f58c-49e0-a9c0-f76ffbed0356",
        "instruction": "pick up the red block",
        "camera_endpoints_by_role": {
            "primary_camera": "d62cf8b3-9533-496a-95a6-34e8188a885a",
            "wrist_camera": "69938649-de93-4db1-bd7e-c413114525c0",
        },
        "camera_twin_uuids": [
            "d62cf8b3-9533-496a-95a6-34e8188a885a",
            "69938649-de93-4db1-bd7e-c413114525c0",
        ],
        "twin_calibration": {
            "_1": {"min": -3.14, "max": 3.14},
            "_2": {"min": -1.57, "max": 1.57},
            "_3": {"min": -3.14, "max": 3.14},
            "_4": {"min": -3.14, "max": 3.14},
            "_5": {"min": -3.14, "max": 3.14},
            "_6": {"min": 0, "max": 1.0},
        },
        "calibration_robot_type": "follower",
        "max_steps": 100,
        "mode": "live",
        "wait_for_joint_update_seconds": 1.0,
        "action_sleep_seconds": 0.1,
        "actions_per_cycle": 25,
        "inference_loop": True,
    }


@pytest.fixture
def sample_inference_payload_camel_case() -> dict[str, Any]:
    """Sample inference request payload with camelCase keys."""
    return {
        "robotTwinUuid": "b10e8ffa-f58c-49e0-a9c0-f76ffbed0356",
        "instruction": "pick up the red block",
        "cameraEndpointsByRole": {
            "primary_camera": "cam-uuid-1",
        },
        "cameraTwinUuids": ["cam-uuid-1"],
        "twinCalibration": {
            "_1": {"min": -3.14, "max": 3.14},
        },
        "calibrationRobotType": "follower",
        "maxSteps": 50,
        "mode": "live",
        "waitForJointUpdateSeconds": 2.0,
        "actionSleepSeconds": 0.2,
        "actionsPerCycle": 10,
        "inferenceLoop": False,
    }


@pytest.fixture
def sample_training_payload() -> dict[str, Any]:
    """Sample training request payload."""
    return {
        "cyberwave_training_uuid": "eb903edf-dc3e-4e4d-939b-c515e596cda5",
        "cyberwave_token": "test-token-12345",
        "dataset_uuid": "8845dbf8-817c-4f61-9056-5981e1b42db0",
        "dataset_name": "test-dataset",
        "data_root_dir": "./datasets",
        "base_model": "lerobot/smolvla_base",
        "max_steps": 1000,
        "batch_size": 8,
        "lora_r": 16,
        "save_freq": 500,
        "log_freq": 50,
        "environment": "local",
    }


@pytest.fixture
def sample_training_payload_nested() -> dict[str, Any]:
    """Training payload with nested params structure."""
    return {
        "command": "training",
        "request_id": "test-request-id",
        "params": {
            "cyberwave_training_uuid": "eb903edf-dc3e-4e4d-939b-c515e596cda5",
            "dataset_uuid": "8845dbf8-817c-4f61-9056-5981e1b42db0",
            "policy": {
                "max_iterations": 5000,
            },
            "environment": "local",
        },
    }


@pytest.fixture
def sample_train_config() -> dict[str, Any]:
    """Sample SmolVLA train_config.json content with 6 joints."""
    return {
        "policy": {
            "input_features": {
                "observation.state": {"shape": [6], "type": "STATE"},
                "observation.images.cam_abc123": {
                    "shape": [3, 480, 640],
                    "type": "VISUAL",
                },
                "observation.images.cam_def456": {
                    "shape": [3, 480, 640],
                    "type": "VISUAL",
                },
                "observation.images.cam_ghi789": {
                    "shape": [3, 480, 640],
                    "type": "VISUAL",
                },
            },
            "output_features": {
                "action": {"shape": [6], "type": "ACTION"},
            },
        },
        "dataset": {
            "repo_id": "local/test-dataset",
        },
    }


@pytest.fixture
def sample_train_config_7_joints() -> dict[str, Any]:
    """Sample SmolVLA train_config.json content with 7 joints."""
    return {
        "policy": {
            "input_features": {
                "observation.state": {"shape": [7], "type": "STATE"},
                "observation.images.cam_abc123": {
                    "shape": [3, 480, 640],
                    "type": "VISUAL",
                },
            },
            "output_features": {
                "action": {"shape": [7], "type": "ACTION"},
            },
        },
    }


@pytest.fixture
def temp_checkpoint_dir(sample_train_config: dict[str, Any]) -> Path:
    """Create a temporary checkpoint directory with train_config.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint"
        checkpoint_path.mkdir(parents=True)

        config_path = checkpoint_path / "train_config.json"
        config_path.write_text(json.dumps(sample_train_config))

        yield checkpoint_path


@pytest.fixture
def temp_checkpoint_dir_nested(sample_train_config: dict[str, Any]) -> Path:
    """Create checkpoint with config in pretrained_model subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint"
        pretrained_path = checkpoint_path / "pretrained_model"
        pretrained_path.mkdir(parents=True)

        config_path = pretrained_path / "train_config.json"
        config_path.write_text(json.dumps(sample_train_config))

        yield checkpoint_path


@pytest.fixture
def mock_cyberwave_client() -> MagicMock:
    """Create a mock Cyberwave client."""
    client = MagicMock()

    # Mock twins API
    mock_twin = MagicMock()
    mock_twin.joints.get_all.return_value = {
        "_1": 0.0,
        "_2": -1.5,
        "_3": 1.5,
        "_4": 0.0,
        "_5": 0.0,
        "_6": 1.0,
    }
    mock_twin.get_latest_frame.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    mock_twin.get_calibration.return_value = {}

    client.twins.get.return_value = mock_twin

    # Mock MQTT
    client.mqtt.connect.return_value = None
    client.mqtt.subscribe_joint_states.return_value = None
    client.mqtt.update_joints_state.return_value = {"status": "ok"}

    # Mock affect method
    client.affect.return_value = client

    return client


@pytest.fixture
def mock_predict_fn() -> MagicMock:
    """Create a mock predict function."""
    import numpy as np

    mock_fn = MagicMock()
    # Return a mock action chunk: [1, 50, 6] for 50 timesteps, 6 joints
    mock_fn.return_value = np.random.randn(1, 50, 6).astype(np.float32)
    return mock_fn


@pytest.fixture
def sample_mqtt_joint_payload() -> dict[str, Any]:
    """Sample MQTT joint state payload."""
    return {
        "source_type": "edge_follower",
        "positions": {
            "_1": 0.01,
            "_2": -1.54,
            "_3": 1.48,
            "_4": 0.05,
            "_5": -0.02,
            "_6": 0.95,
        },
        "velocities": {
            "_1": 0.0,
            "_2": 0.0,
            "_3": 0.0,
            "_4": 0.0,
            "_5": 0.0,
            "_6": 0.0,
        },
        "efforts": {
            "_1": 0.0,
            "_2": 0.0,
            "_3": 0.0,
            "_4": 0.0,
            "_5": 0.0,
            "_6": 0.0,
        },
    }
