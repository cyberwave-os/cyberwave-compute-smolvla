"""Tests for SmolVLAResolver - camera mapping and config loading."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from smolvla_resolver import SmolVLAResolver


class TestSmolVLAResolverInit:
    """Tests for SmolVLAResolver initialization."""

    def test_init_with_valid_config(self, temp_checkpoint_dir: Path) -> None:
        """Test initialization with valid train_config.json."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))

        assert resolver.checkpoint == str(temp_checkpoint_dir)
        assert resolver.training_config is not None
        assert len(resolver.training_camera_names) == 3

    def test_init_with_nested_config(self, temp_checkpoint_dir_nested: Path) -> None:
        """Test initialization with config in pretrained_model subdirectory."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir_nested))

        assert resolver.training_config is not None
        assert len(resolver.training_camera_names) == 3

    def test_init_without_config(self, tmp_path: Path) -> None:
        """Test initialization without train_config.json."""
        empty_checkpoint = tmp_path / "empty_checkpoint"
        empty_checkpoint.mkdir()

        resolver = SmolVLAResolver(str(empty_checkpoint))

        assert resolver.training_config is None
        assert resolver.training_camera_names == []

    def test_model_slug(self) -> None:
        """Test that MODEL_SLUG is correctly set."""
        assert SmolVLAResolver.MODEL_SLUG == "smolvla"


class TestExtractCameraNames:
    """Tests for camera name extraction from training config."""

    def test_extract_camera_names(self, temp_checkpoint_dir: Path) -> None:
        """Test extracting camera names from input_features."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))

        assert "cam_abc123" in resolver.training_camera_names
        assert "cam_def456" in resolver.training_camera_names
        assert "cam_ghi789" in resolver.training_camera_names

    def test_camera_names_order_preserved(
        self, sample_train_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that camera name order matches config order."""
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        # Create config with specific camera order
        config = {
            "policy": {
                "input_features": {
                    "observation.images.first_cam": {"shape": [3, 480, 640]},
                    "observation.images.second_cam": {"shape": [3, 480, 640]},
                    "observation.images.third_cam": {"shape": [3, 480, 640]},
                }
            }
        }
        (checkpoint / "train_config.json").write_text(json.dumps(config))

        resolver = SmolVLAResolver(str(checkpoint))

        # Note: dict order is preserved in Python 3.7+
        assert resolver.training_camera_names[0] == "first_cam"
        assert resolver.training_camera_names[1] == "second_cam"
        assert resolver.training_camera_names[2] == "third_cam"

    def test_extract_ignores_non_image_features(self, tmp_path: Path) -> None:
        """Test that non-image features are ignored."""
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        config = {
            "policy": {
                "input_features": {
                    "observation.state": {"shape": [6], "type": "STATE"},
                    "observation.images.camera": {"shape": [3, 480, 640]},
                    "some.other.feature": {"shape": [10]},
                }
            }
        }
        (checkpoint / "train_config.json").write_text(json.dumps(config))

        resolver = SmolVLAResolver(str(checkpoint))

        assert resolver.training_camera_names == ["camera"]


class TestBuildCameraMapping:
    """Tests for building camera mappings."""

    def test_mapping_from_dict(self, temp_checkpoint_dir: Path) -> None:
        """Test mapping from dict of role -> uuid."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))

        runtime_cameras = {
            "primary_camera": "uuid-1",
            "wrist_camera": "uuid-2",
            "overhead_camera": "uuid-3",
        }
        mapping = resolver.build_camera_mapping(runtime_cameras)

        # Mapping should be training_name -> runtime_key (role)
        assert len(mapping) == 3
        # Training cameras are mapped to runtime roles by position
        assert mapping["cam_abc123"] == "primary_camera"
        assert mapping["cam_def456"] == "wrist_camera"
        assert mapping["cam_ghi789"] == "overhead_camera"

    def test_mapping_from_list(self, temp_checkpoint_dir: Path) -> None:
        """Test mapping from list of uuids."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))

        runtime_cameras = ["camera-uuid-1", "camera-uuid-2", "camera-uuid-3"]
        mapping = resolver.build_camera_mapping(runtime_cameras)

        assert len(mapping) == 3
        assert mapping["cam_abc123"] == "camera-uuid-1"
        assert mapping["cam_def456"] == "camera-uuid-2"
        assert mapping["cam_ghi789"] == "camera-uuid-3"

    def test_mapping_with_fewer_runtime_cameras(
        self, temp_checkpoint_dir: Path
    ) -> None:
        """Test mapping when fewer runtime cameras than training cameras."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))

        # Only provide 2 cameras for 3 training cameras
        runtime_cameras = ["camera-1", "camera-2"]
        mapping = resolver.build_camera_mapping(runtime_cameras)

        # Should map what's available
        assert len(mapping) == 2
        assert "cam_abc123" in mapping
        assert "cam_def456" in mapping
        assert "cam_ghi789" not in mapping

    def test_mapping_with_more_runtime_cameras(
        self, temp_checkpoint_dir: Path
    ) -> None:
        """Test mapping when more runtime cameras than training cameras."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))

        runtime_cameras = ["cam-1", "cam-2", "cam-3", "cam-4", "cam-5"]
        mapping = resolver.build_camera_mapping(runtime_cameras)

        # Should only use what's needed
        assert len(mapping) == 3

    def test_mapping_empty_runtime_cameras(self, temp_checkpoint_dir: Path) -> None:
        """Test mapping with empty runtime cameras."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))

        mapping = resolver.build_camera_mapping([])
        assert mapping == {}

    def test_mapping_no_training_cameras(self, tmp_path: Path) -> None:
        """Test mapping when resolver has no training cameras."""
        checkpoint = tmp_path / "empty_checkpoint"
        checkpoint.mkdir()

        resolver = SmolVLAResolver(str(checkpoint))
        mapping = resolver.build_camera_mapping(["cam-1", "cam-2"])

        assert mapping == {}


class TestGetExpectedCameraCount:
    """Tests for get_expected_camera_count method."""

    def test_count_with_cameras(self, temp_checkpoint_dir: Path) -> None:
        """Test camera count with valid config."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))
        assert resolver.get_expected_camera_count() == 3

    def test_count_without_config(self, tmp_path: Path) -> None:
        """Test camera count without config."""
        checkpoint = tmp_path / "empty"
        checkpoint.mkdir()

        resolver = SmolVLAResolver(str(checkpoint))
        assert resolver.get_expected_camera_count() == 0


class TestGetExpectedStateDim:
    """Tests for get_expected_state_dim method."""

    def test_state_dim_with_config(self, temp_checkpoint_dir: Path) -> None:
        """Test state dimension extraction from config."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))
        assert resolver.get_expected_state_dim() == 6

    def test_state_dim_without_config(self, tmp_path: Path) -> None:
        """Test state dimension returns 0 without config."""
        checkpoint = tmp_path / "empty"
        checkpoint.mkdir()

        resolver = SmolVLAResolver(str(checkpoint))
        assert resolver.get_expected_state_dim() == 0

    def test_state_dim_different_value(
        self, sample_train_config_7_joints: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test state dimension with different joint count."""
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "train_config.json").write_text(
            json.dumps(sample_train_config_7_joints)
        )

        resolver = SmolVLAResolver(str(checkpoint))
        assert resolver.get_expected_state_dim() == 7


class TestGetExpectedActionDim:
    """Tests for get_expected_action_dim method."""

    def test_action_dim_with_config(self, temp_checkpoint_dir: Path) -> None:
        """Test action dimension extraction from config."""
        resolver = SmolVLAResolver(str(temp_checkpoint_dir))
        assert resolver.get_expected_action_dim() == 6

    def test_action_dim_without_config(self, tmp_path: Path) -> None:
        """Test action dimension returns 0 without config."""
        checkpoint = tmp_path / "empty"
        checkpoint.mkdir()

        resolver = SmolVLAResolver(str(checkpoint))
        assert resolver.get_expected_action_dim() == 0

    def test_action_dim_different_value(
        self, sample_train_config_7_joints: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test action dimension with different joint count."""
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "train_config.json").write_text(
            json.dumps(sample_train_config_7_joints)
        )

        resolver = SmolVLAResolver(str(checkpoint))
        assert resolver.get_expected_action_dim() == 7
