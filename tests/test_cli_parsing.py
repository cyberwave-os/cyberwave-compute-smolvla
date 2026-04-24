"""Tests for CLI argument parsing in deploy.py and train.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _has_torch() -> bool:
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_torch(), reason="torch not available")
class TestDeployCommandParsing:
    """Tests for deploy.py command-line parsing (requires torch)."""

    def test_deploy_main_no_args_returns_error(self) -> None:
        """Test that deploy.main() returns error with no arguments."""
        from deploy import main

        # main() with no args should return 1
        result = main([])
        assert result == 1

    def test_deploy_main_parses_json_file(
        self,
        tmp_path: Path,
        sample_inference_payload: dict[str, Any],
    ) -> None:
        """Test that deploy.main() parses JSON file argument."""
        from deploy import main

        # Create params file
        params_file = tmp_path / "params.json"
        params_file.write_text(json.dumps(sample_inference_payload))

        # Mock the heavy dependencies
        with patch("deploy.build_predict_fn") as mock_build:
            with patch("deploy.CwProcessor") as mock_processor_cls:
                mock_predict = MagicMock()
                mock_build.return_value = mock_predict

                mock_processor = MagicMock()
                mock_processor.resolver.training_camera_names = ["cam1"]
                mock_processor.camera_mapping = {"cam1": "primary"}
                mock_processor.run.return_value = {"status": "ok"}
                mock_processor_cls.return_value = mock_processor

                with patch.dict("os.environ", {"SMOLVLA_CHECKPOINT": "/fake/checkpoint"}):
                    result = main([str(params_file)])

                # Should have called parse_request_payload successfully
                mock_processor_cls.assert_called_once()
                call_args = mock_processor_cls.call_args
                assert call_args.args[0].robot_twin_uuid == sample_inference_payload["robot_twin_uuid"]

    def test_deploy_main_parses_inline_json(
        self,
        sample_inference_payload: dict[str, Any],
    ) -> None:
        """Test that deploy.main() parses inline JSON argument."""
        from deploy import main

        json_str = json.dumps(sample_inference_payload)

        with patch("deploy.build_predict_fn") as mock_build:
            with patch("deploy.CwProcessor") as mock_processor_cls:
                mock_predict = MagicMock()
                mock_build.return_value = mock_predict

                mock_processor = MagicMock()
                mock_processor.resolver.training_camera_names = []
                mock_processor.camera_mapping = {}
                mock_processor.run.return_value = {"status": "ok"}
                mock_processor_cls.return_value = mock_processor

                with patch.dict("os.environ", {"SMOLVLA_CHECKPOINT": "/fake/checkpoint"}):
                    result = main([json_str])

                mock_processor_cls.assert_called_once()

    def test_deploy_main_missing_checkpoint_returns_error(
        self,
        sample_inference_payload: dict[str, Any],
    ) -> None:
        """Test that missing SMOLVLA_CHECKPOINT env var causes error."""
        from deploy import main

        json_str = json.dumps(sample_inference_payload)

        # Ensure SMOLVLA_CHECKPOINT is not set
        with patch.dict("os.environ", {}, clear=True):
            result = main([json_str])

        assert result == 1


class TestTrainCommandParsing:
    """Tests for train.py command-line parsing."""

    def test_train_main_no_args_returns_error(self) -> None:
        """Test that train.main() returns error with no arguments."""
        # Import inside test to avoid import errors from missing lerobot
        try:
            from train import main

            result = main(["train.py"])  # Only program name, no params
            assert result == 1
        except ImportError:
            pytest.skip("train module dependencies not available")

    def test_train_main_parses_json_file(
        self,
        tmp_path: Path,
        sample_training_payload: dict[str, Any],
    ) -> None:
        """Test that train.main() parses JSON file argument."""
        try:
            from train import main
        except ImportError:
            pytest.skip("train module dependencies not available")

        # Create params file
        params_file = tmp_path / "params.json"
        params_file.write_text(json.dumps(sample_training_payload))

        # Mock CwTrainer to avoid actual training
        with patch("cw_trainer.CwTrainer") as mock_trainer_cls:
            mock_trainer = MagicMock()
            mock_trainer.run.return_value = {"status": "completed"}
            mock_trainer_cls.return_value = mock_trainer

            with patch("cw_trainer._get_trainer_registry") as mock_registry:
                mock_registry.return_value = {"smolvla": lambda: MagicMock()}

                result = main(["train.py", str(params_file)])

            # Verify CwTrainer was called with parsed params
            mock_trainer_cls.assert_called_once()
            call_kwargs = mock_trainer_cls.call_args
            assert call_kwargs.kwargs["model_slug"] == "smolvla"

    def test_train_main_handles_nested_params(
        self,
        tmp_path: Path,
        sample_training_payload_nested: dict[str, Any],
    ) -> None:
        """Test that train.main() handles nested params structure."""
        try:
            from train import main
        except ImportError:
            pytest.skip("train module dependencies not available")

        # Create params file with nested structure
        params_file = tmp_path / "params.json"
        params_file.write_text(json.dumps(sample_training_payload_nested))

        with patch("cw_trainer.CwTrainer") as mock_trainer_cls:
            mock_trainer = MagicMock()
            mock_trainer.run.return_value = {"status": "completed"}
            mock_trainer_cls.return_value = mock_trainer

            with patch("cw_trainer._get_trainer_registry") as mock_registry:
                mock_registry.return_value = {"smolvla": lambda: MagicMock()}

                result = main(["train.py", str(params_file)])

            # Verify the nested params were extracted
            mock_trainer_cls.assert_called_once()


class TestCwProcessorIntegration:
    """Integration tests for CwProcessor without heavy dependencies."""

    def test_processor_creation_with_mocks(
        self,
        sample_inference_payload: dict[str, Any],
        mock_cyberwave_client: MagicMock,
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor can be created with mocked dependencies."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=mock_cyberwave_client,
        )

        assert processor.request == request
        assert processor.model_slug == "smolvla"
        assert processor.predict_fn == mock_predict_fn

    def test_processor_action_conversion(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor action tensor conversion."""
        import numpy as np

        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        # Manually set joint configuration
        processor.joint_names = ["_1", "_2", "_3", "_4", "_5", "_6"]
        processor.num_joints = 6

        # Create a mock action tensor [1, 50, 6]
        raw_actions = np.array([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 50])

        action_chunk = processor._convert_raw_actions(raw_actions)

        assert len(action_chunk) == 50
        assert "_1" in action_chunk[0]
        assert action_chunk[0]["_1"] == pytest.approx(0.1)
        assert action_chunk[0]["_6"] == pytest.approx(0.6)

    def test_processor_action_conversion_unbatched(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor handles unbatched action tensors."""
        import numpy as np

        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        processor.joint_names = ["_1", "_2", "_3", "_4", "_5", "_6"]
        processor.num_joints = 6

        # Unbatched tensor [50, 6]
        raw_actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 50)

        action_chunk = processor._convert_raw_actions(raw_actions)

        assert len(action_chunk) == 50
        assert action_chunk[0]["_1"] == pytest.approx(0.1)

    def test_processor_derives_joint_names_from_twin_schema(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor derives joint names from twin's get_controllable_joint_names()."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        # Create mock twin that returns SO-101 style joint names
        mock_twin = MagicMock()
        mock_twin.get_controllable_joint_names.return_value = [
            "_1", "_2", "_3", "_4", "_5", "_6"
        ]

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        # Set robot_twin and call _derive_joint_names_from_twin
        processor.robot_twin = mock_twin
        processor._derive_joint_names_from_twin()

        # Verify joint names were derived from twin schema
        mock_twin.get_controllable_joint_names.assert_called_once()
        assert processor.joint_names == ["_1", "_2", "_3", "_4", "_5", "_6"]
        assert processor.num_joints == 6

    def test_processor_derives_joint_names_custom_robot(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor derives joint names from a robot with different joint names."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        # Create mock twin that returns custom joint names (e.g., a different robot)
        mock_twin = MagicMock()
        mock_twin.get_controllable_joint_names.return_value = [
            "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "gripper"
        ]

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        processor.robot_twin = mock_twin
        processor._derive_joint_names_from_twin()

        assert processor.joint_names == [
            "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "gripper"
        ]
        assert processor.num_joints == 6

    def test_processor_raises_when_robot_twin_is_none(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor raises error when robot_twin is None."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        # robot_twin is None by default
        with pytest.raises(RuntimeError, match="robot_twin is None"):
            processor._derive_joint_names_from_twin()

    def test_processor_raises_when_method_missing(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor raises error when get_controllable_joint_names method is missing."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        # Create mock twin with no get_controllable_joint_names method
        mock_twin = MagicMock(spec=[])  # Empty spec means no methods

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        processor.robot_twin = mock_twin

        with pytest.raises(RuntimeError, match="does not have get_controllable_joint_names"):
            processor._derive_joint_names_from_twin()

    def test_processor_raises_when_schema_returns_empty(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor raises error when twin schema returns empty list."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        # Create mock twin that returns empty list
        mock_twin = MagicMock()
        mock_twin.get_controllable_joint_names.return_value = []

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        processor.robot_twin = mock_twin

        with pytest.raises(RuntimeError, match="has no controllable joints"):
            processor._derive_joint_names_from_twin()

    def test_processor_validates_joint_count_matches_model(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor validates joint count matches model's expected state dim."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        # Create mock twin that returns 6 joints (matching the fixture config)
        mock_twin = MagicMock()
        mock_twin.get_controllable_joint_names.return_value = [
            "_1", "_2", "_3", "_4", "_5", "_6"
        ]

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        processor.robot_twin = mock_twin
        # Should not raise - 6 joints matches config's state_dim=6
        processor._derive_joint_names_from_twin()

        assert processor.num_joints == 6

    def test_processor_raises_on_joint_count_mismatch(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test CwProcessor raises error when joint count doesn't match model."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        # Create mock twin that returns 4 joints (mismatches config's state_dim=6)
        mock_twin = MagicMock()
        mock_twin.get_controllable_joint_names.return_value = [
            "_1", "_2", "_3", "_4"
        ]

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        processor.robot_twin = mock_twin

        with pytest.raises(RuntimeError, match="Joint count mismatch"):
            processor._derive_joint_names_from_twin()
