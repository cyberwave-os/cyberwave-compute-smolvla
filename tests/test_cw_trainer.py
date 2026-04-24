"""Tests for cw_trainer module - training orchestration and utilities."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cw_trainer import (
    CwTrainer,
    CyberwaveLogger,
    LogEvent,
    _get_api_base_url,
    _is_local_api_url,
    load_json_argument,
)


class TestLoadJsonArgument:
    """Tests for load_json_argument function."""

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading JSON from a file."""
        params = {"training_uuid": "test-123", "max_steps": 1000}
        json_file = tmp_path / "params.json"
        json_file.write_text(json.dumps(params))

        result = load_json_argument(str(json_file))

        assert result == params

    def test_load_from_inline_json(self) -> None:
        """Test loading from inline JSON string."""
        params = {"training_uuid": "inline-123"}
        result = load_json_argument(json.dumps(params))

        assert result == params

    def test_load_invalid_json_raises(self) -> None:
        """Test that invalid JSON raises error."""
        with pytest.raises(FileNotFoundError):
            load_json_argument("not valid json and not a file")


class TestGetApiBaseUrl:
    """Tests for _get_api_base_url function."""

    def test_local_environment(self) -> None:
        """Test URL for local environment."""
        assert _get_api_base_url("local") == "http://localhost:8000"
        assert _get_api_base_url("localhost") == "http://localhost:8000"
        assert _get_api_base_url("dev") == "http://localhost:8000"
        assert _get_api_base_url("development") == "http://localhost:8000"

    def test_production_environment(self) -> None:
        """Test URL for production environment."""
        assert _get_api_base_url("production") == "https://api.cyberwave.com"

    def test_default_environment(self) -> None:
        """Test URL for unknown environment defaults to dev."""
        assert _get_api_base_url("unknown") == "https://api-dev.cyberwave.com"

    def test_env_var_override(self) -> None:
        """Test that env var overrides default."""
        with patch.dict(os.environ, {"CYBERWAVE_API_URL": "https://custom.api.com"}):
            assert _get_api_base_url("production") == "https://custom.api.com"


class TestIsLocalApiUrl:
    """Tests for _is_local_api_url function."""

    def test_localhost(self) -> None:
        """Test localhost detection."""
        assert _is_local_api_url("http://localhost:8000") is True
        assert _is_local_api_url("https://localhost:8000") is True

    def test_127_0_0_1(self) -> None:
        """Test 127.0.0.1 detection."""
        assert _is_local_api_url("http://127.0.0.1:8000") is True

    def test_remote_url(self) -> None:
        """Test remote URL detection."""
        assert _is_local_api_url("https://api.cyberwave.com") is False
        assert _is_local_api_url("https://api-dev.cyberwave.com") is False


class TestLogEvent:
    """Tests for LogEvent dataclass."""

    def test_create_metrics_event(self) -> None:
        """Test creating a metrics log event."""
        event = LogEvent(
            event_type="metrics",
            payload={"step": 100, "loss": 0.5},
            step=100,
        )

        assert event.event_type == "metrics"
        assert event.payload["step"] == 100
        assert event.step == 100

    def test_create_checkpoint_event(self) -> None:
        """Test creating a checkpoint log event."""
        event = LogEvent(
            event_type="checkpoint",
            payload={"checkpoint_dir": "/path/to/checkpoint"},
        )

        assert event.event_type == "checkpoint"
        assert event.step is None


class TestCyberwaveLogger:
    """Tests for CyberwaveLogger class."""

    @pytest.fixture
    def mock_train_config(self) -> MagicMock:
        """Create a mock TrainPipelineConfig."""
        config = MagicMock()
        config.wandb = MagicMock()
        config.wandb.disable_artifact = False
        config.output_dir = "/tmp/output"
        config.steps = 10000
        return config

    def test_logger_init(self, mock_train_config: MagicMock) -> None:
        """Test logger initialization."""
        events = []
        logger = CyberwaveLogger(
            mock_train_config,
            on_log=lambda e: events.append(e),
            max_steps=5000,
        )

        assert logger._max_steps == 5000

    def test_log_dict_emits_event(self, mock_train_config: MagicMock) -> None:
        """Test that log_dict emits a log event."""
        events = []
        logger = CyberwaveLogger(
            mock_train_config,
            on_log=lambda e: events.append(e),
        )

        logger.log_dict({"loss": 0.5, "lr": 0.001}, step=100, mode="train")

        assert len(events) == 1
        assert events[0].event_type == "metrics"
        assert events[0].step == 100
        assert events[0].payload["step"] == 100
        assert events[0].payload["loss"] == 0.5

    def test_log_dict_formats_metric_names(self, mock_train_config: MagicMock) -> None:
        """Test that metric names are properly formatted."""
        events = []
        logger = CyberwaveLogger(
            mock_train_config,
            on_log=lambda e: events.append(e),
        )

        logger.log_dict({"policy_loss": 0.3}, step=50, mode="train")

        log_dict = events[0].payload["log"]
        assert "train/Policy Loss" in log_dict

    def test_log_dict_requires_step_or_custom_key(
        self, mock_train_config: MagicMock
    ) -> None:
        """Test that log_dict requires step or custom_step_key."""
        logger = CyberwaveLogger(mock_train_config, on_log=lambda e: None)

        with pytest.raises(ValueError, match="step or custom_step_key"):
            logger.log_dict({"loss": 0.5})

    def test_log_policy(self, mock_train_config: MagicMock, tmp_path: Path) -> None:
        """Test log_policy emits checkpoint event."""
        events = []
        logger = CyberwaveLogger(
            mock_train_config,
            on_log=lambda e: events.append(e),
        )

        checkpoint_dir = tmp_path / "checkpoint_1000"
        checkpoint_dir.mkdir()

        logger.log_policy(checkpoint_dir)

        assert len(events) == 1
        assert events[0].event_type == "checkpoint"

    def test_log_policy_disabled(self, mock_train_config: MagicMock) -> None:
        """Test log_policy is no-op when artifacts disabled."""
        mock_train_config.wandb.disable_artifact = True
        events = []
        logger = CyberwaveLogger(
            mock_train_config,
            on_log=lambda e: events.append(e),
        )

        logger.log_policy(Path("/some/path"))

        assert len(events) == 0


class TestCwTrainerInit:
    """Tests for CwTrainer initialization."""

    def test_init_with_valid_params(
        self, sample_training_payload: dict[str, Any]
    ) -> None:
        """Test initialization with valid parameters."""
        # Clear env vars to ensure params take precedence for non-local environments
        sample_training_payload["environment"] = "production"

        with patch("cw_trainer._get_trainer_registry") as mock_registry:
            mock_trainer = MagicMock()
            mock_registry.return_value = {"smolvla": lambda: mock_trainer}

            with patch.dict(os.environ, {}, clear=True):
                trainer = CwTrainer(sample_training_payload, model_slug="smolvla")

            assert trainer.training_uuid == "eb903edf-dc3e-4e4d-939b-c515e596cda5"
            assert trainer.token == "test-token-12345"
            assert trainer.environment == "production"
            assert trainer.model_slug == "smolvla"

    def test_init_unknown_model_slug_raises(
        self, sample_training_payload: dict[str, Any]
    ) -> None:
        """Test that unknown model_slug raises ValueError."""
        with patch("cw_trainer._get_trainer_registry") as mock_registry:
            mock_registry.return_value = {"smolvla": MagicMock}

            with pytest.raises(ValueError, match="Unknown model_slug"):
                CwTrainer(sample_training_payload, model_slug="unknown_model")

    def test_init_token_from_env(self) -> None:
        """Test token is read from environment if not in params."""
        params = {
            "cyberwave_training_uuid": "test-uuid",
            "environment": "local",
        }

        with patch.dict(os.environ, {"CYBERWAVE_API_KEY": "env-token"}):
            with patch("cw_trainer._get_trainer_registry") as mock_registry:
                mock_registry.return_value = {"smolvla": lambda: MagicMock()}

                trainer = CwTrainer(params, model_slug="smolvla")

                assert trainer.token == "env-token"


class TestCwTrainerNestedParams:
    """Tests for CwTrainer handling of nested params."""

    def test_nested_params_handling(self) -> None:
        """Test that nested params are properly merged."""
        # This tests the train.py behavior where nested params are merged
        params = {
            "command": "training",
            "params": {
                "cyberwave_training_uuid": "nested-uuid",
                "max_steps": 5000,
            },
        }

        # Simulate train.py's param merging logic
        if "params" in params and isinstance(params["params"], dict):
            nested = params.pop("params")
            for key, value in nested.items():
                if key not in params:
                    params[key] = value

        assert params.get("cyberwave_training_uuid") == "nested-uuid"
        assert params.get("max_steps") == 5000


class TestCwTrainerResults:
    """Tests for CwTrainer results handling."""

    def test_results_folder_default(
        self, sample_training_payload: dict[str, Any]
    ) -> None:
        """Test default results folder."""
        with patch("cw_trainer._get_trainer_registry") as mock_registry:
            mock_registry.return_value = {"smolvla": lambda: MagicMock()}

            trainer = CwTrainer(sample_training_payload, model_slug="smolvla")

            assert trainer.results_folder == Path("./runs/artifacts")

    def test_results_folder_custom(
        self, sample_training_payload: dict[str, Any]
    ) -> None:
        """Test custom results folder."""
        sample_training_payload["results_folder"] = "/custom/results"

        with patch("cw_trainer._get_trainer_registry") as mock_registry:
            mock_registry.return_value = {"smolvla": lambda: MagicMock()}

            trainer = CwTrainer(sample_training_payload, model_slug="smolvla")

            assert trainer.results_folder == Path("/custom/results")
