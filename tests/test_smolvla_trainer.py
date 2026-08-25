"""Tests for smolvla_trainer.build_pipeline_config."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBuildPipelineConfig:
    def test_output_dir_is_path_not_str(self) -> None:
        """Regression: lerobot builds checkpoint paths via
        ``cfg.output_dir / "checkpoints" / ...`` at save time. If output_dir is a
        plain ``str`` the first checkpoint crashes with
        ``TypeError: unsupported operand type(s) for /: 'str' and 'str'`` — after
        several training steps have already run (save_freq).
        """
        pytest.importorskip("lerobot")
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        from smolvla_trainer import SmolVLATrainer

        trainer = SmolVLATrainer()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"  # must NOT pre-exist (lerobot validate() guard)
            with patch.object(
                SmolVLATrainer, "_load_policy_config", return_value=SmolVLAConfig()
            ):
                cfg = trainer.build_pipeline_config(
                    {"max_steps": 5, "batch_size": 2},
                    dataset_root=Path(d) / "ds",
                    dataset_repo_id="local/x",
                    base_model_path="lerobot/smolvla_base",
                    output_dir=out,
                )

            assert isinstance(cfg.output_dir, Path), (
                "output_dir must be a Path so lerobot can build checkpoint paths"
            )
            # The exact operation lerobot performs at checkpoint time:
            _ = cfg.output_dir / "checkpoints" / "last"

    def _build(self, params: dict) -> Any:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        from smolvla_trainer import SmolVLATrainer

        trainer = SmolVLATrainer()
        with tempfile.TemporaryDirectory() as d:
            with patch.object(
                SmolVLATrainer, "_load_policy_config", return_value=SmolVLAConfig()
            ):
                return trainer.build_pipeline_config(
                    params,
                    dataset_root=Path(d) / "ds",
                    dataset_repo_id="local/x",
                    base_model_path="lerobot/smolvla_base",
                    output_dir=Path(d) / "run",
                )

    def test_explicit_max_steps_overrides_policy_max_iterations(self) -> None:
        """Regression: cyberwave.yml config.train max_steps=5 was ignored because
        ``policy.max_iterations`` (5000) always won. An explicit top-level
        max_steps must take precedence so operator smoke-test caps take effect.
        """
        pytest.importorskip("lerobot")
        cfg = self._build({"max_steps": 5, "policy": {"max_iterations": 5000}, "batch_size": 2})
        assert cfg.steps == 5

    def test_policy_max_iterations_used_when_no_explicit_steps(self) -> None:
        pytest.importorskip("lerobot")
        cfg = self._build({"policy": {"max_iterations": 1234}, "batch_size": 2})
        assert cfg.steps == 1234

    def test_steps_and_iterations_aliases(self) -> None:
        pytest.importorskip("lerobot")
        assert self._build({"steps": 7, "batch_size": 2}).steps == 7
        assert (
            self._build({"iterations": 9, "policy": {"max_iterations": 5000}, "batch_size": 2}).steps
            == 9
        )
